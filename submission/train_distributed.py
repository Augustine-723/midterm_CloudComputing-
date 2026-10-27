#!/usr/bin/env python3
"""
submission/train_distributed.py
Advanced Bonus Deliverable: Synchronous Data Parallelism Training with tf.distribute.MirroredStrategy.

Technical Specifications:
- Device Virtualization: Splits single physical CPU into 2 virtual Logical CPU devices
- Strategy: tf.distribute.MirroredStrategy(devices=['/device:CPU:0', '/device:CPU:1'])
- Replicas in Sync: 2 (Global Batch Size 64 -> 32 samples per replica)
- Low-Level Step: tf.GradientTape captured in step_fn, executed via strategy.run()
- Loss Scaling: tf.nn.compute_average_loss(per_example_loss, global_batch_size=64)
- Synchronous All-Reduce gradient application
- Strictly avoids model.fit(), model.evaluate(), and model.predict()
"""

import os
import sys
import time
import argparse
import numpy as np

# 1. Device Virtualization before TF runtime initialization
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

cpus = tf.config.list_physical_devices("CPU")
if cpus:
    try:
        tf.config.set_logical_device_configuration(
            cpus[0],
            [tf.config.LogicalDeviceConfiguration(), tf.config.LogicalDeviceConfiguration()]
        )
    except RuntimeError:
        pass

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

def resolve_data_dir():
    candidates = [
        os.path.join(SCRIPT_DIR, "dataset"),
        os.path.join(SCRIPT_DIR, "..", "dataset"),
        "/app/dataset"
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.npz")):
            return c
    raise FileNotFoundError("Could not locate train.npz in candidate directories: " + str(candidates))

DATA_DIR = resolve_data_dir()
DEFAULT_MODEL_OUT = os.path.join(SCRIPT_DIR, "models", "distributed_cpu_model.h5")


def load_datasets():
    train_file = os.path.join(DATA_DIR, "train.npz")
    val_file = os.path.join(DATA_DIR, "validation.npz")

    train_raw = np.load(train_file)
    val_raw = np.load(val_file)

    x_train = train_raw["data"].astype(np.float32).reshape(-1, 25)
    y_train = train_raw["label"].astype(np.int32).flatten()

    x_val = val_raw["data"].astype(np.float32).reshape(-1, 25)
    y_val = val_raw["label"].astype(np.int32).flatten()

    return (x_train, y_train), (x_val, y_val)


def main():
    parser = argparse.ArgumentParser(description="Synchronous Data Parallelism Training")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--global_batch_size", type=int, default=64, help="Global batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output", type=str, default=DEFAULT_MODEL_OUT, help="Saved model path")
    args = parser.parse_args()

    tf.random.set_seed(42)
    np.random.seed(42)

    # Strategy setup
    logical_cpus = tf.config.list_logical_devices("CPU")
    device_names = [d.name for d in logical_cpus]
    strategy = tf.distribute.MirroredStrategy(devices=device_names)
    num_replicas = strategy.num_replicas_in_sync

    print("=" * 70)
    print(" ADVANCED BONUS: SYNCHRONOUS DATA PARALLELISM (MIRROREDSTRATEGY)")
    print("=" * 70)
    print(f"Logical CPU Devices:   {device_names}")
    print(f"Replicas in Sync:      {num_replicas}")
    print(f"Global Batch Size:     {args.global_batch_size} ({args.global_batch_size // num_replicas} per replica)")

    (x_train, y_train), (x_val, y_val) = load_datasets()

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=10000, seed=42)
        .batch(args.global_batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(args.global_batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    dist_train_ds = strategy.experimental_distribute_dataset(train_ds)
    dist_val_ds = strategy.experimental_distribute_dataset(val_ds)

    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(25,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(5, activation="softmax")
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )

        train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")
        val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
        val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="val_acc")

        def compute_loss(labels, preds):
            per_example = loss_object(labels, preds)
            return tf.nn.compute_average_loss(per_example, global_batch_size=args.global_batch_size)

    @tf.function
    def train_step(dist_inputs):
        def step_fn(inputs):
            f, l = inputs
            with tf.GradientTape() as tape:
                preds = model(f, training=True)
                loss = compute_loss(l, preds)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss_metric.update_state(loss * num_replicas)
            train_acc_metric.update_state(l, preds)
            return loss

        per_rep = strategy.run(step_fn, args=(dist_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_rep, axis=None)

    @tf.function
    def val_step(dist_inputs):
        def val_step_fn(inputs):
            f, l = inputs
            preds = model(f, training=False)
            per_example = loss_object(l, preds)
            val_loss_metric.update_state(per_example)
            val_acc_metric.update_state(l, preds)

        strategy.run(val_step_fn, args=(dist_inputs,))

    best_val_loss = float("inf")
    patience = 0
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss_metric.reset_states()
        train_acc_metric.reset_states()
        val_loss_metric.reset_states()
        val_acc_metric.reset_states()

        for b in dist_train_ds:
            train_step(b)

        for b in dist_val_ds:
            val_step(b)

        dur = time.time() - t0
        t_loss = float(train_loss_metric.result())
        t_acc = float(train_acc_metric.result()) * 100.0
        v_loss = float(val_loss_metric.result())
        v_acc = float(val_acc_metric.result()) * 100.0

        if v_loss < best_val_loss - 1e-4:
            best_val_loss = v_loss
            patience = 0
            model.save(args.output)
            tag = f"[Saved -> val_loss: {v_loss:.5f}]"
        else:
            patience += 1
            tag = f"[Patience: {patience}/10]"

        print(f"Epoch {epoch:02d}/{args.epochs:02d} ({dur:.2f}s) — loss: {t_loss:.4f} - acc: {t_acc:.2f}% | val_loss: {v_loss:.4f} - val_acc: {v_acc:.2f}% {tag}")

        if patience >= 10:
            print(f"EarlyStopping triggered. Best val_loss: {best_val_loss:.5f}")
            break

    print(f"\nModel saved to: {args.output}")


if __name__ == "__main__":
    main()

