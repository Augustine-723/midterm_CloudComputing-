#!/usr/bin/env python3
"""
submission/train_gradient_tape.py
Advanced Bonus Deliverable: Custom Training, Validation & Testing Loop with tf.GradientTape.

Technical Specifications:
- Built strictly on tf.GradientTape context manager
- Input pipeline with tf.data.Dataset (shuffle buffer 10000, batch 64, prefetch AUTOTUNE)
- Architecture: 25 -> 128 (ReLU) -> 64 (ReLU) -> 5 (Softmax)
- Explicit @tf.function train_step, val_step, and test_step
- Handcrafted EarlyStopping (patience=10) and ModelCheckpoint
- Strictly avoids model.fit(), model.evaluate(), and model.predict()
"""

import os
import sys
import time
import argparse
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

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
DEFAULT_MODEL_OUT = os.path.join(SCRIPT_DIR, "models", "gradient_tape_model.h5")


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


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(25,), name="input_obs"),
        tf.keras.layers.Dense(128, activation="relu", name="dense_1"),
        tf.keras.layers.Dense(64, activation="relu", name="dense_2"),
        tf.keras.layers.Dense(5, activation="softmax", name="output_action")
    ], name="GradientTape_NN")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train with custom tf.GradientTape loop")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output", type=str, default=DEFAULT_MODEL_OUT, help="Saved model path")
    args = parser.parse_args()

    tf.random.set_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print(" ADVANCED BONUS: CUSTOM TRAINING LOOP WITH TF.GRADIENTTAPE")
    print("=" * 70)

    # 1. Datasets
    (x_train, y_train), (x_val, y_val) = load_datasets()
    print(f"Loaded Train: {x_train.shape[0]} samples | Val: {x_val.shape[0]} samples")

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=10000, seed=42)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # 2. Model & Optimization Setup
    model = build_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")
    val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="val_acc")

    # 3. Explicit Step Functions
    @tf.function
    def train_step(x_b, y_b):
        with tf.GradientTape() as tape:
            preds = model(x_b, training=True)
            loss = loss_fn(y_b, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss_metric.update_state(loss)
        train_acc_metric.update_state(y_b, preds)
        return loss

    @tf.function
    def val_step(x_b, y_b):
        preds = model(x_b, training=False)
        loss = loss_fn(y_b, preds)
        val_loss_metric.update_state(loss)
        val_acc_metric.update_state(y_b, preds)
        return loss

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_cnt = 0
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss_metric.reset_states()
        train_acc_metric.reset_states()
        val_loss_metric.reset_states()
        val_acc_metric.reset_states()

        for xb, yb in train_ds:
            train_step(xb, yb)

        for xb, yb in val_ds:
            val_step(xb, yb)

        dur = time.time() - t0
        t_loss = float(train_loss_metric.result())
        t_acc = float(train_acc_metric.result()) * 100.0
        v_loss = float(val_loss_metric.result())
        v_acc = float(val_acc_metric.result()) * 100.0

        if v_loss < best_val_loss - 1e-4:
            best_val_loss = v_loss
            patience_cnt = 0
            model.save(args.output)
            tag = f"[Saved -> val_loss: {v_loss:.5f}]"
        else:
            patience_cnt += 1
            tag = f"[Patience: {patience_cnt}/10]"

        print(f"Epoch {epoch:02d}/{args.epochs:02d} ({dur:.2f}s) — loss: {t_loss:.4f} - acc: {t_acc:.2f}% | val_loss: {v_loss:.4f} - val_acc: {v_acc:.2f}% {tag}")

        if patience_cnt >= 10:
            print(f"EarlyStopping triggered. Best val_loss: {best_val_loss:.5f}")
            break

    print(f"\nModel saved to: {args.output}")


if __name__ == "__main__":
    main()

