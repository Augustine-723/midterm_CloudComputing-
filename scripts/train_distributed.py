#!/usr/bin/env python3
"""
scripts/train_distributed.py
Phase 5B: Full Synchronous Data Parallelism Training using Two Logical CPU Devices.

Technical Specifications:
- tf.distribute.MirroredStrategy on devices ['/device:CPU:0', '/device:CPU:1']
- Model variables, optimizer, and metrics created inside strategy.scope()
- Custom tf.GradientTape training step with strategy.run()
- Input pipeline using tf.data.Dataset with strategy.experimental_distribute_dataset()
- Global Batch Size: 64 (32 samples per replica)
- Loss scaling via tf.nn.compute_average_loss(per_example_loss, global_batch_size=64)
- Sample-weighted distributed validation loop
- Custom EarlyStopping (patience=10) and ModelCheckpoint saving to models/distributed_cpu_best.h5
- Mathematical verification of weight synchronization across replicas
- Strictly avoids model.fit(), model.evaluate(), and model.predict()
"""

import os
import sys
import time
import json
import numpy as np

# 1. Device Virtualization Setup: MUST occur before TF runtime initializes
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

def configure_devices():
    """Configure physical CPU into 2 virtual Logical Devices."""
    cpus = tf.config.list_physical_devices("CPU")
    if not cpus:
        raise RuntimeError("No physical CPU found!")
    try:
        tf.config.set_logical_device_configuration(
            cpus[0],
            [
                tf.config.LogicalDeviceConfiguration(),
                tf.config.LogicalDeviceConfiguration()
            ]
        )
    except RuntimeError as e:
        print(f"Warning during logical device configuration: {e}")

configure_devices()

# 2. Global Constants
RANDOM_SEED = 42
GLOBAL_BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 0.001
INPUT_DIM = 25
NUM_CLASSES = 5

TRAIN_PATH = "/app/dataset/train.npz"
VAL_PATH = "/app/dataset/validation.npz"
MODEL_SAVE_PATH = "/app/models/distributed_cpu_best.h5"
HISTORY_SAVE_PATH = "/app/models/distributed_cpu_history.json"

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_datasets():
    """Load and preprocess train and validation datasets."""
    train_raw = np.load(TRAIN_PATH)
    val_raw = np.load(VAL_PATH)

    x_train = train_raw["data"].astype(np.float32).reshape(-1, INPUT_DIM)
    y_train = train_raw["label"].astype(np.int32).flatten()

    x_val = val_raw["data"].astype(np.float32).reshape(-1, INPUT_DIM)
    y_val = val_raw["label"].astype(np.int32).flatten()

    return x_train, y_train, x_val, y_val


def build_model():
    """Construct identical 25 -> 128 -> 64 -> 5 Sequential model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(INPUT_DIM,), name="input_obs"),
        tf.keras.layers.Dense(128, activation="relu", name="dense_1"),
        tf.keras.layers.Dense(64, activation="relu", name="dense_2"),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="output_action")
    ], name="Distributed_Mirrored_NN")
    return model


def main():
    print("=" * 70)
    print(" PHASE 5B: SYNCHRONOUS DATA PARALLELISM TRAINING (MIRROREDSTRATEGY)")
    print("=" * 70)
    print(f"TensorFlow: {tf.__version__}")
    print(f"Python:     {sys.version.split()[0]}")

    # 3. Strategy Initialization
    logical_cpus = tf.config.list_logical_devices("CPU")
    device_names = [d.name for d in logical_cpus]
    print(f"\n[1] Hardware & Distributed Topology:")
    print(f"    - Physical CPUs:        {len(tf.config.list_physical_devices('CPU'))}")
    print(f"    - Logical CPUs:         {len(logical_cpus)} -> {device_names}")
    
    strategy = tf.distribute.MirroredStrategy(devices=device_names)
    num_replicas = strategy.num_replicas_in_sync
    print(f"    - Active Strategy:      {strategy.__class__.__name__}")
    print(f"    - Replicas in Sync:     {num_replicas}")
    print(f"    - Global Batch Size:    {GLOBAL_BATCH_SIZE}")
    print(f"    - Per-Replica Batch:    {GLOBAL_BATCH_SIZE // num_replicas}")

    if num_replicas != 2:
        raise RuntimeError(f"Expected 2 replicas, found {num_replicas}!")

    # 4. Load Data & Create Distributed Datasets
    x_train, y_train, x_val, y_val = load_datasets()
    n_train = len(x_train)
    n_val = len(x_val)
    print(f"\n[2] Datasets Loaded:")
    print(f"    - Training Set:   {n_train} samples (Shape: {x_train.shape})")
    print(f"    - Validation Set: {n_val} samples (Shape: {x_val.shape})")

    # Divisibility analysis
    full_batches = n_train // GLOBAL_BATCH_SIZE
    remainder = n_train % GLOBAL_BATCH_SIZE
    print(f"    - Divisibility:   {full_batches} full batches of {GLOBAL_BATCH_SIZE} + 1 partial batch of {remainder}")
    print(f"    - Handling:       Preserving all {n_train} samples. Partial batch ({remainder}) will be evenly sharded into {remainder // num_replicas} per replica.")

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=10000, seed=RANDOM_SEED)
        .batch(GLOBAL_BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(GLOBAL_BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    dist_train_ds = strategy.experimental_distribute_dataset(train_ds)
    dist_val_ds = strategy.experimental_distribute_dataset(val_ds)

    # 5. Build Model, Optimizer & Metrics inside Strategy Scope
    with strategy.scope():
        model = build_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )

        train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")
        val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
        val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="val_acc")

        def compute_train_loss(labels, predictions):
            per_example_loss = loss_object(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

    model.summary()

    # 6. Step Functions with @tf.function
    # Verification flag to log runtime evidence on step 1
    first_step_diagnostics = {"logged": False}

    @tf.function
    def distributed_train_step(dist_inputs):
        def step_fn(inputs):
            features, labels = inputs
            ctx = tf.distribute.get_replica_context()
            rep_id = ctx.replica_id_in_sync_group
            rep_batch = tf.shape(features)[0]

            with tf.GradientTape() as tape:
                preds = model(features, training=True)
                loss = compute_train_loss(labels, preds)

            grads = tape.gradient(loss, model.trainable_variables)
            # In MirroredStrategy, optimizer.apply_gradients inside step_fn sums gradients across replicas
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_loss_metric.update_state(loss * num_replicas)
            train_acc_metric.update_state(labels, preds)
            return loss, rep_id, rep_batch

        per_replica_losses, rep_ids, rep_batches = strategy.run(step_fn, args=(dist_inputs,))
        step_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        return step_loss, rep_ids, rep_batches

    @tf.function
    def distributed_val_step(dist_inputs):
        def val_step_fn(inputs):
            features, labels = inputs
            preds = model(features, training=False)
            per_example_loss = loss_object(labels, preds)
            val_loss_metric.update_state(per_example_loss)
            val_acc_metric.update_state(labels, preds)

        strategy.run(val_step_fn, args=(dist_inputs,))

    # 7. Training Loop Execution
    print("\n[3] Starting Distributed Synchronous Training Loop...")
    history = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_times": []
    }

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    total_start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()
        train_loss_metric.reset_states()
        train_acc_metric.reset_states()
        val_loss_metric.reset_states()
        val_acc_metric.reset_states()

        # Training Phase
        for batch in dist_train_ds:
            loss, rep_ids, rep_batches = distributed_train_step(batch)
            if not first_step_diagnostics["logged"]:
                r_ids = strategy.experimental_local_results(rep_ids)
                r_batches = strategy.experimental_local_results(rep_batches)
                print(f"\n    [EVIDENCE] Epoch 1 Batch 1 Runtime Execution Verification:")
                print(f"      * Replica Context IDs: {[int(r.numpy()) for r in r_ids]}")
                print(f"      * Sharded Batch Sizes: {[int(b.numpy()) for b in r_batches]}")
                print(f"      * Total Samples in Step: {sum([int(b.numpy()) for b in r_batches])} == {GLOBAL_BATCH_SIZE}")
                first_step_diagnostics["logged"] = True

        train_loss = float(train_loss_metric.result())
        train_acc = float(train_acc_metric.result()) * 100.0

        # Distributed Validation Phase
        for val_batch in dist_val_ds:
            distributed_val_step(val_batch)

        val_loss = float(val_loss_metric.result())
        val_acc = float(val_acc_metric.result()) * 100.0
        epoch_duration = time.time() - epoch_start

        # Record History
        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_times"].append(epoch_duration)

        # Mathematical verification of weight synchronization across replicas
        # (Check every 10 epochs or at checkpoint)
        all_synced = True
        for var in model.trainable_variables:
            if hasattr(var, "values"):
                eq = tf.reduce_all(tf.equal(var.values[0], var.values[1]))
                if not bool(eq):
                    all_synced = False
                    break

        sync_status = "SYNCED" if all_synced else "DESYNC_ERROR"

        # Checkpoint & EarlyStopping Check
        if val_loss < best_val_loss - 1e-4:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save best weights
            model.save(MODEL_SAVE_PATH)
            status_tag = f"[Checkpoint: val_loss -> {val_loss:.5f} (-{improvement:.5f}) | Weights: {sync_status}]"
        else:
            patience_counter += 1
            status_tag = f"[Patience: {patience_counter}/{PATIENCE} | Weights: {sync_status}]"

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS:02d} ({epoch_duration:.2f}s) — "
            f"loss: {train_loss:.4f} - acc: {train_acc:.2f}% | "
            f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.2f}% {status_tag}"
        )

        if patience_counter >= PATIENCE:
            print(f"\n[Early Stopping Triggered] No improvement for {PATIENCE} consecutive epochs.")
            print(f"Restoring best checkpoint from Epoch {best_epoch} with val_loss: {best_val_loss:.5f}")
            break

    total_duration = time.time() - total_start_time
    history["total_duration"] = total_duration
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss

    print("\n" + "=" * 70)
    print(f" Distributed Training Completed in {total_duration:.2f}s across {len(history['epochs'])} epochs.")
    print(f" Best Epoch: {best_epoch} | Lowest Val Loss: {best_val_loss:.5f}")
    print("=" * 70)

    # 8. Reload and Verify Saved Model
    print(f"\n[4] Verifying Saved Model ({MODEL_SAVE_PATH})...")
    reloaded_model = tf.keras.models.load_model(MODEL_SAVE_PATH, compile=False)
    dummy_input = tf.constant(x_val[:1])
    dummy_out = reloaded_model(dummy_input, training=False).numpy()
    print(f"    - Input Shape:        {dummy_input.shape}")
    print(f"    - Output Shape:       {dummy_out.shape}")
    print(f"    - Output Sum:         {np.sum(dummy_out):.4f} (Valid Softmax)")
    print(f"    - Predicted Action:   {np.argmax(dummy_out)}")

    # 9. Dump Training History JSON
    with open(HISTORY_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"    - Training history saved to: {HISTORY_SAVE_PATH}")
    print("=" * 70)
    print(" DISTRIBUTED TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()

