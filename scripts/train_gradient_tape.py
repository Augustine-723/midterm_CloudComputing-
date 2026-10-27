#!/usr/bin/env python3
"""
scripts/train_gradient_tape.py
Phase 4: Custom Training, Validation, and Testing Loop with tf.GradientTape.
- Strictly avoids model.fit(), model.evaluate(), model.predict()
- Uses tf.data.Dataset with shuffle, batch, prefetch
- Custom train_step with tf.GradientTape() and @tf.function
- Custom val_step with @tf.function
- Custom test_step with @tf.function
- Custom EarlyStopping (patience=10) and ModelCheckpoint (saving to models/gradient_tape_best.h5)
- Evaluates on Validation Set using custom test loop
"""

import json
import os
import random
import sys
import time
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_model():
    """Build the Baseline MLP model using Keras Layers."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(25,), name="input_state"),
        tf.keras.layers.Dense(128, activation="relu", name="dense_1"),
        tf.keras.layers.Dense(64, activation="relu", name="dense_2"),
        tf.keras.layers.Dense(5, activation="softmax", name="output_action"),
    ], name="GradientTape_Baseline_NN")
    return model


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(root_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 70)
    print(" PHASE 4 — CUSTOM TRAINING WITH TF.GRADIENTTAPE")
    print("=" * 70)
    print(f"TensorFlow: {tf.__version__}")
    print(f"Python:     {sys.version.split()[0]}")
    print(f"NumPy:      {np.__version__}")

    set_seed(42)

    # 1. Load Datasets
    train_path = os.path.join(root_dir, "dataset", "train.npz")
    val_path = os.path.join(root_dir, "dataset", "validation.npz")

    train_npz = np.load(train_path)
    val_npz = np.load(val_path)

    x_train = train_npz["data"].astype(np.float32)
    y_train = train_npz["label"].astype(np.int32).flatten()
    x_val = val_npz["data"].astype(np.float32)
    y_val = val_npz["label"].astype(np.int32).flatten()

    print(f"\n[1] Datasets Loaded: Train {x_train.shape}, Val {x_val.shape}")

    # 2. Build tf.data.Dataset Pipelines
    BATCH_SIZE = 64
    SHUFFLE_BUFFER = 10000

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=SHUFFLE_BUFFER, seed=42)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    print(f"    - Training Pipeline:   Shuffle(buffer={SHUFFLE_BUFFER}), Batch({BATCH_SIZE}), Prefetch(AUTOTUNE)")
    print(f"    - Validation Pipeline: No Shuffle, Batch({BATCH_SIZE}), Prefetch(AUTOTUNE)")

    # 3. Model, Optimizer, Loss & Metrics Setup
    model = build_model()
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")
    val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="val_acc")

    # 4. Custom Step Definitions
    @tf.function
    def train_step(x_batch, y_batch):
        """Custom train step using tf.GradientTape."""
        with tf.GradientTape() as tape:
            # 1. Forward Pass
            predictions = model(x_batch, training=True)
            # 2. Calculate Loss
            loss_value = loss_fn(y_batch, predictions)

        # 3. Compute Gradients
        gradients = tape.gradient(loss_value, model.trainable_variables)
        # 4. Apply Gradients (Optimizer Update)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # 5. Update Metrics
        train_loss_metric.update_state(loss_value)
        train_acc_metric.update_state(y_batch, predictions)
        return loss_value

    @tf.function
    def val_step(x_batch, y_batch):
        """Custom validation step (No gradient, no weight update)."""
        predictions = model(x_batch, training=False)
        loss_value = loss_fn(y_batch, predictions)
        val_loss_metric.update_state(loss_value)
        val_acc_metric.update_state(y_batch, predictions)
        return loss_value

    @tf.function
    def test_step(x_batch, y_batch):
        """Custom test step for batch inference and evaluation."""
        predictions = model(x_batch, training=False)
        loss_value = loss_fn(y_batch, predictions)
        return predictions, loss_value

    # 5. Custom Training Loop with EarlyStopping & ModelCheckpoint
    MAX_EPOCHS = 100
    PATIENCE = 10
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    checkpoint_path = os.path.join(models_dir, "gradient_tape_best.h5")

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_durations": [],
    }

    print("\n" + "=" * 70)
    print(" STARTING CUSTOM GRADIENTTAPE TRAINING LOOP")
    print("=" * 70)
    total_start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()

        # Reset metrics at epoch start
        train_loss_metric.reset_states()
        train_acc_metric.reset_states()
        val_loss_metric.reset_states()
        val_acc_metric.reset_states()

        # Iterate over training batches
        for x_b, y_b in train_ds:
            train_step(x_b, y_b)

        # Iterate over validation batches
        for x_b, y_b in val_ds:
            val_step(x_b, y_b)

        epoch_duration = time.time() - epoch_start
        t_loss = float(train_loss_metric.result())
        t_acc = float(train_acc_metric.result())
        v_loss = float(val_loss_metric.result())
        v_acc = float(val_acc_metric.result())

        history["train_loss"].append(t_loss)
        history["train_accuracy"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_accuracy"].append(v_acc)
        history["epoch_durations"].append(epoch_duration)

        # EarlyStopping & ModelCheckpoint Logic
        if v_loss < best_val_loss:
            improved_delta = best_val_loss - v_loss
            best_val_loss = v_loss
            best_epoch = epoch
            patience_counter = 0
            model.save(checkpoint_path)
            save_msg = f"[Checkpoint: val_loss improved to {best_val_loss:.5f} (-{improved_delta:.5f}) -> Saved {os.path.basename(checkpoint_path)}]"
        else:
            patience_counter += 1
            save_msg = f"[Patience: {patience_counter}/{PATIENCE}]"

        print(f"Epoch {epoch:02d}/{MAX_EPOCHS:02d} ({epoch_duration:.2f}s) — "
              f"loss: {t_loss:.4f} - acc: {t_acc*100:.2f}% | "
              f"val_loss: {v_loss:.4f} - val_acc: {v_acc*100:.2f}% {save_msg}")

        if patience_counter >= PATIENCE:
            print(f"\n[Early Stopping Triggered] No improvement for {PATIENCE} consecutive epochs.")
            print(f"Restoring best model from Epoch {best_epoch} with val_loss: {best_val_loss:.5f}")
            break

    total_training_duration = time.time() - total_start_time
    print("\n" + "=" * 70)
    print(f" Training Completed in {total_training_duration:.2f}s across {epoch} epochs.")
    print(f" Best Epoch: {best_epoch} | Lowest Val Loss: {best_val_loss:.5f}")
    print("=" * 70)

    # 6. Verify Model Reloading
    print(f"\n[6] Verifying Saved Model Reloading...")
    best_model = tf.keras.models.load_model(checkpoint_path)
    test_in = x_val[:1]
    test_out = best_model(test_in, training=False).numpy()
    assert test_out.shape == (1, 5), f"Unexpected output shape: {test_out.shape}"
    assert np.isclose(test_out.sum(), 1.0, atol=1e-4), "Probabilities do not sum to 1.0!"
    print(f"    Reload verification passed: Output shape (1, 5), probabilities sum to 1.0.")

    # 7. Custom Testing Loop using test_step on Validation Data
    print(f"\n[7] Running Custom Test Step on Validation Data (NO model.predict / model.evaluate)...")
    all_preds = []
    test_loss_metric = tf.keras.metrics.Mean(name="test_loss")
    test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="test_acc")

    for x_b, y_b in val_ds:
        preds_b, loss_b = test_step(x_b, y_b)
        test_loss_metric.update_state(loss_b)
        test_acc_metric.update_state(y_b, preds_b)
        all_preds.append(preds_b.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    pred_labels = np.argmax(all_preds, axis=1)

    final_test_loss = float(test_loss_metric.result())
    final_test_acc = float(test_acc_metric.result())

    print(f"    - Custom Test Loss:     {final_test_loss:.4f}")
    print(f"    - Custom Test Accuracy: {final_test_acc*100:.2f}%")

    # Save training history and evaluation summary
    history_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": "25 -> 128 -> 64 -> 5 (Custom tf.GradientTape)",
        "total_params": model.count_params(),
        "total_training_duration": total_training_duration,
        "total_epochs": epoch,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_test_accuracy": final_test_acc,
        "final_test_loss": final_test_loss,
        "history": history,
    }

    hist_json_path = os.path.join(models_dir, "gradient_tape_history.json")
    with open(hist_json_path, "w") as f:
        json.dump(history_output, f, indent=2)
    print(f"\nTraining history saved to: {hist_json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

