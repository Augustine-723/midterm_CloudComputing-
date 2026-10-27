#!/usr/bin/env python3
"""
scripts/train_baseline.py
Train Baseline Neural Networks for highway-env roundabout-v0 (Supervised Learning).
- Experiment A: Standard Baseline (No class weights) -> models/baseline_standard.h5
- Experiment B: Class Weighted Baseline -> models/baseline_weighted.h5
"""

import json
import os
import random
import sys
import time
import numpy as np

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_model():
    """Build the baseline feedforward neural network."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(25,), name="input_state"),
        tf.keras.layers.Dense(128, activation="relu", name="dense_1"),
        tf.keras.layers.Dense(64, activation="relu", name="dense_2"),
        tf.keras.layers.Dense(5, activation="softmax", name="output_action"),
    ], name="Roundabout_Baseline_NN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def compute_balanced_class_weights(y, n_classes=5):
    """
    Compute balanced class weights:
    weight_i = N / (n_classes * count_i)
    """
    n_samples = len(y)
    counts = np.bincount(y, minlength=n_classes)
    class_weights = {}
    for i in range(n_classes):
        weight = n_samples / (n_classes * counts[i]) if counts[i] > 0 else 1.0
        class_weights[i] = float(weight)
    return class_weights, counts


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(root_dir, "models")
    reports_dir = os.path.join(root_dir, "reports")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 70)
    print(" PHASE 2 — BASELINE NEURAL NETWORK TRAINING")
    print("=" * 70)
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Python Version:     {sys.version.split()[0]}")
    print(f"NumPy Version:      {np.__version__}")

    # 1. Load Datasets
    train_path = os.path.join(root_dir, "dataset", "train.npz")
    val_path = os.path.join(root_dir, "dataset", "validation.npz")

    print(f"\n[1] Loading Datasets...")
    train_npz = np.load(train_path)
    val_npz = np.load(val_path)

    x_train = train_npz["data"].astype(np.float32)
    y_train = train_npz["label"].astype(np.int32).flatten()

    x_val = val_npz["data"].astype(np.float32)
    y_val = val_npz["label"].astype(np.int32).flatten()

    print(f"  Training features:   {x_train.shape}, dtype={x_train.dtype}, min={x_train.min():.2f}, max={x_train.max():.2f}")
    print(f"  Training labels:     {y_train.shape}, dtype={y_train.dtype}, unique={np.unique(y_train)}")
    print(f"  Validation features: {x_val.shape}, dtype={x_val.dtype}, min={x_val.min():.2f}, max={x_val.max():.2f}")
    print(f"  Validation labels:   {y_val.shape}, dtype={y_val.dtype}, unique={np.unique(y_val)}")

    # Validation of data specifications
    assert x_train.shape == (35000, 25), f"Unexpected x_train shape: {x_train.shape}"
    assert y_train.shape == (35000,), f"Unexpected y_train shape: {y_train.shape}"
    assert x_val.shape == (5000, 25), f"Unexpected x_val shape: {x_val.shape}"
    assert y_val.shape == (5000,), f"Unexpected y_val shape: {y_val.shape}"
    assert set(np.unique(y_train)).issubset({0, 1, 2, 3, 4}), "y_train contains invalid classes!"
    assert set(np.unique(y_val)).issubset({0, 1, 2, 3, 4}), "y_val contains invalid classes!"
    print("  [PASS] Dataset dimensions and label ranges verified.")

    # Model parameters summary
    temp_model = create_model()
    temp_model.summary()
    total_params = temp_model.count_params()

    # Training metadata dictionary
    training_meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tf_version": tf.__version__,
        "python_version": sys.version.split()[0],
        "total_params": total_params,
        "batch_size": 64,
        "max_epochs": 100,
        "seed": 42,
    }

    # =========================================================================
    # EXPERIMENT A: Standard Baseline (No class weights)
    # =========================================================================
    print("\n" + "=" * 70)
    print(" [EXPERIMENT A] Training Standard Baseline NN (Unweighted)")
    print("=" * 70)
    set_seed(42)
    model_a = create_model()
    checkpoint_path_a = os.path.join(models_dir, "baseline_standard.h5")

    callbacks_a = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=checkpoint_path_a, monitor="val_loss", save_best_only=True, verbose=1),
    ]

    start_time_a = time.time()
    history_a = model_a.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=64,
        epochs=100,
        callbacks=callbacks_a,
        verbose=1,
    )
    duration_a = time.time() - start_time_a

    # Determine best epoch for A (min val_loss)
    best_epoch_a = int(np.argmin(history_a.history["val_loss"])) + 1
    best_val_loss_a = float(np.min(history_a.history["val_loss"]))
    best_val_acc_a = float(history_a.history["val_accuracy"][best_epoch_a - 1])

    print(f"\n[Exp A Completed] Duration: {duration_a:.2f}s, Best Epoch: {best_epoch_a}")
    print(f"                  Best Val Loss: {best_val_loss_a:.4f}, Best Val Acc: {best_val_acc_a:.4f}")
    print(f"                  Saved to: {checkpoint_path_a}")

    # =========================================================================
    # EXPERIMENT B: Class Weighted Baseline
    # =========================================================================
    print("\n" + "=" * 70)
    print(" [EXPERIMENT B] Training Class Weighted Baseline NN")
    print("=" * 70)
    set_seed(42)
    model_b = create_model()
    checkpoint_path_b = os.path.join(models_dir, "baseline_weighted.h5")

    # Compute class weights on training set only
    class_weights, train_counts = compute_balanced_class_weights(y_train, n_classes=5)
    print("\nCalculated Class Weights (based solely on Training Set):")
    action_names = ["0: LANE_LEFT", "1: IDLE", "2: LANE_RIGHT", "3: FASTER", "4: SLOWER"]
    for i in range(5):
        print(f"  - Action {action_names[i]}: count = {train_counts[i]:5d}, weight = {class_weights[i]:.4f}")

    callbacks_b = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=checkpoint_path_b, monitor="val_loss", save_best_only=True, verbose=1),
    ]

    start_time_b = time.time()
    history_b = model_b.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=64,
        epochs=100,
        class_weight=class_weights,
        callbacks=callbacks_b,
        verbose=1,
    )
    duration_b = time.time() - start_time_b

    best_epoch_b = int(np.argmin(history_b.history["val_loss"])) + 1
    best_val_loss_b = float(np.min(history_b.history["val_loss"]))
    best_val_acc_b = float(history_b.history["val_accuracy"][best_epoch_b - 1])

    print(f"\n[Exp B Completed] Duration: {duration_b:.2f}s, Best Epoch: {best_epoch_b}")
    print(f"                  Best Val Loss: {best_val_loss_b:.4f}, Best Val Acc: {best_val_acc_b:.4f}")
    print(f"                  Saved to: {checkpoint_path_b}")

    # =========================================================================
    # Save Histories for Evaluation & Plotting
    # =========================================================================
    histories_data = {
        "metadata": training_meta,
        "class_weights": {str(k): v for k, v in class_weights.items()},
        "exp_a": {
            "duration": duration_a,
            "best_epoch": best_epoch_a,
            "best_val_loss": best_val_loss_a,
            "best_val_acc": best_val_acc_a,
            "history": {k: [float(x) for x in v] for k, v in history_a.history.items()},
        },
        "exp_b": {
            "duration": duration_b,
            "best_epoch": best_epoch_b,
            "best_val_loss": best_val_loss_b,
            "best_val_acc": best_val_acc_b,
            "history": {k: [float(x) for x in v] for k, v in history_b.history.items()},
        },
    }

    history_json_path = os.path.join(models_dir, "training_history.json")
    with open(history_json_path, "w") as f:
        json.dump(histories_data, f, indent=2)
    print(f"\nTraining history and metadata saved to: {history_json_path}")
    print("=" * 70)
    print(" TRAINING SCRIPT FINISHED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()

