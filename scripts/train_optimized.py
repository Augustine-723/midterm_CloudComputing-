#!/usr/bin/env python3
"""
scripts/train_optimized.py
Train Optimized Neural Network Models for Phase 3B:
- Experiment B: Wider Network (25 -> 256 -> 128 -> 5, lr=0.001) -> models/optimized_wider.h5
- Experiment C: Lower Learning Rate (25 -> 128 -> 64 -> 5, lr=0.0003) -> models/optimized_low_lr.h5
"""

import json
import os
import random
import sys
import time
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_wider_model():
    """Build Wider Network: 25 -> 256 -> 128 -> 5."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(25,), name="input_state"),
        tf.keras.layers.Dense(256, activation="relu", name="dense_wider_1"),
        tf.keras.layers.Dense(128, activation="relu", name="dense_wider_2"),
        tf.keras.layers.Dense(5, activation="softmax", name="output_action"),
    ], name="Roundabout_Wider_NN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def build_low_lr_model():
    """Build Baseline Architecture with Lower Learning Rate (0.0003): 25 -> 128 -> 64 -> 5."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(25,), name="input_state"),
        tf.keras.layers.Dense(128, activation="relu", name="dense_1"),
        tf.keras.layers.Dense(64, activation="relu", name="dense_2"),
        tf.keras.layers.Dense(5, activation="softmax", name="output_action"),
    ], name="Roundabout_LowLR_NN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(root_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 70)
    print(" PHASE 3B — TASK C: HYPERPARAMETER OPTIMIZATION TRAINING")
    print("=" * 70)
    print(f"TensorFlow: {tf.__version__}")
    print(f"Python:     {sys.version.split()[0]}")
    print(f"NumPy:      {np.__version__}")

    # Load Datasets
    train_path = os.path.join(root_dir, "dataset", "train.npz")
    val_path = os.path.join(root_dir, "dataset", "validation.npz")

    train_npz = np.load(train_path)
    val_npz = np.load(val_path)

    x_train = train_npz["data"].astype(np.float32)
    y_train = train_npz["label"].astype(np.int32).flatten()
    x_val = val_npz["data"].astype(np.float32)
    y_val = val_npz["label"].astype(np.int32).flatten()

    print(f"\nDatasets loaded: Train {x_train.shape}, Val {x_val.shape}")

    history_records = {}

    # =========================================================================
    # EXPERIMENT B: Wider Network (25 -> 256 -> 128 -> 5, lr=0.001)
    # =========================================================================
    print("\n" + "=" * 70)
    print(" [EXPERIMENT B] Training Wider Network (25 -> 256 -> 128 -> 5, lr=0.001)")
    print("=" * 70)
    set_seed(42)
    model_wider = build_wider_model()
    model_wider.summary()
    save_path_wider = os.path.join(models_dir, "optimized_wider.h5")

    callbacks_wider = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=save_path_wider, monitor="val_loss", save_best_only=True, verbose=1),
    ]

    start_wider = time.time()
    hist_wider = model_wider.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=64,
        epochs=100,
        callbacks=callbacks_wider,
        verbose=1,
    )
    dur_wider = time.time() - start_wider

    best_epoch_wider = int(np.argmin(hist_wider.history["val_loss"])) + 1
    best_val_loss_wider = float(np.min(hist_wider.history["val_loss"]))
    best_val_acc_wider = float(hist_wider.history["val_accuracy"][best_epoch_wider - 1])

    print(f"\n[Exp B Finished] Duration: {dur_wider:.2f}s, Best Epoch: {best_epoch_wider}")
    print(f"                 Val Loss: {best_val_loss_wider:.4f}, Val Acc: {best_val_acc_wider*100:.2f}%")
    print(f"                 Saved to: {save_path_wider}")

    history_records["exp_b_wider"] = {
        "architecture": "25 -> 256 -> 128 -> 5",
        "learning_rate": 0.001,
        "total_params": model_wider.count_params(),
        "duration": dur_wider,
        "best_epoch": best_epoch_wider,
        "best_val_loss": best_val_loss_wider,
        "best_val_acc": best_val_acc_wider,
        "history": {k: [float(x) for x in v] for k, v in hist_wider.history.items()},
    }

    # =========================================================================
    # EXPERIMENT C: Lower Learning Rate (25 -> 128 -> 64 -> 5, lr=0.0003)
    # =========================================================================
    print("\n" + "=" * 70)
    print(" [EXPERIMENT C] Training Lower Learning Rate (25 -> 128 -> 64 -> 5, lr=0.0003)")
    print("=" * 70)
    set_seed(42)
    model_low_lr = build_low_lr_model()
    model_low_lr.summary()
    save_path_low_lr = os.path.join(models_dir, "optimized_low_lr.h5")

    callbacks_low_lr = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=save_path_low_lr, monitor="val_loss", save_best_only=True, verbose=1),
    ]

    start_low_lr = time.time()
    hist_low_lr = model_low_lr.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=64,
        epochs=100,
        callbacks=callbacks_low_lr,
        verbose=1,
    )
    dur_low_lr = time.time() - start_low_lr

    best_epoch_low_lr = int(np.argmin(hist_low_lr.history["val_loss"])) + 1
    best_val_loss_low_lr = float(np.min(hist_low_lr.history["val_loss"]))
    best_val_acc_low_lr = float(hist_low_lr.history["val_accuracy"][best_epoch_low_lr - 1])

    print(f"\n[Exp C Finished] Duration: {dur_low_lr:.2f}s, Best Epoch: {best_epoch_low_lr}")
    print(f"                 Val Loss: {best_val_loss_low_lr:.4f}, Val Acc: {best_val_acc_low_lr*100:.2f}%")
    print(f"                 Saved to: {save_path_low_lr}")

    history_records["exp_c_low_lr"] = {
        "architecture": "25 -> 128 -> 64 -> 5",
        "learning_rate": 0.0003,
        "total_params": model_low_lr.count_params(),
        "duration": dur_low_lr,
        "best_epoch": best_epoch_low_lr,
        "best_val_loss": best_val_loss_low_lr,
        "best_val_acc": best_val_acc_low_lr,
        "history": {k: [float(x) for x in v] for k, v in hist_low_lr.history.items()},
    }

    # Save histories
    out_hist_path = os.path.join(models_dir, "optimized_training_history.json")
    with open(out_hist_path, "w") as f:
        json.dump(history_records, f, indent=2)
    print(f"\nSaved training history to: {out_hist_path}")
    print("=" * 70)
    print(" OPTIMIZATION TRAINING FINISHED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()

