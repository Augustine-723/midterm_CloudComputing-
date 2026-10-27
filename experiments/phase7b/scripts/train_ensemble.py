#!/usr/bin/env python3
"""
experiments/phase7b/scripts/train_ensemble.py
Phase 7B — Independent Seed Training & Keras Functional Ensemble Model Construction.

Architecture: 25 -> 128 -> 64 -> 5 (Baseline Standard)
Optimizer: Adam(learning_rate=0.001)
Batch Size: 64
EarlyStopping: patience=10, restore_best_weights=True, max_epochs=100
Seeds: 42, 123, 2026

Split:
- 80/20 Stratified Split from dataset/train.npz (35,000 samples):
  - Internal Train: 28,000 samples
  - Internal Dev:    7,000 samples
- dataset/validation.npz (5,000 samples) is strictly reserved for final evaluation.
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


def set_seed(seed: int):
    """Set all random seeds for deterministic behavior."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_stratified_train_dev_split(x, y, dev_ratio=0.20, split_seed=42):
    """
    Perform a deterministic stratified split on (x, y).
    Returns (x_train, y_train), (x_dev, y_dev), (train_idx, dev_idx)
    """
    rng = np.random.RandomState(split_seed)
    n_classes = len(np.unique(y))
    train_indices = []
    dev_indices = []

    for c in range(n_classes):
        cls_idx = np.where(y == c)[0]
        rng.shuffle(cls_idx)
        n_dev = int(round(len(cls_idx) * dev_ratio))
        dev_indices.extend(cls_idx[:n_dev])
        train_indices.extend(cls_idx[n_dev:])

    train_indices = np.array(train_indices, dtype=np.int32)
    dev_indices = np.array(dev_indices, dtype=np.int32)
    rng.shuffle(train_indices)
    rng.shuffle(dev_indices)

    x_train, y_train = x[train_indices], y[train_indices]
    x_dev, y_dev = x[dev_indices], y[dev_indices]
    return (x_train, y_train), (x_dev, y_dev), (train_indices, dev_indices)


def build_baseline_model(name_suffix=""):
    """Build the Baseline feedforward network with unique layer names."""
    suffix = f"_{name_suffix}" if name_suffix else ""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(25,), name=f"input_state{suffix}"),
        tf.keras.layers.Dense(128, activation="relu", name=f"dense_1{suffix}"),
        tf.keras.layers.Dense(64, activation="relu", name=f"dense_2{suffix}"),
        tf.keras.layers.Dense(5, activation="softmax", name=f"output_action{suffix}"),
    ], name=f"Roundabout_Baseline{suffix}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    phase_dir = os.path.join(repo_root, "experiments", "phase7b")
    models_dir = os.path.join(phase_dir, "models")
    reports_dir = os.path.join(phase_dir, "reports")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 75)
    print(" PHASE 7B — INDEPENDENT SEED TRAINING & ENSEMBLE BUILD")
    print("=" * 75)
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Python Version:     {sys.version.split()[0]}")
    print(f"NumPy Version:      {np.__version__}")

    # 1. Load train.npz
    train_path = os.path.join(repo_root, "dataset", "train.npz")
    print(f"\n[1] Loading Training Dataset from: {train_path}")
    raw_data = np.load(train_path)
    x_full = raw_data["data"].astype(np.float32)
    y_full = raw_data["label"].astype(np.int32).flatten()

    assert x_full.shape == (35000, 25), f"Unexpected shape: {x_full.shape}"
    assert y_full.shape == (35000,), f"Unexpected label shape: {y_full.shape}"

    # 2. Stratified Split (80% train / 20% internal dev)
    print("\n[2] Performing 80/20 Stratified Split for Internal Dev Evaluation...")
    (x_tr, y_tr), (x_dev, y_dev), (tr_idx, dev_idx) = get_stratified_train_dev_split(
        x_full, y_full, dev_ratio=0.20, split_seed=42
    )
    print(f"  Internal Train: {x_tr.shape[0]} samples (80.0%)")
    print(f"  Internal Dev:   {x_dev.shape[0]} samples (20.0%)")
    print(f"  Train class counts: {np.bincount(y_tr, minlength=5)}")
    print(f"  Dev class counts:   {np.bincount(y_dev, minlength=5)}")

    # Save split indices for exact reproducibility in evaluation script
    split_info_path = os.path.join(phase_dir, "split_indices.npz")
    np.savez_compressed(
        split_info_path,
        train_indices=tr_idx,
        dev_indices=dev_idx,
    )
    print(f"  Saved split indices to: {split_info_path}")

    # 3. Train 3 Independent Models
    seeds = [42, 123, 2026]
    trained_models = {}
    training_summary = {}

    for seed in seeds:
        print("\n" + "-" * 75)
        print(f" Training Model Seed {seed} (Baseline: 25 -> 128 -> 64 -> 5)")
        print("-" * 75)
        set_seed(seed)

        model = build_baseline_model(name_suffix=f"seed{seed}")
        checkpoint_path = os.path.join(models_dir, f"model_seed{seed}.h5")

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
        ]

        t0 = time.time()
        history = model.fit(
            x_tr, y_tr,
            validation_data=(x_dev, y_dev),
            batch_size=64,
            epochs=100,
            callbacks=callbacks,
            verbose=1,
        )
        duration = time.time() - t0

        # Reload best weights
        model = tf.keras.models.load_model(checkpoint_path)
        trained_models[seed] = model

        # Best metrics on dev set
        dev_loss, dev_acc = model.evaluate(x_dev, y_dev, verbose=0)
        best_epoch = int(np.argmin(history.history["val_loss"])) + 1
        total_epochs = len(history.history["loss"])

        print(f"\n[Seed {seed} Complete] Time: {duration:.2f}s | Stopped Epoch: {total_epochs} | Best Epoch: {best_epoch}")
        print(f"                       Dev Loss: {dev_loss:.4f} | Dev Accuracy: {dev_acc*100:.2f}%")
        print(f"                       Saved Model: {checkpoint_path}")

        training_summary[f"seed_{seed}"] = {
            "seed": seed,
            "best_epoch": best_epoch,
            "total_epochs": total_epochs,
            "duration_seconds": round(duration, 2),
            "dev_loss": float(dev_loss),
            "dev_accuracy": float(dev_acc),
            "model_path": os.path.relpath(checkpoint_path, repo_root),
            "history": {
                "loss": [float(v) for v in history.history["loss"]],
                "val_loss": [float(v) for v in history.history["val_loss"]],
                "accuracy": [float(v) for v in history.history["accuracy"]],
                "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
            }
        }

    # 4. Construct Unified Functional Ensemble Model
    print("\n" + "=" * 75)
    print(" [4] Constructing Single-File Keras Functional Ensemble Model")
    print("=" * 75)

    inp = tf.keras.layers.Input(shape=(25,), name="state_input")
    out42 = trained_models[42](inp)
    out123 = trained_models[123](inp)
    out2026 = trained_models[2026](inp)

    # Built-in Keras Average layer computes elementwise mean: (P1 + P2 + P3) / 3
    avg_out = tf.keras.layers.Average(name="soft_voting_average")([out42, out123, out2026])

    ensemble_model = tf.keras.Model(
        inputs=inp,
        outputs=avg_out,
        name="Roundabout_Seed_Ensemble_Model"
    )
    ensemble_model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    ensemble_path = os.path.join(models_dir, "ensemble_unified.h5")
    ensemble_model.save(ensemble_path)
    print(f"Unified Ensemble Model saved to: {ensemble_path}")
    print(f"Ensemble Total Params: {ensemble_model.count_params():,}")
    ensemble_model.summary()

    # 5. Verify Unified Model Consistency and Standalone Reloading
    print("\n[5] Verifying Unified Model Reloading & Consistency...")
    # Test on random sample
    test_batch = x_dev[:100]
    p42 = trained_models[42].predict(test_batch, verbose=0)
    p123 = trained_models[123].predict(test_batch, verbose=0)
    p2026 = trained_models[2026].predict(test_batch, verbose=0)
    manual_avg = (p42 + p123 + p2026) / 3.0

    unified_pred = ensemble_model.predict(test_batch, verbose=0)
    np.testing.assert_allclose(unified_pred, manual_avg, atol=1e-6)
    print("  [PASS] Unified Model output exactly matches manual soft-voting average (atol=1e-6).")

    # Reload from disk
    reloaded_ens = tf.keras.models.load_model(ensemble_path, compile=False)
    reloaded_pred = reloaded_ens.predict(test_batch, verbose=0)
    np.testing.assert_allclose(reloaded_pred, manual_avg, atol=1e-6)
    print("  [PASS] Reloaded model matches manual soft-voting average without custom objects.")

    # Dev evaluation of ensemble
    ens_dev_loss, ens_dev_acc = ensemble_model.evaluate(x_dev, y_dev, verbose=0)
    print(f"\n[Ensemble Dev Performance]")
    print(f"  Dev Loss:     {ens_dev_loss:.4f}")
    print(f"  Dev Accuracy: {ens_dev_acc*100:.2f}%")

    training_summary["ensemble_unified"] = {
        "model_path": os.path.relpath(ensemble_path, repo_root),
        "total_params": ensemble_model.count_params(),
        "dev_loss": float(ens_dev_loss),
        "dev_accuracy": float(ens_dev_acc),
    }

    # Save training intermediate json
    train_record_path = os.path.join(phase_dir, "training_history.json")
    with open(train_record_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)
    print(f"\nSaved training metadata to: {train_record_path}")
    print("\nPhase 7B Training Step Completed Successfully!")


if __name__ == "__main__":
    main()

