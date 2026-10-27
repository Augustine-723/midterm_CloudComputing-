#!/usr/bin/env python3
"""
submission/NN_train.py
Standard Neural Network Training for Highway-env Roundabout Midterm Project.

Course Requirements Implemented:
- Input: 25 observation features from dataset/train.npz
- Output: 5 discrete action probabilities (Softmax)
- Architecture: 25 -> 128 (ReLU) -> 64 (ReLU) -> 5 (Softmax)
- Optimizer: Adam (lr=0.001)
- Loss: SparseCategoricalCrossentropy
- Batch Size: 64, Epochs: 100 with EarlyStopping (patience=10)
- Validation on dataset/validation.npz
- Model saving: YOURMODEL.h5
- Fully portable with relative path resolution
"""

import os
import sys
import argparse
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# Resolve base directories dynamically
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

def resolve_data_dir():
    """Find dataset directory either locally or in parent folder."""
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
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "YOURMODEL.h5")


def load_datasets():
    """Load training and validation datasets."""
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
    """Construct 25 -> 128 -> 64 -> 5 Sequential model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(25,), name="input_obs"),
        tf.keras.layers.Dense(128, activation="relu", name="dense_1"),
        tf.keras.layers.Dense(64, activation="relu", name="dense_2"),
        tf.keras.layers.Dense(5, activation="softmax", name="output_action")
    ], name="Roundabout_NN")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Roundabout Neural Network")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs (default: 100)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--output", type=str, default=DEFAULT_MODEL_PATH, help="Output .h5 model path")
    args = parser.parse_args()

    tf.random.set_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print(" TRAINING STANDARD NEURAL NETWORK FOR HIGHWAY-ENV ROUNDABOUT")
    print("=" * 70)

    # 1. Load Data
    (x_train, y_train), (x_val, y_val) = load_datasets()
    print(f"Loaded Training Data:   {x_train.shape[0]} samples (features: {x_train.shape[1]})")
    print(f"Loaded Validation Data: {x_val.shape[0]} samples (features: {x_val.shape[1]})")

    # 2. Build Model
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    model.summary()

    # 3. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.output,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    # 4. Train
    print("\nStarting Training...")
    history = model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # 5. Evaluate on Validation Set
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    print("\n" + "=" * 70)
    print(f" Training Completed!")
    print(f" Best Validation Loss:     {val_loss:.4f}")
    print(f" Best Validation Accuracy: {val_acc * 100:.2f}%")
    print(f" Model Saved To:           {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()

