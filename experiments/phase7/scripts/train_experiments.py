#!/usr/bin/env python3
"""
experiments/phase7/scripts/train_experiments.py
Phase 7A: Train Experiment B (Feature Engineering MLP) and Experiment C (Vehicle-Aware Network).

Technical Details:
- Preserves Baseline Standard (models/baseline_standard.h5) untouched as Control.
- Experiment B: Relative Kinematics Feature Engineering Pipeline (dx, dy, dist, dvx, dvy, rel_speed, masked).
- Experiment C: Vehicle-Aware Network with Shared Dense Encoder, Presence Masking, and Ego Fusion.
- Both models accept raw (None, 25) observations directly, ensuring 100% compatibility with .h5 reload and NN_test.py.
- Training settings: Adam(lr=0.001), SparseCategoricalCrossentropy, Batch Size 64, EarlyStopping(patience=10), Seed 42.
- Outputs saved to experiments/phase7/models/
"""

import os
import sys
import json
import time
import argparse
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODELS_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase7", "models")

RANDOM_SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 0.001


def load_data():
    train_raw = np.load(os.path.join(DATA_DIR, "train.npz"))
    val_raw = np.load(os.path.join(DATA_DIR, "validation.npz"))

    x_train = train_raw["data"].astype(np.float32).reshape(-1, 25)
    y_train = train_raw["label"].astype(np.int32).flatten()

    x_val = val_raw["data"].astype(np.float32).reshape(-1, 25)
    y_val = val_raw["label"].astype(np.int32).flatten()

    return (x_train, y_train), (x_val, y_val)


def build_experiment_b_model():
    """
    Experiment B: Feature Engineering MLP
    Takes raw (None, 25) -> Reshape (5, 5) -> Computes 24 masked relative kinematics features:
    (dx, dy, dist, dvx, dvy, rel_speed) for each of the 4 nearby vehicles -> Concatenates to 49 dims -> Dense(128) -> Dense(64) -> Dense(5).
    """
    inp = tf.keras.layers.Input(shape=(25,), name="raw_obs")
    v = tf.keras.layers.Reshape((5, 5))(inp)

    # Ego vehicle is index 0
    ego_x = v[:, 0, 1:2]
    ego_y = v[:, 0, 2:3]
    ego_vx = v[:, 0, 3:4]
    ego_vy = v[:, 0, 4:5]

    rel_feats = []
    for i in range(1, 5):
        p_i = v[:, i, 0:1]  # Presence mask (0.0 or 1.0)
        dx_i = (v[:, i, 1:2] - ego_x) * p_i
        dy_i = (v[:, i, 2:3] - ego_y) * p_i
        dist_i = tf.sqrt(tf.square(dx_i) + tf.square(dy_i) + 1e-8) * p_i
        dvx_i = (v[:, i, 3:4] - ego_vx) * p_i
        dvy_i = (v[:, i, 4:5] - ego_vy) * p_i
        rel_speed_i = tf.sqrt(tf.square(dvx_i) + tf.square(dvy_i) + 1e-8) * p_i
        rel_feats.extend([dx_i, dy_i, dist_i, dvx_i, dvy_i, rel_speed_i])

    all_rel = tf.keras.layers.Concatenate(name="relative_features")(rel_feats)  # 24 dimensions
    combined = tf.keras.layers.Concatenate(name="combined_features")([inp, all_rel])  # 25 + 24 = 49 dimensions

    x = tf.keras.layers.Dense(128, activation="relu", name="dense_1")(combined)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_2")(x)
    out = tf.keras.layers.Dense(5, activation="softmax", name="action_output")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="Feature_Engineered_MLP")
    return model


def build_experiment_c_model():
    """
    Experiment C: Vehicle-Aware Network
    Takes raw (None, 25) -> Reshape (5, 5) ->
    - Ego Feature Encoder: Dense(32) on Ego vehicle features
    - Shared Vehicle Encoder: TimeDistributed(Dense(32)) across nearby vehicles 1..4
    - Masking: Zeroes out absent vehicle encodings using presence mask
    - Aggregation: Sum pooling and Max pooling across nearby vehicles
    - Feature Fusion: Concatenate [ego_enc, sum_pool, max_pool] (96 dims) -> Dense(64) -> Dense(5).
    """
    inp = tf.keras.layers.Input(shape=(25,), name="raw_obs")
    v = tf.keras.layers.Reshape((5, 5))(inp)

    # Ego vehicle features (index 0)
    ego_raw = tf.keras.layers.Lambda(lambda t: t[:, 0, :], name="slice_ego")(v)
    ego_enc = tf.keras.layers.Dense(32, activation="relu", name="ego_encoder")(ego_raw)

    # Other vehicles (indices 1 to 4)
    other_raw = tf.keras.layers.Lambda(lambda t: t[:, 1:, :], name="slice_other")(v)
    presence_mask = tf.keras.layers.Lambda(lambda t: t[:, :, 0:1], name="slice_presence")(other_raw)

    shared_enc = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(32, activation="relu"), name="shared_vehicle_encoder"
    )(other_raw)

    masked_enc = tf.keras.layers.Multiply(name="masked_vehicle_enc")([shared_enc, presence_mask])

    # Aggregation
    sum_pool = tf.keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name="sum_pooling")(masked_enc)
    max_pool = tf.keras.layers.Lambda(lambda t: tf.reduce_max(t, axis=1), name="max_pooling")(masked_enc)

    # Feature Fusion
    fusion = tf.keras.layers.Concatenate(name="fusion")([ego_enc, sum_pool, max_pool])  # 96 dims

    x = tf.keras.layers.Dense(64, activation="relu", name="decision_dense")(fusion)
    out = tf.keras.layers.Dense(5, activation="softmax", name="action_output")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="Vehicle_Aware_Network")
    return model


def train_single_model(model_fn, model_name, save_filename, x_train, y_train, x_val, y_val):
    print("\n" + "=" * 70)
    print(f" TRAINING: {model_name}")
    print("=" * 70)

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model = model_fn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    model.summary()

    save_path = os.path.join(MODELS_DIR, save_filename)
    os.makedirs(MODELS_DIR, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    t0 = time.time()
    hist = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    train_duration = time.time() - t0

    # Best epoch stats
    val_losses = hist.history["val_loss"]
    best_ep = int(np.argmin(val_losses)) + 1
    best_val_loss = float(val_losses[best_ep - 1])
    best_val_acc = float(hist.history["val_accuracy"][best_ep - 1]) * 100.0

    print(f"\n{model_name} Training Completed in {train_duration:.2f}s across {len(val_losses)} epochs.")
    print(f"Best Epoch: {best_ep} | Val Loss: {best_val_loss:.4f} | Val Accuracy: {best_val_acc:.2f}%")
    print(f"Saved Checkpoint: {save_path}")

    # Re-verify loaded model
    reloaded = tf.keras.models.load_model(save_path, compile=False)
    eval_preds = np.argmax(reloaded(tf.constant(x_val), training=False).numpy(), axis=1)
    reloaded_acc = float(np.mean(eval_preds == y_val) * 100.0)
    print(f"Reload verification accuracy: {reloaded_acc:.2f}%")

    history_payload = {
        "model_name": model_name,
        "param_count": int(model.count_params()),
        "train_duration_sec": round(train_duration, 2),
        "total_epochs": len(val_losses),
        "best_epoch": best_ep,
        "best_val_loss": round(best_val_loss, 5),
        "best_val_accuracy": round(best_val_acc, 2),
        "reloaded_accuracy": round(reloaded_acc, 2),
        "train_loss": [float(x) for x in hist.history["loss"]],
        "train_acc": [float(x) * 100.0 for x in hist.history["accuracy"]],
        "val_loss": [float(x) for x in hist.history["val_loss"]],
        "val_acc": [float(x) * 100.0 for x in hist.history["val_accuracy"]]
    }
    return history_payload


def main():
    print("=" * 70)
    print(" PHASE 7A: MODEL OPTIMIZATION & FEATURE ENGINEERING TRAINING")
    print("=" * 70)

    (x_train, y_train), (x_val, y_val) = load_data()
    print(f"Training Samples:   {x_train.shape[0]}")
    print(f"Validation Samples: {x_val.shape[0]}")

    # Train Experiment B
    hist_b = train_single_model(
        model_fn=build_experiment_b_model,
        model_name="Experiment B: Feature Engineering MLP",
        save_filename="fe_mlp_best.h5",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val
    )

    # Train Experiment C
    hist_c = train_single_model(
        model_fn=build_experiment_c_model,
        model_name="Experiment C: Vehicle-Aware Network",
        save_filename="vehicle_aware_best.h5",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val
    )

    # Save training histories
    history_file = os.path.join(MODELS_DIR, "training_history.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump({"experiment_b": hist_b, "experiment_c": hist_c}, f, indent=2)
    print(f"\nAll training histories saved to: {history_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()

