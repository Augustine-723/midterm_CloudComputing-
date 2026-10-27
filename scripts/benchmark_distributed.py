#!/usr/bin/env python3
"""
scripts/benchmark_distributed.py
Phase 5B: Controlled Performance Benchmark between Single-Device and Distributed Training.

Technical Constraints:
- Equal hardware: Host Intel i5-9400 CPU
- Equal dataset: train.npz (35,000 samples)
- Equal global batch size: 64
- Equal fixed epochs: 5 benchmark epochs
- 1 warm-up epoch excluded from timing to eliminate graph compilation (@tf.function) overhead
- Peak memory measured via resource.getrusage()
- Clean subprocess isolation for device runtime configurations
- Generates reports/figures/dist_throughput_comparison.png
- Appends results into reports/phase5b_results.json
"""

import os
import sys
import time
import json
import argparse
import resource
import subprocess
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN_PATH = "/app/dataset/train.npz"
RESULTS_JSON_PATH = "/app/reports/phase5b_results.json"
FIGURE_PATH = "/app/reports/figures/dist_throughput_comparison.png"

BENCHMARK_EPOCHS = 5
GLOBAL_BATCH_SIZE = 64
INPUT_DIM = 25
NUM_CLASSES = 5


def run_single_worker():
    """Worker process for single-device training benchmark."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf

    tf.random.set_seed(42)
    np.random.seed(42)

    data = np.load(TRAIN_PATH)
    x = data["data"].astype(np.float32).reshape(-1, INPUT_DIM)
    y = data["label"].astype(np.int32).flatten()
    n_samples = len(x)

    ds = (
        tf.data.Dataset.from_tensor_slices((x, y))
        .shuffle(10000, seed=42)
        .batch(GLOBAL_BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(INPUT_DIM,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    @tf.function
    def train_step(x_b, y_b):
        with tf.GradientTape() as tape:
            preds = model(x_b, training=True)
            loss = loss_fn(y_b, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # Warm-up Epoch (not timed)
    for xb, yb in ds:
        _ = train_step(xb, yb)

    epoch_times = []
    start_total = time.time()
    for ep in range(BENCHMARK_EPOCHS):
        ep_start = time.time()
        for xb, yb in ds:
            _ = train_step(xb, yb)
        ep_dur = time.time() - ep_start
        epoch_times.append(ep_dur)

    total_time = time.time() - start_total
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux KB to MB

    total_samples = n_samples * BENCHMARK_EPOCHS
    throughput = total_samples / total_time
    avg_epoch_time = float(np.mean(epoch_times))

    out = {
        "mode": "Single-device (1 CPU)",
        "num_replicas": 1,
        "epochs": BENCHMARK_EPOCHS,
        "epoch_times": epoch_times,
        "avg_epoch_time_sec": round(avg_epoch_time, 4),
        "total_time_sec": round(total_time, 4),
        "throughput_samples_per_sec": round(throughput, 2),
        "peak_rss_mb": round(peak_rss_mb, 2)
    }
    print(json.dumps(out))


def run_distributed_worker():
    """Worker process for 2-Logical-CPU MirroredStrategy benchmark."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf

    cpus = tf.config.list_physical_devices("CPU")
    tf.config.set_logical_device_configuration(
        cpus[0],
        [tf.config.LogicalDeviceConfiguration(), tf.config.LogicalDeviceConfiguration()]
    )

    strategy = tf.distribute.MirroredStrategy(devices=["/device:CPU:0", "/device:CPU:1"])

    tf.random.set_seed(42)
    np.random.seed(42)

    data = np.load(TRAIN_PATH)
    x = data["data"].astype(np.float32).reshape(-1, INPUT_DIM)
    y = data["label"].astype(np.int32).flatten()
    n_samples = len(x)

    ds = (
        tf.data.Dataset.from_tensor_slices((x, y))
        .shuffle(10000, seed=42)
        .batch(GLOBAL_BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    dist_ds = strategy.experimental_distribute_dataset(ds)

    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(INPUT_DIM,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )

        def compute_loss(labels, preds):
            per_example = loss_object(labels, preds)
            return tf.nn.compute_average_loss(per_example, global_batch_size=GLOBAL_BATCH_SIZE)

    @tf.function
    def dist_train_step(dist_inputs):
        def step_fn(inputs):
            features, labels = inputs
            with tf.GradientTape() as tape:
                preds = model(features, training=True)
                loss = compute_loss(labels, preds)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        per_rep_losses = strategy.run(step_fn, args=(dist_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_rep_losses, axis=None)

    # Warm-up Epoch (not timed)
    for b in dist_ds:
        _ = dist_train_step(b)

    epoch_times = []
    start_total = time.time()
    for ep in range(BENCHMARK_EPOCHS):
        ep_start = time.time()
        for b in dist_ds:
            _ = dist_train_step(b)
        ep_dur = time.time() - ep_start
        epoch_times.append(ep_dur)

    total_time = time.time() - start_total
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    total_samples = n_samples * BENCHMARK_EPOCHS
    throughput = total_samples / total_time
    avg_epoch_time = float(np.mean(epoch_times))

    out = {
        "mode": "Distributed MirroredStrategy (2 Logical CPUs)",
        "num_replicas": 2,
        "epochs": BENCHMARK_EPOCHS,
        "epoch_times": epoch_times,
        "avg_epoch_time_sec": round(avg_epoch_time, 4),
        "total_time_sec": round(total_time, 4),
        "throughput_samples_per_sec": round(throughput, 2),
        "peak_rss_mb": round(peak_rss_mb, 2)
    }
    print(json.dumps(out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "distributed", "orchestrate"], default="orchestrate")
    args = parser.parse_args()

    if args.mode == "single":
        run_single_worker()
        return
    elif args.mode == "distributed":
        run_distributed_worker()
        return

    # Master Orchestrator Mode
    print("=" * 70)
    print(" PHASE 5B: CONTROLLED PERFORMANCE BENCHMARK")
    print(" Single-Device vs Two Logical CPU MirroredStrategy")
    print("=" * 70)
    print(f"Dataset: train.npz (35,000 samples)")
    print(f"Global Batch Size: {GLOBAL_BATCH_SIZE}")
    print(f"Benchmark Epochs: {BENCHMARK_EPOCHS} (Warm-up epoch excluded)")

    # 1. Run Single-Device Worker Subprocess
    print("\n[1] Running Single-Device Benchmark Worker...")
    res_single = subprocess.check_output(
        [sys.executable, __file__, "--mode", "single"],
        text=True
    ).strip().splitlines()[-1]
    data_single = json.loads(res_single)
    print(f"    -> Avg Epoch Time: {data_single['avg_epoch_time_sec']:.2f} s")
    print(f"    -> Throughput:     {data_single['throughput_samples_per_sec']:.1f} samples/sec")

    # 2. Run Distributed Worker Subprocess
    print("\n[2] Running Distributed MirroredStrategy Benchmark Worker...")
    res_dist = subprocess.check_output(
        [sys.executable, __file__, "--mode", "distributed"],
        text=True
    ).strip().splitlines()[-1]
    data_dist = json.loads(res_dist)
    print(f"    -> Avg Epoch Time: {data_dist['avg_epoch_time_sec']:.2f} s")
    print(f"    -> Throughput:     {data_dist['throughput_samples_per_sec']:.1f} samples/sec")

    # 3. Print Comparison Table
    print("\n" + "=" * 70)
    print(f"{'Performance Metric':<28} | {'Single-Device':<18} | {'Distributed (2 CPUs)':<18}")
    print("-" * 70)
    print(f"{'Replicas in Sync':<28} | {data_single['num_replicas']:<18d} | {data_dist['num_replicas']:<18d}")
    print(f"{'Avg Epoch Time':<28} | {data_single['avg_epoch_time_sec']:<14.2f} s | {data_dist['avg_epoch_time_sec']:<14.2f} s")
    print(f"{'Total 5-Epoch Time':<28} | {data_single['total_time_sec']:<14.2f} s | {data_dist['total_time_sec']:<14.2f} s")
    print(f"{'Training Throughput':<28} | {data_single['throughput_samples_per_sec']:<12.1f} s/s | {data_dist['throughput_samples_per_sec']:<12.1f} s/s")
    print(f"{'Peak RSS Memory':<28} | {data_single['peak_rss_mb']:<13.1f} MB | {data_dist['peak_rss_mb']:<13.1f} MB")
    
    speedup = data_single['total_time_sec'] / data_dist['total_time_sec']
    print(f"{'Relative Speedup Factor':<28} | {'1.00x (Baseline)':<18} | {speedup:<14.2f}x")
    print("=" * 70)

    # 4. Generate Figure: Throughput & Epoch Time Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    modes = ["Single-Device (1 CPU)", "Distributed (2 CPUs)"]
    throughputs = [data_single["throughput_samples_per_sec"], data_dist["throughput_samples_per_sec"]]
    epoch_times = [data_single["avg_epoch_time_sec"], data_dist["avg_epoch_time_sec"]]
    colors = ["#1f77b4", "#2ca02c"]

    # Throughput
    bars1 = ax1.bar(modes, throughputs, color=colors, width=0.45)
    ax1.set_ylabel("Throughput (samples / sec)", fontsize=11)
    ax1.set_title("Training Throughput Comparison", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6, axis="y")
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 500, f"{yval:.1f} s/s", ha="center", va="bottom", fontweight="bold")

    # Avg Epoch Time
    bars2 = ax2.bar(modes, epoch_times, color=colors, width=0.45)
    ax2.set_ylabel("Avg Time per Epoch (seconds)", fontsize=11)
    ax2.set_title("Epoch Duration Comparison", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6, axis="y")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f} s", ha="center", va="bottom", fontweight="bold")

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIGURE_PATH), exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=300)
    plt.close(fig)
    print(f"\nSaved benchmark figure: {FIGURE_PATH}")

    # 5. Append to reports/phase5b_results.json
    benchmark_payload = {
        "hardware": "Intel Core i5-9400 CPU @ 2.90GHz (6 Cores)",
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "benchmark_epochs": BENCHMARK_EPOCHS,
        "single_device": data_single,
        "distributed_mirrored": data_dist,
        "speedup_ratio": round(speedup, 3),
        "notes": "On CPU logical devices sharing the same physical socket, inter-thread synchronization and All-Reduce communication introduce overhead compared to single-device vectorized C++ Eigen threads. Both configurations execute identical mathematical updates."
    }

    results = {}
    if os.path.exists(RESULTS_JSON_PATH):
        try:
            with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            pass

    results["benchmark"] = benchmark_payload
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Benchmark results appended to: {RESULTS_JSON_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()

