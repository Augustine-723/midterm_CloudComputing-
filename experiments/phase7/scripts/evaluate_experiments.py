#!/usr/bin/env python3
"""
experiments/phase7/scripts/evaluate_experiments.py
Phase 7A: Comprehensive Evaluation of Experiment A (Control), B (Feature Engineering), and C (Vehicle-Aware).

Calculates:
- Validation Accuracy, Loss, Macro/Weighted F1
- Per-class Recall and Precision
- Longitudinal Error Analysis: FASTER -> SLOWER and SLOWER -> FASTER errors
- Parameter Count & Model Size
- Inference Latency (ms per sample, measured over 5,000 validation samples)
- Generates 3 comparison figures in experiments/phase7/figures/
- Saves complete metrics into experiments/phase7/reports/phase7_results.json
"""

import os
import sys
import json
import time
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = ["0: LANE_LEFT", "1: IDLE", "2: LANE_RIGHT", "3: FASTER", "4: SLOWER"]
SHORT_NAMES = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
VAL_PATH = os.path.join(PROJECT_ROOT, "dataset", "validation.npz")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase7", "figures")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase7", "reports")
HISTORY_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase7", "models", "training_history.json")

MODELS = {
    "control_baseline": {
        "name": "Control: Baseline Standard",
        "path": os.path.join(PROJECT_ROOT, "models", "baseline_standard.h5"),
        "color": "#1f77b4"
    },
    "experiment_b": {
        "name": "Exp B: Feature Engineering MLP",
        "path": os.path.join(PROJECT_ROOT, "experiments", "phase7", "models", "fe_mlp_best.h5"),
        "color": "#ff7f0e"
    },
    "experiment_c": {
        "name": "Exp C: Vehicle-Aware Network",
        "path": os.path.join(PROJECT_ROOT, "experiments", "phase7", "models", "vehicle_aware_best.h5"),
        "color": "#2ca02c"
    }
}


def compute_model_metrics(model, x_val, y_val):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Latency benchmark (single sample inference)
    latencies = []
    # Warm-up
    _ = model(tf.constant(x_val[:10]), training=False)
    
    # Timing over 500 samples for latency precision
    n_latency_samples = 500
    for i in range(n_latency_samples):
        t0 = time.perf_counter()
        _ = model(tf.constant(x_val[i:i+1]), training=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

    avg_latency_ms = float(np.mean(latencies))

    # Full batch inference
    batch_size = 128
    probs_list = []
    for i in range(0, len(x_val), batch_size):
        xb = tf.constant(x_val[i:i+batch_size])
        probs_list.append(model(xb, training=False).numpy())
    probs = np.concatenate(probs_list, axis=0)
    preds = np.argmax(probs, axis=1)
    loss = float(loss_fn(y_val, probs).numpy())

    cm = np.zeros((5, 5), dtype=int)
    for t, p in zip(y_val, preds):
        cm[t, p] += 1

    per_class = {}
    precisions, recalls, f1s, supports = [], [], [], []
    for c in range(5):
        tp = int(cm[c, c])
        fp = int(np.sum(cm[:, c]) - tp)
        fn = int(np.sum(cm[c, :]) - tp)
        supp = int(np.sum(cm[c, :]))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        supports.append(supp)
        per_class[CLASS_NAMES[c]] = {
            "name": SHORT_NAMES[c],
            "support": supp,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": float(p),
            "recall": float(r),
            "f1_score": float(f1)
        }

    accuracy = float(np.sum(np.diag(cm)) / len(y_val))
    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))
    weighted_f1 = float(np.sum(np.array(f1s) * np.array(supports)) / np.sum(supports))
    errors = int(len(y_val) - np.sum(np.diag(cm)))

    # Longitudinal confusion analysis
    # Class 3: FASTER, Class 4: SLOWER
    faster_as_slower = int(cm[3, 4])  # True FASTER predicted as SLOWER
    slower_as_faster = int(cm[4, 3])  # True SLOWER predicted as FASTER
    longitudinal_errors = faster_as_slower + slower_as_faster
    longitudinal_error_rate = float(longitudinal_errors / errors * 100.0) if errors > 0 else 0.0

    return {
        "accuracy": accuracy,
        "loss": loss,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "total_errors": errors,
        "longitudinal_analysis": {
            "faster_predicted_as_slower": faster_as_slower,
            "slower_predicted_as_faster": slower_as_faster,
            "total_longitudinal_errors": longitudinal_errors,
            "percent_of_all_errors": round(longitudinal_error_rate, 2)
        },
        "avg_latency_ms": round(avg_latency_ms, 3),
        "param_count": int(model.count_params()),
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }


def main():
    print("=" * 70)
    print(" PHASE 7A: EVALUATION OF EXPERIMENTS A, B, AND C")
    print("=" * 70)

    val_raw = np.load(VAL_PATH)
    x_val = val_raw["data"].astype(np.float32).reshape(-1, 25)
    y_val = val_raw["label"].astype(np.int32).flatten()
    print(f"Validation Set Loaded: {len(y_val)} samples.")

    results = {}
    for key, info in MODELS.items():
        print(f"\nEvaluating {info['name']} ({info['path']})...")
        model = tf.keras.models.load_model(info["path"], compile=False)
        res = compute_model_metrics(model, x_val, y_val)
        res["display_name"] = info["name"]
        results[key] = res
        print(f"  -> Accuracy:     {res['accuracy']*100:.2f}% (Errors: {res['total_errors']})")
        print(f"  -> Loss:         {res['loss']:.4f}")
        print(f"  -> Macro F1:     {res['macro_f1']*100:.2f}%")
        print(f"  -> Params:       {res['param_count']:,}")
        print(f"  -> Avg Latency:  {res['avg_latency_ms']:.3f} ms / sample")
        print(f"  -> Long. Errors: {res['longitudinal_analysis']['total_longitudinal_errors']} ({res['longitudinal_analysis']['percent_of_all_errors']}%)")

    # Print Summary Table
    print("\n" + "=" * 90)
    print(f"{'Metric':<25} | {'Control (Baseline)':<18} | {'Exp B (FE MLP)':<18} | {'Exp C (Vehicle-Aware)':<18}")
    print("-" * 90)
    print(f"{'Validation Accuracy':<25} | {results['control_baseline']['accuracy']*100:<16.2f}% | {results['experiment_b']['accuracy']*100:<16.2f}% | {results['experiment_c']['accuracy']*100:<16.2f}%")
    print(f"{'Validation Loss':<25} | {results['control_baseline']['loss']:<18.4f} | {results['experiment_b']['loss']:<18.4f} | {results['experiment_c']['loss']:<18.4f}")
    print(f"{'Macro F1-Score':<25} | {results['control_baseline']['macro_f1']*100:<16.2f}% | {results['experiment_b']['macro_f1']*100:<16.2f}% | {results['experiment_c']['macro_f1']*100:<16.2f}%")
    print(f"{'Weighted F1-Score':<25} | {results['control_baseline']['weighted_f1']*100:<16.2f}% | {results['experiment_b']['weighted_f1']*100:<16.2f}% | {results['experiment_c']['weighted_f1']*100:<16.2f}%")
    print(f"{'Total Errors':<25} | {results['control_baseline']['total_errors']:<18d} | {results['experiment_b']['total_errors']:<18d} | {results['experiment_c']['total_errors']:<18d}")
    print(f"{'FASTER <-> SLOWER Errors':<25} | {results['control_baseline']['longitudinal_analysis']['total_longitudinal_errors']:<18d} | {results['experiment_b']['longitudinal_analysis']['total_longitudinal_errors']:<18d} | {results['experiment_c']['longitudinal_analysis']['total_longitudinal_errors']:<18d}")
    print(f"{'Model Parameters':<25} | {results['control_baseline']['param_count']:<18,d} | {results['experiment_b']['param_count']:<18,d} | {results['experiment_c']['param_count']:<18,d}")
    print(f"{'Latency per Sample':<25} | {results['control_baseline']['avg_latency_ms']:<15.3f} ms | {results['experiment_b']['avg_latency_ms']:<15.3f} ms | {results['experiment_c']['avg_latency_ms']:<15.3f} ms")
    print("=" * 90)

    # 4. Generate Figures
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Figure 1: Training Curves (if history exists)
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            h = json.load(f)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        
        # Loss
        ax1.plot(h["experiment_b"]["val_loss"], label="Exp B Val Loss", color="#ff7f0e", linewidth=2)
        ax1.plot(h["experiment_c"]["val_loss"], label="Exp C Val Loss", color="#2ca02c", linewidth=2)
        ax1.axhline(results["control_baseline"]["loss"], color="#1f77b4", linestyle=":", label=f"Control ({results['control_baseline']['loss']:.4f})")
        ax1.set_title("Validation Loss Progression", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        # Accuracy
        ax1_r = ax2
        ax1_r.plot(h["experiment_b"]["val_acc"], label="Exp B Val Acc", color="#ff7f0e", linewidth=2)
        ax1_r.plot(h["experiment_c"]["val_acc"], label="Exp C Val Acc", color="#2ca02c", linewidth=2)
        ax1_r.axhline(results["control_baseline"]["accuracy"] * 100, color="#1f77b4", linestyle=":", label=f"Control ({results['control_baseline']['accuracy']*100:.2f}%)")
        ax1_r.set_title("Validation Accuracy Progression", fontsize=12, fontweight="bold")
        ax1_r.set_xlabel("Epoch")
        ax1_r.set_ylabel("Accuracy (%)")
        ax1_r.grid(True, linestyle="--", alpha=0.6)
        ax1_r.legend()

        fig.tight_layout()
        fig1_path = os.path.join(FIGURES_DIR, "phase7_training_curves.png")
        fig.savefig(fig1_path, dpi=300)
        plt.close(fig)
        print(f"\nSaved: {fig1_path}")

    # Figure 2: Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cm_keys = ["control_baseline", "experiment_b", "experiment_c"]
    titles = ["Control: Baseline", "Exp B: Feature Eng MLP", "Exp C: Vehicle-Aware"]
    
    for ax, k, title in zip(axes, cm_keys, titles):
        cm = np.array(results[k]["confusion_matrix"])
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(title, fontsize=12, fontweight="bold")
        tick_marks = np.arange(5)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels([s.replace("LANE_", "L_") for s in SHORT_NAMES], fontsize=9)
        ax.set_yticklabels([s.replace("LANE_", "L_") for s in SHORT_NAMES], fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        
        # Numbers inside cells
        thresh = cm.max() / 2.0
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=8)

    fig.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "phase7_confusion_comparison.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig2_path}")

    # Figure 3: Bar Comparison (Accuracy & Longitudinal Errors)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    m_names = ["Control", "Exp B (FE)", "Exp C (VA)"]
    accs = [results[k]["accuracy"] * 100 for k in cm_keys]
    long_errs = [results[k]["longitudinal_analysis"]["total_longitudinal_errors"] for k in cm_keys]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    bars1 = ax1.bar(m_names, accs, color=colors, width=0.45)
    ax1.set_ylabel("Validation Accuracy (%)")
    ax1.set_title("Validation Accuracy Comparison", fontweight="bold")
    ax1.set_ylim(85, 98)
    ax1.grid(True, linestyle="--", alpha=0.6, axis="y")
    for bar in bars1:
        y = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, y + 0.2, f"{y:.2f}%", ha="center", va="bottom", fontweight="bold")

    bars2 = ax2.bar(m_names, long_errs, color=colors, width=0.45)
    ax2.set_ylabel("Error Count")
    ax2.set_title("FASTER <-> SLOWER Mutual Errors", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6, axis="y")
    for bar in bars2:
        y = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, y + 2, f"{int(y)}", ha="center", va="bottom", fontweight="bold")

    fig.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "phase7_metric_bars.png")
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig3_path}")

    # Save Results JSON
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_json = os.path.join(REPORTS_DIR, "phase7_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 7 evaluation metrics saved to: {out_json}")


if __name__ == "__main__":
    main()

