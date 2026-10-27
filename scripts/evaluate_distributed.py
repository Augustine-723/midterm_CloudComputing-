#!/usr/bin/env python3
"""
scripts/evaluate_distributed.py
Phase 5B: Evaluation and Three-Model Comparison:
  1. Baseline Standard (model.fit)
  2. Custom GradientTape (Single device)
  3. Distributed GradientTape (2 Logical CPU Replicas)

Strict Constraints:
- Evaluates strictly on validation dataset (5,000 samples)
- Strictly avoids model.fit(), model.evaluate(), and model.predict()
- Direct forward pass tensor evaluation: model(x, training=False)
- Generates 3 comparison figures:
    1. reports/figures/dist_training_loss.png
    2. reports/figures/dist_validation_accuracy.png
    3. reports/figures/dist_three_model_comparison.png
- Compiles complete results into reports/phase5b_results.json
"""

import os
import sys
import json
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = ["0: LANE_LEFT", "1: IDLE", "2: LANE_RIGHT", "3: FASTER", "4: SLOWER"]
SHORT_NAMES = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]

BASELINE_PATH = "/app/models/baseline_standard.h5"
CUSTOM_GT_PATH = "/app/models/gradient_tape_best.h5"
DISTRIBUTED_PATH = "/app/models/distributed_cpu_best.h5"
VAL_DATA_PATH = "/app/dataset/validation.npz"

GT_HIST_PATH = "/app/models/gradient_tape_history.json"
DIST_HIST_PATH = "/app/models/distributed_cpu_history.json"
OUTPUT_JSON_PATH = "/app/reports/phase5b_results.json"
FIGURES_DIR = "/app/reports/figures"


def compute_metrics_direct(model, x_val, y_val):
    """Compute all evaluation metrics using direct tensor forward pass in batches."""
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    batch_size = 128
    probs_list = []

    for i in range(0, len(x_val), batch_size):
        x_b = tf.constant(x_val[i:i+batch_size])
        out_b = model(x_b, training=False)
        probs_list.append(out_b.numpy())

    probs = np.concatenate(probs_list, axis=0)
    preds = np.argmax(probs, axis=1)
    loss = float(loss_fn(y_val, probs).numpy())

    n_classes = 5
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_val, preds):
        cm[t, p] += 1

    per_class = {}
    precisions, recalls, f1s, supports = [], [], [], []

    for c in range(n_classes):
        tp = int(cm[c, c])
        fp = int(np.sum(cm[:, c]) - tp)
        fn = int(np.sum(cm[c, :]) - tp)
        supp = int(np.sum(cm[c, :]))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(supp)

        per_class[CLASS_NAMES[c]] = {
            "name": SHORT_NAMES[c],
            "support": supp,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1)
        }

    accuracy = float(np.sum(np.diag(cm)) / len(y_val))
    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))
    weighted_f1 = float(np.sum(np.array(f1s) * np.array(supports)) / np.sum(supports))
    errors = int(len(y_val) - np.sum(np.diag(cm)))

    return {
        "accuracy": accuracy,
        "loss": loss,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "total_errors": errors,
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }


def main():
    print("=" * 70)
    print(" PHASE 5B: THREE-MODEL EVALUATION & DISTRIBUTED COMPARISON")
    print("=" * 70)

    # 1. Load Validation Data
    val_raw = np.load(VAL_DATA_PATH)
    x_val = val_raw["data"].astype(np.float32).reshape(-1, 25)
    y_val = val_raw["label"].astype(np.int32).flatten()
    print(f"Validation dataset loaded: {x_val.shape[0]} samples.")

    # 2. Load Models
    print(f"Loading Baseline Standard:   {BASELINE_PATH}")
    baseline_model = tf.keras.models.load_model(BASELINE_PATH, compile=False)

    print(f"Loading Custom GradientTape: {CUSTOM_GT_PATH}")
    custom_gt_model = tf.keras.models.load_model(CUSTOM_GT_PATH, compile=False)

    print(f"Loading Distributed Model:   {DISTRIBUTED_PATH}")
    dist_model = tf.keras.models.load_model(DISTRIBUTED_PATH, compile=False)

    # 3. Compute Metrics via Direct Forward Pass
    print("\nComputing metrics via direct forward pass model(x, training=False)...")
    baseline_metrics = compute_metrics_direct(baseline_model, x_val, y_val)
    custom_gt_metrics = compute_metrics_direct(custom_gt_model, x_val, y_val)
    dist_metrics = compute_metrics_direct(dist_model, x_val, y_val)

    # Load history data
    gt_hist = {}
    if os.path.exists(GT_HIST_PATH):
        with open(GT_HIST_PATH, "r") as f:
            gt_hist = json.load(f)

    dist_hist = {}
    if os.path.exists(DIST_HIST_PATH):
        with open(DIST_HIST_PATH, "r") as f:
            dist_hist = json.load(f)

    # 4. Print Comparison Table
    print("\n" + "=" * 80)
    print(f"{'Metric':<25} | {'Baseline (fit)':<14} | {'Custom GT':<14} | {'Distributed (Mirrored)':<14}")
    print("-" * 80)
    print(f"{'Validation Accuracy':<25} | {baseline_metrics['accuracy']*100:<12.2f}% | {custom_gt_metrics['accuracy']*100:<12.2f}% | {dist_metrics['accuracy']*100:<12.2f}%")
    print(f"{'Validation Loss':<25} | {baseline_metrics['loss']:<14.4f} | {custom_gt_metrics['loss']:<14.4f} | {dist_metrics['loss']:<14.4f}")
    print(f"{'Macro Precision':<25} | {baseline_metrics['macro_precision']*100:<12.2f}% | {custom_gt_metrics['macro_precision']*100:<12.2f}% | {dist_metrics['macro_precision']*100:<12.2f}%")
    print(f"{'Macro Recall':<25} | {baseline_metrics['macro_recall']*100:<12.2f}% | {custom_gt_metrics['macro_recall']*100:<12.2f}% | {dist_metrics['macro_recall']*100:<12.2f}%")
    print(f"{'Macro F1-score':<25} | {baseline_metrics['macro_f1']*100:<12.2f}% | {custom_gt_metrics['macro_f1']*100:<12.2f}% | {dist_metrics['macro_f1']*100:<12.2f}%")
    print(f"{'Weighted F1-score':<25} | {baseline_metrics['weighted_f1']*100:<12.2f}% | {custom_gt_metrics['weighted_f1']*100:<12.2f}% | {dist_metrics['weighted_f1']*100:<12.2f}%")
    print(f"{'Total Errors':<25} | {baseline_metrics['total_errors']:<14d} | {custom_gt_metrics['total_errors']:<14d} | {dist_metrics['total_errors']:<14d}")
    print("=" * 80)

    print("\nPer-class Recall Comparison:")
    for c_name in CLASS_NAMES:
        b_rec = baseline_metrics["per_class"][c_name]["recall"] * 100
        gt_rec = custom_gt_metrics["per_class"][c_name]["recall"] * 100
        d_rec = dist_metrics["per_class"][c_name]["recall"] * 100
        print(f"  {c_name:<16}: Baseline = {b_rec:6.2f}% | Custom GT = {gt_rec:6.2f}% | Distributed = {d_rec:6.2f}%")

    # 5. Generate Figures
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Figure 1: Training Loss Comparison Curve
    fig, ax = plt.subplots(figsize=(9, 5))
    if gt_hist.get("train_loss"):
        ax.plot(gt_hist["epochs"], gt_hist["train_loss"], label="Custom GT (Single CPU)", color="#1f77b4", linewidth=2)
    if dist_hist.get("train_loss"):
        ax.plot(dist_hist["epochs"], dist_hist["train_loss"], label="Distributed (2 Logical CPUs)", color="#d62728", linewidth=2, linestyle="--")
    ax.set_title("Training Loss: Single-Device GT vs Distributed MirroredStrategy", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Sparse Categorical Crossentropy", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "dist_training_loss.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"\nSaved: {fig1_path}")

    # Figure 2: Validation Accuracy Comparison Curve
    fig, ax = plt.subplots(figsize=(9, 5))
    if gt_hist.get("val_acc"):
        ax.plot(gt_hist["epochs"], gt_hist["val_acc"], label="Custom GT Val Acc", color="#1f77b4", linewidth=2)
    if dist_hist.get("val_acc"):
        ax.plot(dist_hist["epochs"], dist_hist["val_acc"], label="Distributed Val Acc", color="#2ca02c", linewidth=2, linestyle="--")
    ax.axhline(baseline_metrics["accuracy"] * 100, color="#ff7f0e", linestyle=":", linewidth=2, label=f"Baseline Standard ({baseline_metrics['accuracy']*100:.2f}%)")
    ax.set_title("Validation Accuracy Trajectory Across Epochs", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "dist_validation_accuracy.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig2_path}")

    # Figure 3: Three-Model Bar Comparison (Overall & Per-class Recall)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Overall Metrics
    metrics_names = ["Val Acc", "Macro F1", "Weighted F1"]
    b_vals = [baseline_metrics["accuracy"]*100, baseline_metrics["macro_f1"]*100, baseline_metrics["weighted_f1"]*100]
    gt_vals = [custom_gt_metrics["accuracy"]*100, custom_gt_metrics["macro_f1"]*100, custom_gt_metrics["weighted_f1"]*100]
    d_vals = [dist_metrics["accuracy"]*100, dist_metrics["macro_f1"]*100, dist_metrics["weighted_f1"]*100]

    x = np.arange(len(metrics_names))
    width = 0.25
    ax1.bar(x - width, b_vals, width, label="Baseline (fit)", color="#1f77b4")
    ax1.bar(x, gt_vals, width, label="Custom GT (Single)", color="#ff7f0e")
    ax1.bar(x + width, d_vals, width, label="Distributed (2 CPUs)", color="#2ca02c")
    ax1.set_ylabel("Percentage (%)", fontsize=11)
    ax1.set_title("Overall Model Metrics", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names, fontsize=11)
    ax1.set_ylim(85, 98)
    ax1.grid(True, linestyle="--", alpha=0.6, axis="y")
    ax1.legend(fontsize=10)

    # Subplot 2: Per-Class Recall
    c_labels = [s.replace("LANE_", "L_") for s in SHORT_NAMES]
    b_recs = [baseline_metrics["per_class"][c]["recall"]*100 for c in CLASS_NAMES]
    gt_recs = [custom_gt_metrics["per_class"][c]["recall"]*100 for c in CLASS_NAMES]
    d_recs = [dist_metrics["per_class"][c]["recall"]*100 for c in CLASS_NAMES]

    x2 = np.arange(len(c_labels))
    ax2.bar(x2 - width, b_recs, width, label="Baseline", color="#1f77b4")
    ax2.bar(x2, gt_recs, width, label="Custom GT", color="#ff7f0e")
    ax2.bar(x2 + width, d_recs, width, label="Distributed", color="#2ca02c")
    ax2.set_ylabel("Recall (%)", fontsize=11)
    ax2.set_title("Per-Class Recall Comparison", fontsize=12, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(c_labels, fontsize=10)
    ax2.set_ylim(60, 100)
    ax2.grid(True, linestyle="--", alpha=0.6, axis="y")
    ax2.legend(fontsize=10)

    fig.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "dist_three_model_comparison.png")
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig3_path}")

    # 6. Save JSON Data
    results_payload = {
        "metadata": {
            "title": "Phase 5B Three-Model Evaluation & Comparison",
            "eval_dataset": "validation.npz (5000 samples)",
            "tensorflow_version": tf.__version__,
            "python_version": sys.version.split()[0]
        },
        "baseline_standard": baseline_metrics,
        "custom_gradient_tape": custom_gt_metrics,
        "distributed_gradient_tape": dist_metrics
    }

    # If phase5b_results.json already exists with benchmark info, preserve it
    if os.path.exists(OUTPUT_JSON_PATH):
        try:
            with open(OUTPUT_JSON_PATH, "r") as f:
                existing = json.load(f)
            if "benchmark" in existing:
                results_payload["benchmark"] = existing["benchmark"]
        except Exception:
            pass

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)
    print(f"\nFinal evaluation saved to: {OUTPUT_JSON_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()

