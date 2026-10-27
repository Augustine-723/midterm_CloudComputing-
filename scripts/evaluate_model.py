#!/usr/bin/env python3
"""
scripts/evaluate_model.py
Comprehensive evaluation and visualization for Baseline Neural Networks.
Calculates:
1. Validation Accuracy
2. Validation Loss
3. Macro Precision
4. Macro Recall
5. Macro F1-score
6. Weighted F1-score
7. Per-class Precision (0-4)
8. Per-class Recall (0-4)
9. Per-class F1-score (0-4)
10. Confusion Matrix

Generates figures in reports/figures/:
- loss_curve.png
- accuracy_curve.png
- confusion_matrix_standard.png
- confusion_matrix_weighted.png
- recall_comparison.png
"""

import json
import os
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = [
    "0: LANE_LEFT",
    "1: IDLE",
    "2: LANE_RIGHT",
    "3: FASTER",
    "4: SLOWER"
]
SHORT_NAMES = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]


def compute_metrics(y_true, y_pred, y_prob=None, loss_fn=None):
    """Compute comprehensive classification metrics with pure NumPy."""
    n_classes = len(CLASS_NAMES)
    n_samples = len(y_true)

    # Confusion matrix: rows = true, cols = predicted
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class = {}
    precisions = []
    recalls = []
    f1s = []
    supports = []

    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(int(support))

        per_class[CLASS_NAMES[i]] = {
            "class_id": i,
            "name": SHORT_NAMES[i],
            "support": int(support),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
        }

    accuracy = float(np.trace(cm) / n_samples)
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    weighted_f1 = float(sum(f1 * s for f1, s in zip(f1s, supports)) / n_samples)
    weighted_precision = float(sum(p * s for p, s in zip(precisions, supports)) / n_samples)
    weighted_recall = float(sum(r * s for r, s in zip(recalls, supports)) / n_samples)

    loss_val = None
    if loss_fn is not None and y_prob is not None:
        loss_val = float(loss_fn(y_true, y_prob).numpy())

    return {
        "accuracy": accuracy,
        "loss": loss_val,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion_matrix(cm, title, save_path):
    """Plot and save confusion matrix heatmap using Matplotlib."""
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=SHORT_NAMES,
        yticklabels=SHORT_NAMES,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label"
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Annotate numbers and row percentages
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        row_total = cm[i].sum()
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / row_total * 100) if row_total > 0 else 0
            text = f"{val}\n({pct:.1f}%)"
            color = "white" if val > thresh else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_learning_curves(history_data, figures_dir):
    """Plot loss and accuracy curves comparing Exp A and Exp B."""
    hist_a = history_data["exp_a"]["history"]
    hist_b = history_data["exp_b"]["history"]

    epochs_a = range(1, len(hist_a["loss"]) + 1)
    epochs_b = range(1, len(hist_b["loss"]) + 1)

    # 1. Loss Curve
    plt.figure(figsize=(9, 5))
    plt.plot(epochs_a, hist_a["loss"], "b--", label="Exp A Train Loss")
    plt.plot(epochs_a, hist_a["val_loss"], "b-", linewidth=2, label="Exp A Val Loss")
    plt.plot(epochs_b, hist_b["loss"], "r--", label="Exp B Train Loss (Weighted)")
    plt.plot(epochs_b, hist_b["val_loss"], "r-", linewidth=2, label="Exp B Val Loss (Weighted)")
    plt.title("Training and Validation Loss Curve (Exp A vs Exp B)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Sparse Categorical Crossentropy)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    loss_path = os.path.join(figures_dir, "loss_curve.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"Saved: {loss_path}")

    # 2. Accuracy Curve
    plt.figure(figsize=(9, 5))
    plt.plot(epochs_a, hist_a["accuracy"], "b--", label="Exp A Train Acc")
    plt.plot(epochs_a, hist_a["val_accuracy"], "b-", linewidth=2, label="Exp A Val Acc")
    plt.plot(epochs_b, hist_b["accuracy"], "r--", label="Exp B Train Acc (Weighted)")
    plt.plot(epochs_b, hist_b["val_accuracy"], "r-", linewidth=2, label="Exp B Val Acc (Weighted)")
    plt.title("Training and Validation Accuracy Curve (Exp A vs Exp B)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    acc_path = os.path.join(figures_dir, "accuracy_curve.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print(f"Saved: {acc_path}")


def plot_recall_comparison(metrics_a, metrics_b, save_path):
    """Plot bar chart comparing Per-class Recall between Exp A and Exp B."""
    classes = SHORT_NAMES
    recall_a = [metrics_a["per_class"][c]["recall"] for c in CLASS_NAMES]
    recall_b = [metrics_b["per_class"][c]["recall"] for c in CLASS_NAMES]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width/2, [r * 100 for r in recall_a], width, label="Exp A: Standard", color="#4285F4")
    rects2 = ax.bar(x + width/2, [r * 100 for r in recall_b], width, label="Exp B: Class Weighted", color="#EA4335")

    ax.set_ylabel("Recall (%)")
    ax.set_title("Per-class Recall Comparison: Standard vs Class Weighted")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.7)

    # Attach percentage labels
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(root_dir, "models")
    reports_dir = os.path.join(root_dir, "reports")
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 70)
    print(" PHASE 2 — MODEL EVALUATION AND VISUALIZATION")
    print("=" * 70)

    # Load Validation Set
    val_path = os.path.join(root_dir, "dataset", "validation.npz")
    val_npz = np.load(val_path)
    x_val = val_npz["data"].astype(np.float32)
    y_val = val_npz["label"].astype(np.int32).flatten()
    print(f"Loaded Validation Set: {x_val.shape}, {y_val.shape}")

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Load Model A
    path_a = os.path.join(models_dir, "baseline_standard.h5")
    print(f"\n[1] Loading Model A from: {path_a}")
    model_a = tf.keras.models.load_model(path_a)
    print("    Model A successfully loaded.")

    # Model A Inference Verification (shape=(1, 25) -> shape=(1, 5))
    dummy_input = np.zeros((1, 25), dtype=np.float32)
    dummy_out_a = model_a.predict(dummy_input, verbose=0)
    assert dummy_out_a.shape == (1, 5), f"Unexpected output shape: {dummy_out_a.shape}"
    assert np.isclose(dummy_out_a.sum(), 1.0, atol=1e-4), "Probabilities do not sum to 1.0!"
    print(f"    Inference verification: Input (1, 25) -> Output (1, 5) [PASS]")

    # Evaluate Model A
    probs_a = model_a.predict(x_val, batch_size=64, verbose=0)
    preds_a = np.argmax(probs_a, axis=1)
    metrics_a = compute_metrics(y_val, preds_a, probs_a, loss_fn)

    # Load Model B
    path_b = os.path.join(models_dir, "baseline_weighted.h5")
    print(f"\n[2] Loading Model B from: {path_b}")
    model_b = tf.keras.models.load_model(path_b)
    print("    Model B successfully loaded.")

    dummy_out_b = model_b.predict(dummy_input, verbose=0)
    assert dummy_out_b.shape == (1, 5), f"Unexpected output shape: {dummy_out_b.shape}"
    assert np.isclose(dummy_out_b.sum(), 1.0, atol=1e-4), "Probabilities do not sum to 1.0!"
    print(f"    Inference verification: Input (1, 25) -> Output (1, 5) [PASS]")

    # Evaluate Model B
    probs_b = model_b.predict(x_val, batch_size=64, verbose=0)
    preds_b = np.argmax(probs_b, axis=1)
    metrics_b = compute_metrics(y_val, preds_b, probs_b, loss_fn)

    # Print Comparison Table
    print("\n" + "=" * 70)
    print(f"{'Metric':<25} | {'Exp A: Standard':<20} | {'Exp B: Weighted':<20}")
    print("-" * 70)
    print(f"{'Validation Accuracy':<25} | {metrics_a['accuracy']*100:<19.2f}% | {metrics_b['accuracy']*100:<19.2f}%")
    print(f"{'Validation Loss':<25} | {metrics_a['loss']:<20.4f} | {metrics_b['loss']:<20.4f}")
    print(f"{'Macro Precision':<25} | {metrics_a['macro_precision']*100:<19.2f}% | {metrics_b['macro_precision']*100:<19.2f}%")
    print(f"{'Macro Recall':<25} | {metrics_a['macro_recall']*100:<19.2f}% | {metrics_b['macro_recall']*100:<19.2f}%")
    print(f"{'Macro F1-score':<25} | {metrics_a['macro_f1']*100:<19.2f}% | {metrics_b['macro_f1']*100:<19.2f}%")
    print(f"{'Weighted F1-score':<25} | {metrics_a['weighted_f1']*100:<19.2f}% | {metrics_b['weighted_f1']*100:<19.2f}%")
    print("=" * 70)

    print("\nPer-class Recall Comparison:")
    for name in CLASS_NAMES:
        rec_a = metrics_a["per_class"][name]["recall"] * 100
        rec_b = metrics_b["per_class"][name]["recall"] * 100
        sup = metrics_a["per_class"][name]["support"]
        print(f"  {name:<18} (n={sup:4d}): Exp A = {rec_a:5.1f}% | Exp B = {rec_b:5.1f}%")

    # Visualizations
    print("\n[3] Generating Visualizations...")
    cm_path_a = os.path.join(figures_dir, "confusion_matrix_standard.png")
    plot_confusion_matrix(metrics_a["confusion_matrix"], "Confusion Matrix — Standard Baseline (Exp A)", cm_path_a)

    cm_path_b = os.path.join(figures_dir, "confusion_matrix_weighted.png")
    plot_confusion_matrix(metrics_b["confusion_matrix"], "Confusion Matrix — Class Weighted Baseline (Exp B)", cm_path_b)

    recall_cmp_path = os.path.join(figures_dir, "recall_comparison.png")
    plot_recall_comparison(metrics_a, metrics_b, recall_cmp_path)

    # Plot learning curves if training_history.json exists
    history_json_path = os.path.join(models_dir, "training_history.json")
    if os.path.exists(history_json_path):
        with open(history_json_path, "r") as f:
            hist_data = json.load(f)
        plot_learning_curves(hist_data, figures_dir)
    else:
        print("[Notice] training_history.json not found, skipping curve plotting.")

    # Save complete evaluation results
    results_path = os.path.join(reports_dir, "evaluation_results.json")
    eval_output = {
        "exp_a_standard": metrics_a,
        "exp_b_weighted": metrics_b,
    }
    with open(results_path, "w") as f:
        json.dump(eval_output, f, indent=2)
    print(f"\nComplete evaluation results saved to: {results_path}")
    print("=" * 70)
    print(" EVALUATION SCRIPT FINISHED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()

