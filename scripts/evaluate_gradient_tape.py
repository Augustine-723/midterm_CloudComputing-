#!/usr/bin/env python3
"""
scripts/evaluate_gradient_tape.py
Phase 4: Evaluation and Comparison between Baseline Standard (model.fit) and Custom GradientTape.
- Strictly avoids model.fit(), model.evaluate(), model.predict()
- Evaluates both models using direct forward pass model(x, training=False)
- Generates 5 comparison figures in reports/figures/:
    1. gt_training_loss_curve.png
    2. gt_validation_loss_curve.png
    3. gt_accuracy_curve.png
    4. gt_confusion_matrix.png
    5. gt_baseline_vs_custom_comparison.png
- Compiles complete comparison report into reports/phase4_results.json
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

CLASS_NAMES = ["0: LANE_LEFT", "1: IDLE", "2: LANE_RIGHT", "3: FASTER", "4: SLOWER"]
SHORT_NAMES = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]


def compute_metrics_direct(model, x_val, y_val):
    """
    Compute metrics by directly calling model(x, training=False).
    STRICTLY avoids model.predict() and model.evaluate().
    """
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Direct tensor forward call in batches to conserve memory
    batch_size = 128
    probs_list = []
    for i in range(0, len(x_val), batch_size):
        x_b = tf.constant(x_val[i:i+batch_size])
        out_b = model(x_b, training=False)
        probs_list.append(out_b.numpy())

    probs = np.concatenate(probs_list, axis=0)
    preds = np.argmax(probs, axis=1)
    loss = float(loss_fn(y_val, probs).numpy())

    n_samples = len(y_val)
    n_classes = 5

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_val, preds):
        cm[t, p] += 1

    per_class = {}
    precisions, recalls, f1s, supports = [], [], [], []

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
    weighted_f1 = float(sum(f * s for f, s in zip(f1s, supports)) / n_samples)
    total_errors = int((preds != y_val).sum())

    return {
        "accuracy": accuracy,
        "loss": loss,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "total_errors": total_errors,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(root_dir, "models")
    reports_dir = os.path.join(root_dir, "reports")
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 70)
    print(" PHASE 4 — EVALUATION & COMPARISON: BASELINE vs GRADIENTTAPE")
    print("=" * 70)

    val_path = os.path.join(root_dir, "dataset", "validation.npz")
    val_data = np.load(val_path)
    x_val = val_data["data"].astype(np.float32)
    y_val = val_data["label"].astype(np.int32).flatten()

    path_baseline = os.path.join(models_dir, "baseline_standard.h5")
    path_custom = os.path.join(models_dir, "gradient_tape_best.h5")

    assert os.path.exists(path_baseline), f"Baseline model not found: {path_baseline}"
    assert os.path.exists(path_custom), f"Custom model not found: {path_custom}"

    print(f"Loading Baseline Standard (model.fit) from: {path_baseline}")
    model_baseline = tf.keras.models.load_model(path_baseline)
    print(f"Loading Custom GradientTape from:          {path_custom}")
    model_custom = tf.keras.models.load_model(path_custom)

    # Compute metrics strictly without model.predict or model.evaluate
    print("\nComputing metrics via direct forward pass...")
    res_baseline = compute_metrics_direct(model_baseline, x_val, y_val)
    res_custom = compute_metrics_direct(model_custom, x_val, y_val)

    # Load histories for duration and curves
    with open(os.path.join(models_dir, "training_history.json")) as f1:
        hist_baseline = json.load(f1)["exp_a"]
    with open(os.path.join(models_dir, "gradient_tape_history.json")) as f2:
        hist_custom = json.load(f2)

    dur_baseline = hist_baseline["duration"]
    dur_custom = hist_custom["total_training_duration"]

    print("\n" + "=" * 75)
    print(f"{'Metric':<30} | {'Baseline (model.fit)':<20} | {'Custom (GradientTape)':<20}")
    print("-" * 75)
    print(f"{'Validation Accuracy':<30} | {res_baseline['accuracy']*100:<19.2f}% | {res_custom['accuracy']*100:<19.2f}%")
    print(f"{'Validation Loss':<30} | {res_baseline['loss']:<20.4f} | {res_custom['loss']:<20.4f}")
    print(f"{'Macro Precision':<30} | {res_baseline['macro_precision']*100:<19.2f}% | {res_custom['macro_precision']*100:<19.2f}%")
    print(f"{'Macro Recall':<30} | {res_baseline['macro_recall']*100:<19.2f}% | {res_custom['macro_recall']*100:<19.2f}%")
    print(f"{'Macro F1-score':<30} | {res_baseline['macro_f1']*100:<19.2f}% | {res_custom['macro_f1']*100:<19.2f}%")
    print(f"{'Weighted F1-score':<30} | {res_baseline['weighted_f1']*100:<19.2f}% | {res_custom['weighted_f1']*100:<19.2f}%")
    print(f"{'Total Errors':<30} | {res_baseline['total_errors']:<20d} | {res_custom['total_errors']:<20d}")
    print(f"{'Training Duration':<30} | {dur_baseline:<19.2f}s | {dur_custom:<19.2f}s")
    print(f"{'Best Epoch':<30} | {hist_baseline['best_epoch']:<20d} | {hist_custom['best_epoch']:<20d}")
    print("=" * 75)

    print("\nPer-class Recall Comparison:")
    for c in CLASS_NAMES:
        rec_b = res_baseline["per_class"][c]["recall"] * 100
        rec_c = res_custom["per_class"][c]["recall"] * 100
        print(f"  {c:<18}: Baseline = {rec_b:5.2f}% | Custom GradientTape = {rec_c:5.2f}%")

    # =========================================================================
    # GENERATE 5 FIGURES
    # =========================================================================
    print("\nGenerating 5 Figures for Phase 4...")

    t_loss_base = hist_baseline["history"]["loss"]
    v_loss_base = hist_baseline["history"]["val_loss"]
    t_acc_base = [x * 100 for x in hist_baseline["history"]["accuracy"]]
    v_acc_base = [x * 100 for x in hist_baseline["history"]["val_accuracy"]]

    t_loss_custom = hist_custom["history"]["train_loss"]
    v_loss_custom = hist_custom["history"]["val_loss"]
    t_acc_custom = [x * 100 for x in hist_custom["history"]["train_accuracy"]]
    v_acc_custom = [x * 100 for x in hist_custom["history"]["val_accuracy"]]

    # Figure 1: Training Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(t_loss_base, label="Baseline (model.fit)", color="#4285F4", linewidth=2)
    plt.plot(t_loss_custom, label="Custom (tf.GradientTape)", color="#EA4335", linestyle="--", linewidth=2)
    plt.title("Training Loss Convergence Curve", fontsize=12)
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "gt_training_loss_curve.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")

    # Figure 2: Validation Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(v_loss_base, label=f"Baseline Best Val Loss: {min(v_loss_base):.4f}", color="#4285F4", linewidth=2)
    plt.plot(v_loss_custom, label=f"Custom Best Val Loss: {min(v_loss_custom):.4f}", color="#EA4335", linestyle="--", linewidth=2)
    plt.title("Validation Loss Generalization Curve", fontsize=12)
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "gt_validation_loss_curve.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")

    # Figure 3: Training / Validation Accuracy Curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(t_acc_base, label="Baseline Train Acc", color="#4285F4")
    ax1.plot(t_acc_custom, label="Custom Train Acc", color="#EA4335", linestyle="--")
    ax1.set_title("Training Accuracy Curve")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(v_acc_base, label=f"Baseline Max Val Acc: {max(v_acc_base):.2f}%", color="#4285F4")
    ax2.plot(v_acc_custom, label=f"Custom Max Val Acc: {max(v_acc_custom):.2f}%", color="#EA4335", linestyle="--")
    ax2.set_title("Validation Accuracy Curve")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("Accuracy Progression (Baseline vs Custom GradientTape)", fontsize=13)
    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "gt_accuracy_curve.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")

    # Figure 4: Confusion Matrix (Custom GradientTape)
    cm_c = np.array(res_custom["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_c, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(5), yticks=range(5),
        xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
        title=f"Confusion Matrix — Custom GradientTape (Acc: {res_custom['accuracy']*100:.2f}%)",
        ylabel="True Label", xlabel="Predicted Label"
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    thresh = cm_c.max() / 2.0
    for i in range(5):
        for j in range(5):
            val = cm_c[i, j]
            pct = val / cm_c[i].sum() * 100 if cm_c[i].sum() > 0 else 0
            ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=9)
    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, "gt_confusion_matrix.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved: {fig4_path}")

    # Figure 5: Baseline vs Custom GradientTape Comparison
    categories = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]
    val_b = [res_baseline["accuracy"]*100, res_baseline["macro_precision"]*100, res_baseline["macro_recall"]*100, res_baseline["macro_f1"]*100, res_baseline["weighted_f1"]*100]
    val_c = [res_custom["accuracy"]*100, res_custom["macro_precision"]*100, res_custom["macro_recall"]*100, res_custom["macro_f1"]*100, res_custom["weighted_f1"]*100]

    x = np.arange(len(categories))
    w = 0.35
    plt.figure(figsize=(10, 5))
    r1 = plt.bar(x - w/2, val_b, w, label="Baseline (model.fit)", color="#4285F4")
    r2 = plt.bar(x + w/2, val_c, w, label="Custom (tf.GradientTape)", color="#EA4335")
    plt.ylabel("Percentage (%)")
    plt.title("Performance Comparison: model.fit() vs tf.GradientTape()")
    plt.xticks(x, categories, rotation=10)
    plt.ylim(85, 98)
    plt.legend(loc="upper left")
    plt.grid(axis="y", linestyle=":", alpha=0.6)

    for r in r1:
        h = r.get_height()
        plt.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for r in r2:
        h = r.get_height()
        plt.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig5_path = os.path.join(figures_dir, "gt_baseline_vs_custom_comparison.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"Saved: {fig5_path}")

    # Save complete results JSON
    phase4_results = {
        "metadata": {
            "title": "Phase 4 Custom tf.GradientTape Evaluation",
            "framework": "TensorFlow 2.13.0",
            "python_version": sys.version.split()[0],
            "dataset": "validation.npz (Used for testing custom test loop)",
        },
        "baseline_standard": {
            "training_method": "tf.keras.Model.fit()",
            "duration": dur_baseline,
            "best_epoch": hist_baseline["best_epoch"],
            "metrics": res_baseline,
        },
        "custom_gradient_tape": {
            "training_method": "tf.GradientTape() + tf.data.Dataset + @tf.function",
            "duration": dur_custom,
            "best_epoch": hist_custom["best_epoch"],
            "metrics": res_custom,
        }
    }

    results_json_path = os.path.join(reports_dir, "phase4_results.json")
    with open(results_json_path, "w") as f:
        json.dump(phase4_results, f, indent=2)
    print(f"\nFinal Phase 4 results saved to: {results_json_path}")
    print("=" * 70)
    print(" EVALUATION SCRIPT FINISHED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()

