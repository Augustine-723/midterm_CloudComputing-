#!/usr/bin/env python3
"""
experiments/phase7b/scripts/evaluate_ensemble.py
Comprehensive evaluation of Seed Ensemble vs Individual Seeds vs Baseline Control.

Evaluates on:
1. Internal Development Set (7,000 samples, 80/20 train.npz split)
2. Official Validation Set (5,000 samples, dataset/validation.npz)

Calculates:
- Accuracy, Loss, Macro Precision, Macro Recall, Macro F1, Weighted F1
- Per-class Precision, Recall, F1
- Confusion Matrices
- Prediction Disagreement & Voting Diversity Analysis (3-way agreement, 2-1 split, all-disagree)
- FASTER <-> SLOWER longitudinal misclassification frequency
- Inference Latency (Batch & Single-Sample)
- Teacher compatibility & roundabout-v0 environment test

Outputs:
- experiments/phase7b/reports/phase7b_results.json
- experiments/phase7b/figures/phase7b_confusion_matrices.png
- experiments/phase7b/figures/phase7b_metric_comparison.png
- experiments/phase7b/figures/phase7b_seed_diversity.png
"""

import json
import os
import sys
import time
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
SHORT_NAMES = ["L_LEFT", "IDLE", "L_RIGHT", "FASTER", "SLOWER"]


def compute_classification_metrics(y_true, y_pred, y_prob=None):
    """Compute complete classification metrics with pure NumPy."""
    n_classes = len(CLASS_NAMES)
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
        support = int(cm[i, :].sum())

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)

        per_class[CLASS_NAMES[i]] = {
            "class_idx": i,
            "name": SHORT_NAMES[i],
            "support": support,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    total_samples = len(y_true)
    accuracy = float(np.sum(np.diag(cm)) / total_samples)
    macro_prec = float(np.mean(precisions))
    macro_rec = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    weighted_f1 = float(np.sum(np.array(f1s) * np.array(supports)) / total_samples)

    loss = None
    if y_prob is not None:
        eps = 1e-12
        y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
        # normalize row sums
        y_prob_clipped = y_prob_clipped / y_prob_clipped.sum(axis=1, keepdims=True)
        log_preds = -np.log(y_prob_clipped[np.arange(total_samples), y_true])
        loss = float(np.mean(log_preds))

    # Longitudinal FASTER <-> SLOWER confusion
    faster_as_slower = int(cm[3, 4])
    slower_as_faster = int(cm[4, 3])
    total_longitudinal_confusion = faster_as_slower + slower_as_faster

    return {
        "accuracy": round(accuracy, 4),
        "loss": round(loss, 4) if loss is not None else None,
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "total_errors": int(total_samples - np.sum(np.diag(cm))),
        "faster_as_slower": faster_as_slower,
        "slower_as_faster": slower_as_faster,
        "longitudinal_confusion_total": total_longitudinal_confusion,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def analyze_seed_diversity(preds_dict, y_true):
    """
    Analyze agreement, disagreement, and voting patterns across Seed 42, 123, and 2026.
    preds_dict: {'seed_42': pred, 'seed_123': pred, 'seed_2026': pred, 'ensemble': pred}
    """
    p42 = preds_dict["seed_42"]
    p123 = preds_dict["seed_123"]
    p2026 = preds_dict["seed_2026"]
    pens = preds_dict["ensemble"]
    n = len(y_true)

    # Pairwise disagreement
    disagree_42_123 = float(np.mean(p42 != p123))
    disagree_42_2026 = float(np.mean(p42 != p2026))
    disagree_123_2026 = float(np.mean(p123 != p2026))

    # Voting pattern per sample
    # Case 1: Unanimous (all 3 agree)
    unanimous = (p42 == p123) & (p123 == p2026)
    # Case 2: Complete disagreement (all 3 differ)
    complete_disagree = (p42 != p123) & (p123 != p2026) & (p42 != p2026)
    # Case 3: Majority split (2 agree, 1 differs)
    majority_split = (~unanimous) & (~complete_disagree)

    unanimous_count = int(np.sum(unanimous))
    majority_split_count = int(np.sum(majority_split))
    complete_disagree_count = int(np.sum(complete_disagree))

    # When seeds disagree, does ensemble predict correctly?
    disagree_any = ~unanimous
    ensemble_correct = pens == y_true
    ensemble_acc_on_disagreements = (
        float(np.mean(ensemble_correct[disagree_any]))
        if np.sum(disagree_any) > 0 else 0.0
    )

    # Agreement with ensemble
    agree_ens_42 = float(np.mean(p42 == pens))
    agree_ens_123 = float(np.mean(p123 == pens))
    agree_ens_2026 = float(np.mean(p2026 == pens))

    return {
        "pairwise_disagreement": {
            "seed42_vs_seed123": round(disagree_42_123, 4),
            "seed42_vs_seed2026": round(disagree_42_2026, 4),
            "seed123_vs_seed2026": round(disagree_123_2026, 4),
            "mean_pairwise_disagreement": round(float(np.mean([disagree_42_123, disagree_42_2026, disagree_123_2026])), 4),
        },
        "voting_distribution": {
            "unanimous_count": unanimous_count,
            "unanimous_rate": round(unanimous_count / n, 4),
            "majority_split_count": majority_split_count,
            "majority_split_rate": round(majority_split_count / n, 4),
            "complete_disagreement_count": complete_disagree_count,
            "complete_disagreement_rate": round(complete_disagree_count / n, 4),
        },
        "ensemble_benefit": {
            "total_disagreed_samples": int(np.sum(disagree_any)),
            "ensemble_accuracy_on_disagreed_samples": round(ensemble_acc_on_disagreements, 4),
            "agreement_seed42_with_ensemble": round(agree_ens_42, 4),
            "agreement_seed123_with_ensemble": round(agree_ens_123, 4),
            "agreement_seed2026_with_ensemble": round(agree_ens_2026, 4),
        }
    }


def measure_latencies(model, x_val, n_single=200):
    """Measure batch latency and single-sample latency."""
    # Warmup
    _ = model.predict(x_val[:32], verbose=0)

    # 1. Full batch inference (5,000 samples)
    t0 = time.perf_counter()
    _ = model.predict(x_val, batch_size=64, verbose=0)
    batch_time_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Single sample inference
    single_samples = x_val[:n_single]
    single_times = []
    for i in range(n_single):
        s_in = single_samples[i:i+1]
        t_s0 = time.perf_counter()
        _ = model(s_in, training=False).numpy()
        single_times.append((time.perf_counter() - t_s0) * 1000.0)

    return {
        "batch_total_ms": round(batch_time_ms, 2),
        "batch_per_sample_ms": round(batch_time_ms / len(x_val), 4),
        "single_sample_mean_ms": round(float(np.mean(single_times)), 4),
        "single_sample_std_ms": round(float(np.std(single_times)), 4),
    }


def plot_confusion_matrices(cm_dict, output_path):
    """Plot multi-panel confusion matrix comparison."""
    n_models = len(cm_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), dpi=150)
    if n_models == 1:
        axes = [axes]

    for ax, (title, cm) in zip(axes, cm_dict.items()):
        cm = np.array(cm)
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(title, fontsize=12, pad=10, fontweight="bold")
        tick_marks = np.arange(len(SHORT_NAMES))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(SHORT_NAMES, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(SHORT_NAMES, fontsize=9)
        ax.set_ylabel("True Action", fontsize=10)
        ax.set_xlabel("Predicted Action", fontsize=10)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved confusion matrices figure to: {output_path}")


def plot_metric_comparison(summary_dict, output_path):
    """Plot bar chart comparing models on Accuracy, Macro F1, and FASTER<->SLOWER errors."""
    model_names = list(summary_dict.keys())
    accs = [summary_dict[m]["accuracy"] * 100 for m in model_names]
    f1s = [summary_dict[m]["macro_f1"] * 100 for m in model_names]
    long_errs = [summary_dict[m]["longitudinal_confusion_total"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.28

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Subplot 1: Acc & Macro F1
    r1 = ax1.bar(x - width/2, accs, width, label="Accuracy (%)", color="#1f77b4")
    r2 = ax1.bar(x + width/2, f1s, width, label="Macro F1 (%)", color="#ff7f0e")
    ax1.set_ylabel("Percentage (%)", fontsize=11)
    ax1.set_title("Validation Accuracy & Macro F1 Comparison", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=20, ha="right", fontsize=10)
    ax1.set_ylim(85, 96)
    ax1.legend(loc="lower right")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for r in r1:
        h = r.get_height()
        ax1.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for r in r2:
        h = r.get_height()
        ax1.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # Subplot 2: Longitudinal Errors
    colors = ["#7f7f7f", "#2ca02c", "#d62728", "#9467bd", "#e377c2"][:len(model_names)]
    r3 = ax2.bar(x, long_errs, width * 1.5, color=colors)
    ax2.set_ylabel("Error Count (Samples)", fontsize=11)
    ax2.set_title("FASTER <-> SLOWER Confusion Errors (Lower is Better)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=20, ha="right", fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for r in r3:
        h = r.get_height()
        ax2.annotate(f"{int(h)}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved metric comparison figure to: {output_path}")


def plot_seed_diversity(diversity_res, output_path):
    """Plot pie chart and breakdown of seed voting agreement."""
    dist = diversity_res["voting_distribution"]
    labels = [
        f"Unanimous (3 Agree)\n{dist['unanimous_count']} ({dist['unanimous_rate']*100:.1f}%)",
        f"Majority (2 vs 1 Split)\n{dist['majority_split_count']} ({dist['majority_split_rate']*100:.1f}%)",
        f"Complete Disagreement (3 Differ)\n{dist['complete_disagreement_count']} ({dist['complete_disagreement_rate']*100:.1f}%)"
    ]
    sizes = [
        dist["unanimous_count"],
        dist["majority_split_count"],
        dist["complete_disagreement_count"]
    ]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    explode = (0.02, 0.05, 0.1)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140, pctdistance=0.75,
        textprops=dict(color="black", fontsize=10)
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_weight("bold")

    ax.set_title("Seed Ensemble Prediction Agreement Breakdown\n(5,000 Validation Samples)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved seed diversity figure to: {output_path}")


def test_roundabout_environment_compatibility(model_path):
    """
    Simulate running 1 episode in highway-env roundabout-v0 using the exact teacher test pattern.
    Verifies input/output, action argmax, and zero-exception execution.
    """
    print("\n" + "=" * 75)
    print(" [TEACHER COMPATIBILITY] Testing roundabout-v0 Simulation Execution")
    print("=" * 75)
    import gymnasium as gym
    import highway_env

    print(f"Loading candidate model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded successfully via tf.keras.models.load_model().")
    print(f"Model Input Shape:  {model.input_shape}")
    print(f"Model Output Shape: {model.output_shape}")

    env = gym.make("roundabout-v0", render_mode=None)
    obs, info = env.reset(seed=42)
    total_reward = 0.0
    steps = 0
    done = False
    truncated = False

    while not (done or truncated):
        obs_tensor = obs.reshape(1, 25)
        action_prob = model.predict(obs_tensor, verbose=0)
        action = int(np.argmax(action_prob))

        assert 0 <= action <= 4, f"Invalid action: {action}"
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    env.close()
    print(f"  [PASS] Simulation completed successfully: {steps} steps, Episode Reward: {total_reward:.2f}")
    return {
        "status": "PASS",
        "steps": steps,
        "reward": float(total_reward),
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
    }


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    phase_dir = os.path.join(repo_root, "experiments", "phase7b")
    models_dir = os.path.join(phase_dir, "models")
    reports_dir = os.path.join(phase_dir, "reports")
    figures_dir = os.path.join(phase_dir, "figures")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 75)
    print(" PHASE 7B — COMPREHENSIVE EVALUATION OF SEED ENSEMBLE")
    print("=" * 75)

    # 1. Load Data
    split_info_path = os.path.join(phase_dir, "split_indices.npz")
    if not os.path.exists(split_info_path):
        raise FileNotFoundError(f"Missing split indices file: {split_info_path}")

    split_npz = np.load(split_info_path)
    dev_idx = split_npz["dev_indices"]

    train_npz = np.load(os.path.join(repo_root, "dataset", "train.npz"))
    x_train_full = train_npz["data"].astype(np.float32)
    y_train_full = train_npz["label"].astype(np.int32).flatten()
    x_dev = x_train_full[dev_idx]
    y_dev = y_train_full[dev_idx]

    val_npz = np.load(os.path.join(repo_root, "dataset", "validation.npz"))
    x_val = val_npz["data"].astype(np.float32)
    y_val = val_npz["label"].astype(np.int32).flatten()

    print(f"Dataset Loaded:")
    print(f"  Internal Dev Set:     {x_dev.shape} (from train.npz 80/20 split)")
    print(f"  Official Validation:  {x_val.shape} (dataset/validation.npz)")

    # 2. Load Models
    model_paths = {
        "Control (Baseline)": os.path.join(repo_root, "models", "baseline_standard.h5"),
        "Seed 42": os.path.join(models_dir, "model_seed42.h5"),
        "Seed 123": os.path.join(models_dir, "model_seed123.h5"),
        "Seed 2026": os.path.join(models_dir, "model_seed2026.h5"),
        "Unified Ensemble": os.path.join(models_dir, "ensemble_unified.h5"),
    }

    models = {}
    for name, p in model_paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Model file not found: {p}")
        print(f"Loading {name} from {os.path.relpath(p, repo_root)}...")
        models[name] = tf.keras.models.load_model(p, compile=False)

    # 3. Predict & Compute Metrics on Internal Dev Set
    print("\n" + "=" * 75)
    print(" [A] Evaluating Models on Internal Development Set (7,000 samples)")
    print("=" * 75)

    dev_results = {}
    p42_dev = models["Seed 42"].predict(x_dev, verbose=0)
    p123_dev = models["Seed 123"].predict(x_dev, verbose=0)
    p2026_dev = models["Seed 2026"].predict(x_dev, verbose=0)
    pens_soft_dev = (p42_dev + p123_dev + p2026_dev) / 3.0

    preds_prob_dev = {
        "Control (Baseline)": models["Control (Baseline)"].predict(x_dev, verbose=0),
        "Seed 42": p42_dev,
        "Seed 123": p123_dev,
        "Seed 2026": p2026_dev,
        "Soft Voting Ens": pens_soft_dev,
        "Unified Ensemble": models["Unified Ensemble"].predict(x_dev, verbose=0),
    }

    for name, prob in preds_prob_dev.items():
        pred_label = np.argmax(prob, axis=1)
        res = compute_classification_metrics(y_dev, pred_label, y_prob=prob)
        dev_results[name] = res
        print(f"  {name:20s} | Acc: {res['accuracy']*100:.2f}% | Loss: {res['loss']:.4f} | "
              f"Macro F1: {res['macro_f1']*100:.2f}% | Long Errs: {res['longitudinal_confusion_total']}")

    # 4. Predict & Compute Metrics on Official Validation Set (5,000 samples)
    print("\n" + "=" * 75)
    print(" [B] Evaluating Models on Official Validation Set (5,000 samples)")
    print("=" * 75)

    val_results = {}
    p42_val = models["Seed 42"].predict(x_val, verbose=0)
    p123_val = models["Seed 123"].predict(x_val, verbose=0)
    p2026_val = models["Seed 2026"].predict(x_val, verbose=0)
    pens_soft_val = (p42_val + p123_val + p2026_val) / 3.0

    preds_prob_val = {
        "Control (Baseline)": models["Control (Baseline)"].predict(x_val, verbose=0),
        "Seed 42": p42_val,
        "Seed 123": p123_val,
        "Seed 2026": p2026_val,
        "Soft Voting Ens": pens_soft_val,
        "Unified Ensemble": models["Unified Ensemble"].predict(x_val, verbose=0),
    }

    val_cms = {}
    for name, prob in preds_prob_val.items():
        pred_label = np.argmax(prob, axis=1)
        res = compute_classification_metrics(y_val, pred_label, y_prob=prob)
        val_results[name] = res
        val_cms[name] = res["confusion_matrix"]
        print(f"  {name:20s} | Acc: {res['accuracy']*100:.2f}% | Loss: {res['loss']:.4f} | "
              f"Macro F1: {res['macro_f1']*100:.2f}% | Long Errs: {res['longitudinal_confusion_total']}")

    # 5. Seed Diversity & Disagreement Analysis
    print("\n" + "=" * 75)
    print(" [C] Seed Diversity & Disagreement Analysis (Validation Set)")
    print("=" * 75)
    seed_preds = {
        "seed_42": np.argmax(p42_val, axis=1),
        "seed_123": np.argmax(p123_val, axis=1),
        "seed_2026": np.argmax(p2026_val, axis=1),
        "ensemble": np.argmax(pens_soft_val, axis=1),
    }
    diversity_metrics = analyze_seed_diversity(seed_preds, y_val)
    print(f"  Mean Pairwise Disagreement:  {diversity_metrics['pairwise_disagreement']['mean_pairwise_disagreement']*100:.2f}%")
    print(f"  Unanimous Agreement (3/3):   {diversity_metrics['voting_distribution']['unanimous_rate']*100:.2f}% "
          f"({diversity_metrics['voting_distribution']['unanimous_count']} samples)")
    print(f"  Majority Split (2 vs 1):     {diversity_metrics['voting_distribution']['majority_split_rate']*100:.2f}% "
          f"({diversity_metrics['voting_distribution']['majority_split_count']} samples)")
    print(f"  Complete Disagreement (3/3): {diversity_metrics['voting_distribution']['complete_disagreement_rate']*100:.2f}% "
          f"({diversity_metrics['voting_distribution']['complete_disagreement_count']} samples)")
    print(f"  Ensemble Acc on Disagreed:   {diversity_metrics['ensemble_benefit']['ensemble_accuracy_on_disagreed_samples']*100:.2f}%")

    # 6. Latency Benchmarks
    print("\n" + "=" * 75)
    print(" [D] Measuring Inference Latency Benchmarks")
    print("=" * 75)
    latencies = {}
    for name in ["Control (Baseline)", "Seed 42", "Unified Ensemble"]:
        print(f"  Benchmarking {name}...")
        lat = measure_latencies(models[name], x_val, n_single=200)
        latencies[name] = lat
        print(f"    Batch: {lat['batch_total_ms']:.2f} ms | Single Sample Mean: {lat['single_sample_mean_ms']:.4f} ms")

    # 7. Teacher Compatibility Check
    compat_info = test_roundabout_environment_compatibility(model_paths["Unified Ensemble"])

    # 8. Visualizations
    print("\n" + "=" * 75)
    print(" [E] Generating Visualizations")
    print("=" * 75)
    cm_plot_dict = {
        "Control (94.64%)": val_cms["Control (Baseline)"],
        "Seed 42 (94.20%)": val_cms["Seed 42"],
        "Seed 123 (94.46%)": val_cms["Seed 123"],
        "Seed 2026 (94.38%)": val_cms["Seed 2026"],
        "Ensemble (94.70%)": val_cms["Unified Ensemble"],
    }
    # Update title percentages with real values
    cm_plot_dict = {
        f"Control ({val_results['Control (Baseline)']['accuracy']*100:.2f}%)": val_cms["Control (Baseline)"],
        f"Seed 42 ({val_results['Seed 42']['accuracy']*100:.2f}%)": val_cms["Seed 42"],
        f"Seed 123 ({val_results['Seed 123']['accuracy']*100:.2f}%)": val_cms["Seed 123"],
        f"Seed 2026 ({val_results['Seed 2026']['accuracy']*100:.2f}%)": val_cms["Seed 2026"],
        f"Ensemble ({val_results['Unified Ensemble']['accuracy']*100:.2f}%)": val_cms["Unified Ensemble"],
    }
    cm_fig_path = os.path.join(figures_dir, "phase7b_confusion_matrices.png")
    plot_confusion_matrices(cm_plot_dict, cm_fig_path)

    metric_summary_dict = {
        "Control": val_results["Control (Baseline)"],
        "Seed 42": val_results["Seed 42"],
        "Seed 123": val_results["Seed 123"],
        "Seed 2026": val_results["Seed 2026"],
        "Ensemble": val_results["Unified Ensemble"],
    }
    metric_fig_path = os.path.join(figures_dir, "phase7b_metric_comparison.png")
    plot_metric_comparison(metric_summary_dict, metric_fig_path)

    diversity_fig_path = os.path.join(figures_dir, "phase7b_seed_diversity.png")
    plot_seed_diversity(diversity_metrics, diversity_fig_path)

    # 9. Save Results JSON
    final_output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tf_version": tf.__version__,
            "python_version": sys.version.split()[0],
            "seeds": [42, 123, 2026],
            "split": "80/20 stratified from dataset/train.npz",
            "val_samples": len(y_val),
            "dev_samples": len(y_dev),
        },
        "internal_dev_results": dev_results,
        "official_validation_results": val_results,
        "seed_diversity_analysis": diversity_metrics,
        "latencies": latencies,
        "teacher_compatibility": compat_info,
    }

    json_output_path = os.path.join(reports_dir, "phase7b_results.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nFinal results exported to: {json_output_path}")
    print("=" * 75)
    print(" PHASE 7B EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    main()

