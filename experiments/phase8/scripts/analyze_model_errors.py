#!/usr/bin/env python3
"""
experiments/phase8/scripts/analyze_model_errors.py
Phase 8: Multi-Model Validation Error Overlap & Confidence/Entropy Analysis.

Models evaluated on dataset/validation.npz (5,000 samples):
1. Control Baseline Standard:    models/baseline_standard.h5
2. Distributed CPU Best:         models/distributed_cpu_best.h5
3. GradientTape Custom Best:     models/gradient_tape_best.h5
4. Unified Seed Ensemble:        experiments/phase7b/models/ensemble_unified.h5

Calculates:
- Cross-model error overlap matrix
- Persistent errors (all 4 models wrong)
- Model-unique errors (e.g. baseline-only errors)
- High-confidence incorrect predictions (confidence >= 0.80, 0.90)
- Prediction entropy distributions

Outputs figures:
- experiments/phase8/figures/phase8_error_overlap_heatmap.png
- experiments/phase8/figures/phase8_confidence_and_entropy.png
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
SHORT_NAMES = ["L_LEFT", "IDLE", "L_RIGHT", "FASTER", "SLOWER"]


def compute_sample_entropy(probs):
    """Compute Shannon entropy (base e) for probability distributions."""
    eps = 1e-12
    p_clipped = np.clip(probs, eps, 1.0)
    return -np.sum(p_clipped * np.log(p_clipped), axis=-1)


def plot_error_overlap_heatmap(overlap_matrix, model_names, error_counts, output_path):
    """Plot heatmap of cross-model error intersection counts and percentages."""
    n = len(model_names)
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=150)

    # Normalized overlap by row (what % of Model i's errors are also made by Model j)
    pct_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            pct_matrix[i, j] = overlap_matrix[i, j] / error_counts[i] * 100.0

    im = ax.imshow(pct_matrix, cmap="Reds", vmin=50, vmax=100)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("% Overlap with Row Model's Errors", rotation=-90, va="bottom", fontsize=10)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    labels = [f"{m}\n({error_counts[i]} errs)" for i, m in enumerate(model_names)]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(n):
        for j in range(n):
            count = overlap_matrix[i, j]
            pct = pct_matrix[i, j]
            color = "white" if pct > 75 else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center", color=color,
                    fontsize=9, fontweight="bold")

    ax.set_title("Cross-Model Error Overlap Matrix (Validation Set, 5,000 samples)",
                 fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved error overlap heatmap to: {output_path}")


def plot_confidence_and_entropy(base_conf, base_ent, y_val, p_base, persistent_idx, output_path):
    """Plot prediction confidence and entropy comparison across error partitions."""
    correct_mask = (p_base == y_val)
    error_mask = ~correct_mask

    is_persistent = np.zeros(len(y_val), dtype=bool)
    is_persistent[persistent_idx] = True

    non_persistent_errors = error_mask & (~is_persistent)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # 1. Prediction Confidence Boxplots / Histogram
    bins = np.linspace(0.2, 1.0, 33)
    ax1.hist(base_conf[correct_mask], bins=bins, density=True, alpha=0.6, color="#2ca02c", label=f"Correct ({np.sum(correct_mask)})")
    ax1.hist(base_conf[non_persistent_errors], bins=bins, density=True, alpha=0.6, color="#ff7f0e", label=f"Rescuable Errors ({np.sum(non_persistent_errors)})")
    ax1.hist(base_conf[is_persistent], bins=bins, density=True, alpha=0.6, color="#d62728", label=f"Persistent Errors ({np.sum(is_persistent)})")
    ax1.set_xlabel("Prediction Confidence (Max Softmax Probability)", fontsize=10)
    ax1.set_ylabel("Probability Density", fontsize=10)
    ax1.set_title("Prediction Confidence Distribution\n(Baseline Model)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Prediction Entropy Histogram
    ent_bins = np.linspace(0.0, 1.6, 33)
    ax2.hist(base_ent[correct_mask], bins=ent_bins, density=True, alpha=0.6, color="#2ca02c", label=f"Correct ({np.sum(correct_mask)})")
    ax2.hist(base_ent[non_persistent_errors], bins=ent_bins, density=True, alpha=0.6, color="#ff7f0e", label=f"Rescuable Errors ({np.sum(non_persistent_errors)})")
    ax2.hist(base_ent[is_persistent], bins=ent_bins, density=True, alpha=0.6, color="#d62728", label=f"Persistent Errors ({np.sum(is_persistent)})")
    ax2.set_xlabel("Prediction Shannon Entropy (nats)", fontsize=10)
    ax2.set_ylabel("Probability Density", fontsize=10)
    ax2.set_title("Prediction Entropy Distribution\n(Higher Entropy = More Ambiguous)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved confidence and entropy figure to: {output_path}")


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    phase_dir = os.path.join(repo_root, "experiments", "phase8")
    figures_dir = os.path.join(phase_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 75)
    print(" PHASE 8 — MULTI-MODEL VALIDATION ERROR OVERLAP ANALYSIS")
    print("=" * 75)

    val_path = os.path.join(repo_root, "dataset", "validation.npz")
    val_npz = np.load(val_path)
    x_val = val_npz["data"].astype(np.float32)
    y_val = val_npz["label"].astype(np.int32).flatten()
    n_val = len(y_val)

    # Model definitions
    model_configs = [
        ("Baseline Standard", os.path.join(repo_root, "models", "baseline_standard.h5")),
        ("Distributed CPU", os.path.join(repo_root, "models", "distributed_cpu_best.h5")),
        ("GradientTape Custom", os.path.join(repo_root, "models", "gradient_tape_best.h5")),
        ("Unified Ensemble", os.path.join(repo_root, "experiments", "phase7b", "models", "ensemble_unified.h5")),
    ]

    model_names = [m[0] for m in model_configs]
    model_probs = {}
    model_preds = {}
    model_conf = {}
    model_ent = {}
    error_sets = {}

    for name, path in model_configs:
        print(f"Loading & Evaluating {name}...")
        m = tf.keras.models.load_model(path, compile=False)
        probs = m.predict(x_val, batch_size=128, verbose=0)
        preds = np.argmax(probs, axis=1)
        conf = np.max(probs, axis=1)
        ent = compute_sample_entropy(probs)

        model_probs[name] = probs
        model_preds[name] = preds
        model_conf[name] = conf
        model_ent[name] = ent

        err_idx = set(np.where(preds != y_val)[0])
        error_sets[name] = err_idx
        acc = float(np.mean(preds == y_val))
        print(f"  -> Accuracy: {acc*100:.2f}% | Errors: {len(err_idx)} / {n_val}")

    # Overlap Matrix
    n_models = len(model_names)
    overlap_matrix = np.zeros((n_models, n_models), dtype=int)
    error_counts = [len(error_sets[name]) for name in model_names]

    for i, name_i in enumerate(model_names):
        for j, name_j in enumerate(model_names):
            overlap_matrix[i, j] = len(error_sets[name_i] & error_sets[name_j])

    # Intersections & Unions
    persistent_errors = set.intersection(*[error_sets[m] for m in model_names])
    union_errors = set.union(*[error_sets[m] for m in model_names])
    baseline_only_errors = error_sets["Baseline Standard"] - set.union(*[error_sets[m] for m in model_names if m != "Baseline Standard"])

    # High-confidence error analysis
    high_conf_analysis = {}
    for name in model_names:
        err_mask = (model_preds[name] != y_val)
        conf = model_conf[name]
        high_80 = int(np.sum(err_mask & (conf >= 0.80)))
        high_90 = int(np.sum(err_mask & (conf >= 0.90)))
        high_conf_analysis[name] = {
            "total_errors": len(error_sets[name]),
            "errors_conf_ge_80": high_80,
            "errors_conf_ge_80_percent": round(high_80 / len(error_sets[name]) * 100.0, 2),
            "errors_conf_ge_90": high_90,
            "errors_conf_ge_90_percent": round(high_90 / len(error_sets[name]) * 100.0, 2),
            "mean_error_confidence": round(float(np.mean(conf[err_mask])), 4),
            "mean_correct_confidence": round(float(np.mean(conf[~err_mask])), 4),
            "mean_error_entropy": round(float(np.mean(model_ent[name][err_mask])), 4),
            "mean_correct_entropy": round(float(np.mean(model_ent[name][~err_mask])), 4),
        }

    # Ensemble vs Baseline Detailed Transitions
    base_errs = error_sets["Baseline Standard"]
    ens_errs = error_sets["Unified Ensemble"]
    fixed_by_ens = sorted(list(base_errs - ens_errs))
    introduced_by_ens = sorted(list(ens_errs - base_errs))

    print("\n[Cross-Model Error Summary]")
    print(f"  Persistent Errors (All 4 Models Wrong):  {len(persistent_errors)} samples ({len(persistent_errors)/n_val*100:.2f}%)")
    print(f"  Union of All Errors (Any Model Wrong):   {len(union_errors)} samples ({len(union_errors)/n_val*100:.2f}%)")
    print(f"  Baseline-Only Errors (Unique to Base):   {len(baseline_only_errors)} samples")
    print(f"  Ensemble Fixed vs Baseline:              {len(fixed_by_ens)} samples")
    print(f"  Ensemble Introduced vs Baseline:         {len(introduced_by_ens)} samples")

    # Generate Figures
    fig_heatmap = os.path.join(figures_dir, "phase8_error_overlap_heatmap.png")
    plot_error_overlap_heatmap(overlap_matrix, model_names, error_counts, fig_heatmap)

    fig_conf = os.path.join(figures_dir, "phase8_confidence_and_entropy.png")
    plot_confidence_and_entropy(
        model_conf["Baseline Standard"],
        model_ent["Baseline Standard"],
        y_val,
        model_preds["Baseline Standard"],
        list(persistent_errors),
        fig_conf
    )

    # Save Intermediate Results
    error_summary = {
        "model_error_counts": {name: int(len(error_sets[name])) for name in model_names},
        "model_accuracies": {name: round(float(np.mean(model_preds[name] == y_val)), 4) for name in model_names},
        "overlap_matrix": [[int(val) for val in row] for row in overlap_matrix],
        "persistent_error_count": int(len(persistent_errors)),
        "persistent_error_indices": [int(i) for i in sorted(list(persistent_errors))],
        "union_error_count": int(len(union_errors)),
        "baseline_only_error_count": int(len(baseline_only_errors)),
        "ensemble_fixed_count": int(len(fixed_by_ens)),
        "ensemble_introduced_count": int(len(introduced_by_ens)),
        "high_confidence_analysis": high_conf_analysis,
    }

    out_json = os.path.join(phase_dir, "model_errors_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(error_summary, f, indent=2)
    print(f"\nSaved model error summary to: {out_json}")


if __name__ == "__main__":
    main()
