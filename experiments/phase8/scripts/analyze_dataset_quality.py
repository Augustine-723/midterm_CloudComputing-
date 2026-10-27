#!/usr/bin/env python3
"""
experiments/phase8/scripts/analyze_dataset_quality.py
Phase 8: Dataset Quality, Exact Duplicate & Near-Duplicate Ambiguity Audit.

Audits:
1. Dataset Integrity (Shape, Type, Class Distribution, NaNs/Infs, Normalization Range)
2. Exact Duplicate Analysis (In-Train, In-Val, Cross-Split, Type A vs Type B)
3. Near-Duplicate Analysis using cKDTree across multiple distance thresholds epsilon
4. Longitudinal Action Ambiguity (FASTER <-> SLOWER, IDLE <-> SLOWER, etc.)

Outputs figures:
- experiments/phase8/figures/phase8_label_distribution.png
- experiments/phase8/figures/phase8_exact_and_near_duplicates.png
- experiments/phase8/figures/phase8_faster_slower_ambiguity.png
"""

import os
import sys
import json
import time
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = ["0: LANE_LEFT", "1: IDLE", "2: LANE_RIGHT", "3: FASTER", "4: SLOWER"]
SHORT_NAMES = ["L_LEFT", "IDLE", "L_RIGHT", "FASTER", "SLOWER"]


def audit_integrity(data, name):
    """Check shapes, dtypes, NaNs, Infs, and min/max ranges."""
    x = data["data"]
    y = data["label"].flatten()
    n_samples, n_dims = x.shape

    nan_count = int(np.isnan(x).sum())
    inf_count = int(np.isinf(x).sum())
    min_val = float(x.min())
    max_val = float(x.max())
    mean_val = float(x.mean())
    std_val = float(x.std())

    class_counts = {CLASS_NAMES[c]: int(np.sum(y == c)) for c in range(5)}
    class_pcts = {name: round(count / n_samples * 100.0, 2) for name, count in class_counts.items()}

    return {
        "dataset_name": name,
        "n_samples": n_samples,
        "n_features": n_dims,
        "feature_dtype": str(x.dtype),
        "label_dtype": str(y.dtype),
        "has_nan": nan_count > 0,
        "has_inf": inf_count > 0,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "min_value": round(min_val, 4),
        "max_value": round(max_val, 4),
        "mean_value": round(mean_val, 4),
        "std_value": round(std_val, 4),
        "class_counts": class_counts,
        "class_percentages": class_pcts,
    }


def analyze_exact_duplicates(x_tr, y_tr, x_val, y_val):
    """Analyze exact duplicate vectors and distinguish Type A vs Type B."""
    # 1. Training set exact duplicates
    tr_hash_map = defaultdict(list)
    for i in range(len(x_tr)):
        k = x_tr[i].tobytes()
        tr_hash_map[k].append(y_tr[i])

    tr_unique_features = len(tr_hash_map)
    tr_total_duplicates = len(x_tr) - tr_unique_features
    type_a_tr = 0 # same label
    type_b_tr = 0 # conflicting label
    conflicting_feature_count_tr = 0
    duplicate_action_counts = defaultdict(int)

    for k, labels in tr_hash_map.items():
        if len(labels) > 1:
            if len(set(labels)) == 1:
                type_a_tr += (len(labels) - 1)
                for lab in labels:
                    duplicate_action_counts[lab] += 1
            else:
                type_b_tr += (len(labels) - 1)
                conflicting_feature_count_tr += 1

    # 2. Validation set exact duplicates
    val_hash_map = defaultdict(list)
    for i in range(len(x_val)):
        k = x_val[i].tobytes()
        val_hash_map[k].append(y_val[i])

    val_unique_features = len(val_hash_map)
    val_total_duplicates = len(x_val) - val_unique_features
    type_a_val = 0
    type_b_val = 0
    for k, labels in val_hash_map.items():
        if len(labels) > 1:
            if len(set(labels)) == 1:
                type_a_val += (len(labels) - 1)
            else:
                type_b_val += (len(labels) - 1)

    # 3. Cross-split duplicates (Val in Train)
    val_in_train_same = 0
    val_in_train_diff = 0
    cross_action_counts = defaultdict(int)
    for i in range(len(x_val)):
        k = x_val[i].tobytes()
        if k in tr_hash_map:
            tr_labs = tr_hash_map[k]
            if y_val[i] in tr_labs:
                val_in_train_same += 1
                cross_action_counts[y_val[i]] += 1
            else:
                val_in_train_diff += 1

    return {
        "train": {
            "total_samples": len(x_tr),
            "unique_feature_vectors": tr_unique_features,
            "total_duplicate_rows": tr_total_duplicates,
            "duplicate_rate_percent": round(tr_total_duplicates / len(x_tr) * 100.0, 2),
            "type_a_same_label": type_a_tr,
            "type_b_conflicting_label": type_b_tr,
            "conflicting_unique_features": conflicting_feature_count_tr,
            "duplicate_action_distribution": {
                CLASS_NAMES[c]: duplicate_action_counts[c] for c in range(5)
            }
        },
        "validation": {
            "total_samples": len(x_val),
            "unique_feature_vectors": val_unique_features,
            "total_duplicate_rows": val_total_duplicates,
            "duplicate_rate_percent": round(val_total_duplicates / len(x_val) * 100.0, 2),
            "type_a_same_label": type_a_val,
            "type_b_conflicting_label": type_b_val,
        },
        "cross_split": {
            "val_samples_in_train_same_label": val_in_train_same,
            "val_samples_in_train_diff_label": val_in_train_diff,
            "total_cross_split_duplicates": val_in_train_same + val_in_train_diff,
            "cross_split_rate_percent": round((val_in_train_same + val_in_train_diff) / len(x_val) * 100.0, 2),
            "action_distribution": {
                CLASS_NAMES[c]: cross_action_counts[c] for c in range(5)
            }
        }
    }


def analyze_near_duplicates(x_tr, y_tr):
    """
    Nearest neighbor sensitivity sweep across multiple distance thresholds.
    Investigates label disagreement rates and longitudinal conflict proportions.
    """
    print("\n[Near-Duplicate Analysis] Building cKDTree for 35,000 training samples...")
    tree = cKDTree(x_tr)
    # Query 2-NN to get nearest neighbor excluding self
    dists, indices = tree.query(x_tr, k=2)
    d_nn = dists[:, 1]
    idx_nn = indices[:, 1]
    y_nn = y_tr[idx_nn]

    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    sweep_results = []

    for eps in thresholds:
        mask = (d_nn <= eps)
        n_pairs = int(np.sum(mask))
        if n_pairs > 0:
            disagree_mask = mask & (y_tr != y_nn)
            disagree_count = int(np.sum(disagree_mask))
            disagree_rate = round(disagree_count / n_pairs * 100.0, 2)

            y1 = y_tr[disagree_mask]
            y2 = y_nn[disagree_mask]

            faster_slower = int(np.sum(((y1 == 3) & (y2 == 4)) | ((y1 == 4) & (y2 == 3))))
            idle_slower = int(np.sum(((y1 == 1) & (y2 == 4)) | ((y1 == 4) & (y2 == 1))))
            idle_faster = int(np.sum(((y1 == 1) & (y2 == 3)) | ((y1 == 3) & (y2 == 1))))
            lane_change = int(np.sum((y1 == 0) | (y2 == 0) | (y1 == 2) | (y2 == 2)))

            sweep_results.append({
                "epsilon": eps,
                "n_near_pairs": n_pairs,
                "pair_rate_percent": round(n_pairs / len(x_tr) * 100.0, 2),
                "disagree_count": disagree_count,
                "disagree_rate_percent": disagree_rate,
                "faster_slower_count": faster_slower,
                "faster_slower_share_percent": round(faster_slower / disagree_count * 100.0, 1) if disagree_count > 0 else 0.0,
                "idle_slower_count": idle_slower,
                "idle_slower_share_percent": round(idle_slower / disagree_count * 100.0, 1) if disagree_count > 0 else 0.0,
                "idle_faster_count": idle_faster,
                "idle_faster_share_percent": round(idle_faster / disagree_count * 100.0, 1) if disagree_count > 0 else 0.0,
                "lane_change_count": lane_change,
                "lane_change_share_percent": round(lane_change / disagree_count * 100.0, 1) if disagree_count > 0 else 0.0,
            })
        else:
            sweep_results.append({
                "epsilon": eps,
                "n_near_pairs": 0,
                "pair_rate_percent": 0.0,
                "disagree_count": 0,
                "disagree_rate_percent": 0.0,
            })

    # Distribution of distance to nearest same-label vs different-label
    same_dists = []
    diff_dists = []
    # Compute for a subset of 5,000 training samples for speed
    sub_indices = np.random.RandomState(42).choice(len(x_tr), size=5000, replace=False)
    sub_x = x_tr[sub_indices]
    sub_y = y_tr[sub_indices]

    d_all, idx_all = tree.query(sub_x, k=15)
    for i in range(len(sub_x)):
        labels_i = y_tr[idx_all[i]]
        dists_i = d_all[i]
        # Skip self (dist == 0)
        non_self = dists_i > 1e-7
        if not np.any(non_self):
            continue
        valid_labs = labels_i[non_self]
        valid_dists = dists_i[non_self]

        same_mask = (valid_labs == sub_y[i])
        diff_mask = (valid_labs != sub_y[i])

        if np.any(same_mask):
            same_dists.append(float(valid_dists[same_mask][0]))
        if np.any(diff_mask):
            diff_dists.append(float(valid_dists[diff_mask][0]))

    return {
        "sweep_sensitivity": sweep_results,
        "distance_to_same_label": {
            "mean": round(float(np.mean(same_dists)), 4),
            "median": round(float(np.median(same_dists)), 4),
            "std": round(float(np.std(same_dists)), 4),
            "p25": round(float(np.percentile(same_dists, 25)), 4),
            "p75": round(float(np.percentile(same_dists, 75)), 4),
        },
        "distance_to_different_label": {
            "mean": round(float(np.mean(diff_dists)), 4),
            "median": round(float(np.median(diff_dists)), 4),
            "std": round(float(np.std(diff_dists)), 4),
            "p25": round(float(np.percentile(diff_dists, 25)), 4),
            "p75": round(float(np.percentile(diff_dists, 75)), 4),
        }
    }


def plot_label_distribution(integrity_tr, integrity_val, output_path):
    """Plot comparative class distribution between Train and Validation sets."""
    x = np.arange(5)
    width = 0.35

    tr_pcts = [integrity_tr["class_percentages"][CLASS_NAMES[i]] for i in range(5)]
    val_pcts = [integrity_val["class_percentages"][CLASS_NAMES[i]] for i in range(5)]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    r1 = ax.bar(x - width/2, tr_pcts, width, label="Training Set (35,000 samples)", color="#1f77b4")
    r2 = ax.bar(x + width/2, val_pcts, width, label="Validation Set (5,000 samples)", color="#ff7f0e")

    ax.set_ylabel("Class Proportion (%)", fontsize=11)
    ax.set_title("Dataset Action Distribution Comparison (Train vs Validation)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_NAMES, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for r in r1:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for r in r2:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved label distribution figure to: {output_path}")


def plot_exact_and_near_duplicates(sweep_data, output_path):
    """Plot sensitivity curve of near-duplicate label disagreement across epsilon."""
    epsilons = [d["epsilon"] for d in sweep_data]
    rates = [d["disagree_rate_percent"] for d in sweep_data]
    pair_counts = [d["n_near_pairs"] for d in sweep_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)

    # Subplot 1: Label disagreement rate vs epsilon
    ax1.plot(epsilons, rates, marker="o", color="#d62728", linewidth=2, markersize=6)
    ax1.set_xlabel("Distance Threshold (ε)", fontsize=11)
    ax1.set_ylabel("Label Disagreement Rate (%)", fontsize=11)
    ax1.set_title("Near-Duplicate Label Disagreement Sensitivity\n(Training Dataset, 35,000 samples)",
                  fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    for ep, r in zip(epsilons, rates):
        if ep in [0.005, 0.01, 0.05, 0.1, 0.2]:
            ax1.annotate(f"{r:.1f}% (ε={ep})", xy=(ep, r), xytext=(5, 5),
                         textcoords="offset points", fontsize=8, fontweight="bold")

    # Subplot 2: Cumulative Near-Duplicate Pair Count
    ax2.plot(epsilons, pair_counts, marker="s", color="#1f77b4", linewidth=2, markersize=6)
    ax2.set_xlabel("Distance Threshold (ε)", fontsize=11)
    ax2.set_ylabel("Number of Samples with 1-NN <= ε", fontsize=11)
    ax2.set_title("Near-Duplicate Volume vs Distance Threshold", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved near duplicate sensitivity figure to: {output_path}")


def plot_faster_slower_ambiguity(sweep_data, output_path):
    """Plot composition of action conflicts among near-duplicates."""
    # Filter for epsilon >= 0.02
    sub = [d for d in sweep_data if d["epsilon"] >= 0.02 and d["disagree_count"] > 0]
    eps_labels = [f"ε={d['epsilon']}" for d in sub]
    fs = [d["faster_slower_share_percent"] for d in sub]
    is_ = [d["idle_slower_share_percent"] for d in sub]
    if_ = [d["idle_faster_share_percent"] for d in sub]
    lc = [d["lane_change_share_percent"] for d in sub]

    x = np.arange(len(eps_labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    p1 = ax.bar(x, fs, width, label="FASTER <-> SLOWER", color="#d62728")
    p2 = ax.bar(x, is_, width, bottom=fs, label="IDLE <-> SLOWER", color="#ff7f0e")
    bottom_3 = np.array(fs) + np.array(is_)
    p3 = ax.bar(x, if_, width, bottom=bottom_3, label="IDLE <-> FASTER", color="#bcbd22")
    bottom_4 = bottom_3 + np.array(if_)
    p4 = ax.bar(x, lc, width, bottom=bottom_4, label="Lane Changes Involved", color="#1f77b4")

    ax.set_ylabel("Share of All Label Disagreements (%)", fontsize=11)
    ax.set_title("Near-Duplicate Action Conflict Breakdown Across Distance Thresholds",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(eps_labels, fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved faster/slower ambiguity figure to: {output_path}")


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    phase_dir = os.path.join(repo_root, "experiments", "phase8")
    reports_dir = os.path.join(phase_dir, "reports")
    figures_dir = os.path.join(phase_dir, "figures")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 75)
    print(" PHASE 8 — DATASET QUALITY & DUPLICATE AMBIGUITY ANALYSIS")
    print("=" * 75)

    # 1. Load Data
    tr_path = os.path.join(repo_root, "dataset", "train.npz")
    val_path = os.path.join(repo_root, "dataset", "validation.npz")
    train = np.load(tr_path)
    val = np.load(val_path)

    x_tr = train["data"].astype(np.float32)
    y_tr = train["label"].astype(np.int32).flatten()
    x_val = val["data"].astype(np.float32)
    y_val = val["label"].astype(np.int32).flatten()

    # 2. Audit Integrity
    print("\n[1] Checking Dataset Integrity & Dimensional Axioms...")
    integrity_tr = audit_integrity(train, "train.npz")
    integrity_val = audit_integrity(val, "validation.npz")
    print(f"  Train: {integrity_tr['n_samples']} samples, range: [{integrity_tr['min_value']}, {integrity_tr['max_value']}], NaNs: {integrity_tr['nan_count']}")
    print(f"  Val:   {integrity_val['n_samples']} samples, range: [{integrity_val['min_value']}, {integrity_val['max_value']}], NaNs: {integrity_val['nan_count']}")

    # 3. Exact Duplicate Analysis
    print("\n[2] Executing Exact Duplicate Analysis...")
    dup_results = analyze_exact_duplicates(x_tr, y_tr, x_val, y_val)
    print(f"  Train Exact Duplicates: {dup_results['train']['total_duplicate_rows']} "
          f"({dup_results['train']['duplicate_rate_percent']}%)")
    print(f"    - Type A (Same Label):        {dup_results['train']['type_a_same_label']}")
    print(f"    - Type B (Conflicting Label): {dup_results['train']['type_b_conflicting_label']}")
    print(f"  Val Exact Duplicates:   {dup_results['validation']['total_duplicate_rows']} "
          f"({dup_results['validation']['duplicate_rate_percent']}%)")
    print(f"  Cross-Split Duplicates: {dup_results['cross_split']['total_cross_split_duplicates']} "
          f"({dup_results['cross_split']['cross_split_rate_percent']}%)")
    print(f"    - Matching Label:    {dup_results['cross_split']['val_samples_in_train_same_label']}")
    print(f"    - Conflicting Label: {dup_results['cross_split']['val_samples_in_train_diff_label']}")

    # 4. Near-Duplicate Analysis
    near_dup_results = analyze_near_duplicates(x_tr, y_tr)

    # 5. Visualizations
    print("\n[3] Generating Visualizations...")
    fig1 = os.path.join(figures_dir, "phase8_label_distribution.png")
    plot_label_distribution(integrity_tr, integrity_val, fig1)

    fig2 = os.path.join(figures_dir, "phase8_exact_and_near_duplicates.png")
    plot_exact_and_near_duplicates(near_dup_results["sweep_sensitivity"], fig2)

    fig3 = os.path.join(figures_dir, "phase8_faster_slower_ambiguity.png")
    plot_faster_slower_ambiguity(near_dup_results["sweep_sensitivity"], fig3)

    # 6. Export Interim JSON
    quality_summary = {
        "integrity": {
            "train": integrity_tr,
            "validation": integrity_val,
        },
        "exact_duplicates": dup_results,
        "near_duplicates": near_dup_results,
    }
    out_json = os.path.join(phase_dir, "dataset_quality_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(quality_summary, f, indent=2)
    print(f"\nSaved dataset quality summary to: {out_json}")


if __name__ == "__main__":
    main()
