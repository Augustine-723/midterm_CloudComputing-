#!/usr/bin/env python3
"""Reproduce duplicate-aware validation and generalization analyses for Phase 9."""

import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[3]
PHASE_DIR = ROOT / "experiments" / "phase9"
REPORT_DIR = PHASE_DIR / "reports"
FIGURE_DIR = PHASE_DIR / "figures"
LABELS = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]
BOOTSTRAP_SEED = 20261027
BOOTSTRAP_REPLICATES = 20000


def exact_row_key(row):
    """Use exact numeric feature values without decimal rounding."""
    return tuple(row.tolist())


def duplicate_analysis(train_x, train_y, val_x, val_y):
    train_map = {}
    train_label_sets = {}
    for index, row in enumerate(train_x):
        key = exact_row_key(row)
        train_map.setdefault(key, []).append(index)
        train_label_sets.setdefault(key, set()).add(int(train_y[index]))

    val_map = {}
    val_label_sets = {}
    for index, row in enumerate(val_x):
        key = exact_row_key(row)
        val_map.setdefault(key, []).append(index)
        val_label_sets.setdefault(key, set()).add(int(val_y[index]))

    matches = {}
    conflicts = []
    for val_index, row in enumerate(val_x):
        key = exact_row_key(row)
        if key in train_map:
            matches[val_index] = train_map[key]
            if any(label != int(val_y[val_index]) for label in train_label_sets[key]):
                conflicts.append(val_index)

    within_train_conflicts = sum(len(labels) > 1 for labels in train_label_sets.values())
    within_val_conflicts = sum(len(labels) > 1 for labels in val_label_sets.values())
    return {
        "train_unique_features": len(train_map),
        "validation_unique_features": len(val_map),
        "cross_split_validation_indices": sorted(matches),
        "cross_split_train_indices": sorted(
            {train_index for indices in matches.values() for train_index in indices}
        ),
        "validation_to_train_indices": {
            str(val_index): indices for val_index, indices in matches.items()
        },
        "cross_split_conflicting_label_validation_indices": conflicts,
        "within_train_conflicting_unique_features": within_train_conflicts,
        "within_validation_conflicting_unique_features": within_val_conflicts,
        "train_duplicate_rows": len(train_x) - len(train_map),
        "validation_duplicate_rows": len(val_x) - len(val_map),
    }


def reproduce_phase7b_split(train_y):
    rng = np.random.RandomState(42)
    train_indices = []
    dev_indices = []
    for class_id in np.unique(train_y):
        class_indices = np.where(train_y == class_id)[0]
        rng.shuffle(class_indices)
        dev_count = int(round(len(class_indices) * 0.20))
        dev_indices.extend(class_indices[:dev_count])
        train_indices.extend(class_indices[dev_count:])
    train_indices = np.asarray(train_indices, dtype=np.int32)
    dev_indices = np.asarray(dev_indices, dtype=np.int32)
    rng.shuffle(train_indices)
    rng.shuffle(dev_indices)
    return train_indices, dev_indices


def classification_metrics(y_true, probabilities, indices):
    indices = np.asarray(indices, dtype=np.int64)
    y = y_true[indices]
    probs = probabilities[indices]
    predictions = np.argmax(probs, axis=1)
    confusion = np.zeros((5, 5), dtype=np.int64)
    np.add.at(confusion, (y, predictions), 1)
    class_support = confusion.sum(axis=1)
    class_recall = np.divide(
        confusion.diagonal(),
        class_support,
        out=np.zeros(5, dtype=np.float64),
        where=class_support != 0,
    )
    class_precision = np.divide(
        confusion.diagonal(),
        confusion.sum(axis=0),
        out=np.zeros(5, dtype=np.float64),
        where=confusion.sum(axis=0) != 0,
    )
    class_f1 = np.divide(
        2 * class_precision * class_recall,
        class_precision + class_recall,
        out=np.zeros(5, dtype=np.float64),
        where=(class_precision + class_recall) != 0,
    )
    count = len(indices)
    correct = int(np.count_nonzero(predictions == y))
    return {
        "sample_count": count,
        "correct_count": correct,
        "error_count": count - correct,
        "accuracy": correct / count if count else None,
        "macro_f1": float(class_f1.mean()) if count else None,
        "macro_f1_definition": "Unweighted mean over all five action classes; classes absent from the subset contribute F1=0.",
        "cross_entropy_loss": float(
            -np.log(np.clip(probs[np.arange(count), y], 1e-7, 1.0)).mean()
        ) if count else None,
        "per_class_recall": {
            f"{class_id}: {LABELS[class_id]}": float(class_recall[class_id])
            for class_id in range(5)
        },
        "class_support": {
            f"{class_id}: {LABELS[class_id]}": int(class_support[class_id])
            for class_id in range(5)
        },
        "confusion_matrix": confusion.tolist(),
        "predictions": predictions,
        "correct_mask": predictions == y,
        "mean_confidence": float(np.max(probs, axis=1).mean()) if count else None,
    }


def to_json_metrics(metrics):
    return {key: value for key, value in metrics.items() if key not in {
        "predictions", "correct_mask"
    }}


def label_counts(labels, indices):
    counts = np.bincount(labels[np.asarray(indices, dtype=np.int64)], minlength=5)
    return {
        f"{class_id}: {LABELS[class_id]}": int(counts[class_id])
        for class_id in range(5)
    }


def feature_distribution(x, indices):
    subset = x[np.asarray(indices, dtype=np.int64)]
    presence_columns = np.arange(0, 25, 5)
    ego_vx = subset[:, 3]
    ego_vy = subset[:, 4]
    ego_speed = np.sqrt(ego_vx**2 + ego_vy**2)
    vehicle_count = np.sum(subset[:, presence_columns] > 0.5, axis=1)
    return {
        "sample_count": int(len(subset)),
        "ego_speed": {
            "mean": float(ego_speed.mean()),
            "std": float(ego_speed.std()),
            "min": float(ego_speed.min()),
            "median": float(np.median(ego_speed)),
            "p90": float(np.percentile(ego_speed, 90)),
            "max": float(ego_speed.max()),
        },
        "vehicle_presence_count": {
            str(count): int(np.count_nonzero(vehicle_count == count))
            for count in range(6)
        },
        "feature_statistics": {
            f"feature_{column:02d}": {
                "mean": float(subset[:, column].mean()),
                "std": float(subset[:, column].std()),
                "min": float(subset[:, column].min()),
                "max": float(subset[:, column].max()),
            }
            for column in range(subset.shape[1])
        },
    }


def bootstrap_accuracy_difference(baseline_correct, ensemble_correct):
    differences = ensemble_correct.astype(np.int8) - baseline_correct.astype(np.int8)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    batch_size = 250
    for start in range(0, BOOTSTRAP_REPLICATES, batch_size):
        end = min(start + batch_size, BOOTSTRAP_REPLICATES)
        sample_indices = rng.integers(
            0, len(differences), size=(end - start, len(differences))
        )
        boot_means[start:end] = differences[sample_indices].mean(axis=1)

    baseline_only = int(np.count_nonzero(baseline_correct & ~ensemble_correct))
    ensemble_only = int(np.count_nonzero(ensemble_correct & ~baseline_correct))
    discordant = baseline_only + ensemble_only
    p_value = (
        float(binomtest(ensemble_only, discordant, p=0.5, alternative="two-sided").pvalue)
        if discordant
        else 1.0
    )
    observed = float(differences.mean())
    return {
        "sample_count": int(len(differences)),
        "baseline_correct_ensemble_wrong": baseline_only,
        "baseline_wrong_ensemble_correct": ensemble_only,
        "ensemble_fixes_baseline_errors": ensemble_only,
        "ensemble_introduces_new_errors": baseline_only,
        "net_correct_predictions": ensemble_only - baseline_only,
        "accuracy_difference_ensemble_minus_baseline": observed,
        "accuracy_difference_percentage_points": observed * 100.0,
        "paired_bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "percentile_ci": [
                float(np.percentile(boot_means, 2.5)),
                float(np.percentile(boot_means, 97.5)),
            ],
            "percentile_ci_percentage_points": [
                float(np.percentile(boot_means, 2.5) * 100.0),
                float(np.percentile(boot_means, 97.5) * 100.0),
            ],
        },
        "mcnemar_exact_binomial": {
            "test": "two-sided exact binomial test on discordant pairs",
            "discordant_pairs": discordant,
            "p_value": p_value,
        },
        "bootstrap_distribution": boot_means,
    }


def save_figures(results, val_y, val_x, masks, evaluations, bootstrap):
    model_names = ["Baseline", "Ensemble"]
    colors = ["#3973ac", "#ef8a24"]
    groups = ["full_validation", "cross_split_duplicate", "non_duplicate"]
    group_names = ["Original validation", "Exact duplicates", "Non-duplicate"]

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(groups))
    width = 0.36
    for model_index, model in enumerate(("baseline", "ensemble")):
        values = [evaluations[model][group]["accuracy"] * 100 for group in groups]
        ax.bar(positions + (model_index - 0.5) * width, values, width,
               label=model_names[model_index], color=colors[model_index])
    ax.set_xticks(positions, group_names)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(80, 100)
    ax.set_title("Original vs Duplicate-aware Accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "phase9_original_vs_duplicate_aware_accuracy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    positions = np.arange(2)
    for model_index, model in enumerate(("baseline", "ensemble")):
        values = [
            evaluations[model]["cross_split_duplicate"]["accuracy"] * 100,
            evaluations[model]["non_duplicate"]["accuracy"] * 100,
        ]
        ax.bar(positions + (model_index - 0.5) * width, values, width,
               label=model_names[model_index], color=colors[model_index])
    ax.set_xticks(positions, ["Exact duplicate", "Non-duplicate"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(80, 100)
    ax.set_title("Duplicate vs Non-duplicate Accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "phase9_duplicate_vs_non_duplicate_accuracy.png", dpi=180)
    plt.close(fig)

    dup_indices = np.flatnonzero(masks["full_precision"])
    nondup_indices = np.flatnonzero(~masks["full_precision"])
    dup_counts = np.bincount(val_y[dup_indices], minlength=5)
    nondup_counts = np.bincount(val_y[nondup_indices], minlength=5)
    denom = np.maximum(dup_counts.sum(), 1), np.maximum(nondup_counts.sum(), 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(5)
    ax.bar(positions - width / 2, dup_counts / denom[0] * 100, width,
           label="Exact duplicate", color="#6a9f58")
    ax.bar(positions + width / 2, nondup_counts / denom[1] * 100, width,
           label="Non-duplicate", color="#8c75aa")
    ax.set_xticks(positions, [f"{i}: {name}" for i, name in enumerate(LABELS)], rotation=15)
    ax.set_ylabel("Share of subset (%)")
    ax.set_title("Validation Label Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "phase9_duplicate_label_distribution.png", dpi=180)
    plt.close(fig)

    common = evaluations
    fig, ax = plt.subplots(figsize=(7, 5))
    common_accuracies = [
        common["baseline"]["non_duplicate"]["accuracy"] * 100,
        common["ensemble"]["non_duplicate"]["accuracy"] * 100,
    ]
    bars = ax.bar(model_names, common_accuracies, color=colors, width=0.55)
    for bar, value in zip(bars, common_accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.08,
                f"{value:.3f}%", ha="center", va="bottom")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(min(common_accuracies) - 1, 100)
    ax.set_title("Model Comparison on Common Non-duplicate Subset")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "phase9_common_subset_model_comparison.png", dpi=180)
    plt.close(fig)

    distribution = bootstrap["bootstrap_distribution"]
    lower, upper = bootstrap["paired_bootstrap"]["percentile_ci_percentage_points"]
    observed = bootstrap["accuracy_difference_percentage_points"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(distribution * 100, bins=45, color="#6c8ebf", alpha=0.85)
    ax.axvline(observed, color="#c0392b", linewidth=2, label=f"Observed: {observed:+.3f} pp")
    ax.axvline(lower, color="#333333", linestyle="--", label=f"95% CI: [{lower:+.3f}, {upper:+.3f}]")
    ax.axvline(upper, color="#333333", linestyle="--")
    ax.set_xlabel("Ensemble − Baseline accuracy difference (percentage points)")
    ax.set_ylabel("Bootstrap replicates")
    ax.set_title("Paired Bootstrap Accuracy Difference")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "phase9_accuracy_difference_bootstrap_ci.png", dpi=180)
    plt.close(fig)


def write_report(results):
    duplicate_count = results["duplicate_detection"]["cross_split_full_precision"]["match_count"]
    val_count = results["dataset"]["validation_samples"]
    models = results["model_evaluation"]
    full_acc = {
        name: metrics["full_validation"]["accuracy"] * 100
        for name, metrics in models.items()
    }
    dup_acc = {
        name: metrics["cross_split_duplicate"]["accuracy"] * 100
        for name, metrics in models.items()
    }
    nondup_acc = {
        name: metrics["non_duplicate"]["accuracy"] * 100
        for name, metrics in models.items()
    }
    stats = results["paired_comparison_common_non_duplicate"]
    delta = stats["accuracy_difference_percentage_points"]
    ci = stats["paired_bootstrap"]["percentile_ci_percentage_points"]
    p_value = stats["mcnemar_exact_binomial"]["p_value"]
    uplift = {
        name: (dup_acc[name] - nondup_acc[name]) * duplicate_count / val_count
        for name in models
    }
    duplicate_distribution = results["distribution_analysis"]["exact_duplicate"]
    nonduplicate_distribution = results["distribution_analysis"]["non_duplicate"]
    duplicate_labels = ", ".join(
        f"{key}: {value}"
        for key, value in duplicate_distribution["label_counts"].items()
        if value
    )
    duplicate_speed = duplicate_distribution["features"]["ego_speed"]
    nonduplicate_speed = nonduplicate_distribution["features"]["ego_speed"]
    duplicate_one_vehicle = duplicate_distribution["features"][
        "vehicle_presence_count"
    ]["1"]
    if uplift["baseline"] > 0:
        baseline_effect = (
            f"Yes, descriptively: the duplicate subset has higher Baseline accuracy, "
            f"raising the full-set aggregate by about {uplift['baseline']:.3f} percentage "
            "points relative to replacing those rows with non-duplicate performance."
        )
    else:
        baseline_effect = (
            f"No positive aggregate uplift is supported by this descriptive comparison "
            f"(estimated contribution {uplift['baseline']:+.3f} percentage points)."
        )

    metric_lines = []
    for model_name, results_by_group in models.items():
        title = "Baseline" if model_name == "baseline" else "Ensemble"
        metric_lines.append(f"### {title}")
        metric_lines.append("")
        metric_lines.append(
            "| Evaluation subset | N | Accuracy | Errors | Macro F1 | Cross entropy |"
        )
        metric_lines.append("|---|---:|---:|---:|---:|---:|")
        for group, group_label in [
            ("full_validation", "Original validation"),
            ("cross_split_duplicate", "Cross-split exact duplicates"),
            ("non_duplicate", "Non-duplicate / common subset"),
            ("baseline_exposure", "Matches actual Baseline training inputs"),
            ("ensemble_exposure", "Matches actual Ensemble training inputs"),
        ]:
            item = results_by_group[group]
            metric_lines.append(
                f"| {group_label} | {item['sample_count']} | "
                f"{item['accuracy'] * 100:.3f}% | {item['error_count']} | "
                f"{item['macro_f1']:.4f} | {item['cross_entropy_loss']:.6f} |"
            )
        metric_lines.append("")
        recall = results_by_group["full_validation"]["per_class_recall"]
        metric_lines.append(
            "Original-validation per-class recall: "
            + "; ".join(f"{key} {value:.4f}" for key, value in recall.items())
        )
        metric_lines.append("")

    report = f"""# Phase 9 — Duplicate-Aware Validation & Generalization Audit

**Environment:** Docker `roundabout-midterm`; Python {results['environment']['python']}; TensorFlow {results['environment']['tensorflow']}; NumPy {results['environment']['numpy']}  
**Scope:** Existing saved models and datasets only. No retraining or input-file/model/submission changes.  
**Interpretation caution:** This repeatedly used validation set is not an independent test set; inferential results below are descriptive and conditional on this dataset.

## Executive Summary

- Original-precision exact cross-split match: **{duplicate_count}/{val_count}** validation rows; common non-duplicate subset: **{val_count - duplicate_count}** rows.
- Baseline full-validation accuracy: **{full_acc['baseline']:.3f}%**; duplicate subset: **{dup_acc['baseline']:.3f}%**; non-duplicate subset: **{nondup_acc['baseline']:.3f}%**.
- Ensemble full-validation accuracy: **{full_acc['ensemble']:.3f}%**; duplicate subset: **{dup_acc['ensemble']:.3f}%**; non-duplicate subset: **{nondup_acc['ensemble']:.3f}%**.
- On the common subset, Ensemble minus Baseline accuracy: **{delta:+.3f} percentage points**; paired bootstrap 95% CI **[{ci[0]:+.3f}, {ci[1]:+.3f}]**; exact McNemar p-value **{p_value:.6g}**.

## A. Exact Duplicate Reproduction

Exact matching uses the source feature values as loaded, without decimal rounding. Keys are tuples of all 25 feature coordinates at their native dtype. Float32 matching is reported separately after casting both datasets.

| Matching representation | Unique train rows | Unique validation rows | Validation rows matching train | Conflicting labels |
|---|---:|---:|---:|---:|
| Native {results['duplicate_detection']['native_dtype']} | {results['duplicate_detection']['cross_split_full_precision']['train_unique_features']} | {results['duplicate_detection']['cross_split_full_precision']['validation_unique_features']} | {results['duplicate_detection']['cross_split_full_precision']['match_count']} | {len(results['duplicate_detection']['cross_split_full_precision']['conflicting_validation_indices'])} |
| Float32 cast sensitivity | {results['duplicate_detection']['float32_sensitivity']['train_unique_features']} | {results['duplicate_detection']['float32_sensitivity']['validation_unique_features']} | {results['duplicate_detection']['float32_sensitivity']['match_count']} | {len(results['duplicate_detection']['float32_sensitivity']['conflicting_validation_indices'])} |

Within-set rows in excess of unique feature rows: train **{results['duplicate_detection']['cross_split_full_precision']['train_duplicate_rows']}**, validation **{results['duplicate_detection']['cross_split_full_precision']['validation_duplicate_rows']}**. Within-train conflicting feature labels: **{results['duplicate_detection']['cross_split_full_precision']['within_train_conflicting_unique_features']}**; within-validation: **{results['duplicate_detection']['cross_split_full_precision']['within_validation_conflicting_unique_features']}**; cross-split: **{len(results['duplicate_detection']['cross_split_full_precision']['conflicting_validation_indices'])}**.

Full list of matching validation row indices (zero-based): `{json.dumps(results['duplicate_detection']['cross_split_full_precision']['cross_split_validation_indices'])}`.

The JSON contains each matching validation index and all matching training indices. The Float32 sensitivity comparison is not substituted for the native-precision duplicate definition.

## B. Model-specific Training Exposure

The Baseline source training script loads all `train.npz` rows, casting features to Float32 before fitting. Phase 7B saves `split_indices.npz`; the audit independently reconstructed the script's `RandomState(42)` stratified split and compared exact arrays with that saved artifact. See the JSON's `split_reconstruction` for the verification result. Ensemble exposure is the saved 28,000 training indices only; the 7,000 dev rows are excluded.

| Validation exposure definition | Baseline | Ensemble |
|---|---:|---:|
| Matches full train.npz at native precision | {results['exposure']['full_train_native_precision_match_count']} | {results['exposure']['full_train_native_precision_match_count']} |
| Matches model training inputs at actual Float32 input precision | {results['exposure']['baseline_float32_training_match_count']} | {results['exposure']['ensemble_float32_training_match_count']} |
| Matches actual training inputs but not native full-precision match | {results['exposure']['baseline_float32_only_match_count']} | {results['exposure']['ensemble_float32_only_match_count']} |

For each model, duplicate-aware metrics also include its specific actual-training-input exposure subset. These are exposure flags, not proof that a model memorized a row.

## C. Duplicate-aware Model Evaluation

{chr(10).join(metric_lines)}
Weighted accuracy reconstruction on the original validation set uses the exact duplicate and non-duplicate sample counts and their correct counts. Both reconstructions and the stored-original results are included in the JSON; discrepancies are asserted to be within floating point tolerance.

Macro F1 is the unweighted mean over the five fixed action classes; if a subset has no true or predicted examples for a class, that class contributes F1=0. Per-class recall and support are reported alongside it.

## D. Paired Statistical Comparison on the Common Subset

- Baseline errors corrected by Ensemble: **{stats['ensemble_fixes_baseline_errors']}**.
- Baseline correct predictions made wrong by Ensemble: **{stats['ensemble_introduces_new_errors']}**.
- Net change: **{stats['net_correct_predictions']}** correct predictions; **{delta:+.3f} pp**.
- Paired bootstrap percentile 95% CI: **[{ci[0]:+.3f}, {ci[1]:+.3f}] pp** using {stats['paired_bootstrap']['replicates']:,} paired resamples, seed {stats['paired_bootstrap']['seed']}.
- Exact two-sided McNemar/binomial p-value: **{p_value:.6g}** over {stats['mcnemar_exact_binomial']['discordant_pairs']} discordant pairs.

This validation set has been repeatedly used for model selection. These paired tests do not turn it into an independent test set and should not be read as unbiased evidence of future generalization.

## E. Dataset Distribution

For exact duplicates, labels are **{duplicate_labels}**. All **{duplicate_one_vehicle}/{duplicate_count}** rows have exactly one present vehicle (the first/ego slot under the dataset layout). Ego speed from the first vehicle's vx/vy has mean **{duplicate_speed['mean']:.3f}**, median **{duplicate_speed['median']:.3f}**, and minimum **{duplicate_speed['min']:.3f}** in the stored feature scale; the non-duplicate median is **{nonduplicate_speed['median']:.3f}** (mean **{nonduplicate_speed['mean']:.3f}**). Thus the exact-duplicate subset is concentrated in higher-speed, ego-only observations, not in stopped/near-zero-speed samples; this is descriptive and does not establish a causal mechanism or a specific speed cap. The JSON reports per-class counts; 25 feature-wise mean, standard deviation, minimum, and maximum; vehicle-presence counts; and each model's confidence and prediction-error rates for duplicate and non-duplicate rows. These are descriptive associations, not causal findings. Vehicle presence is counted from the five presence slots using a >0.5 threshold; ego speed is the Euclidean magnitude of the first vehicle's vx/vy.

## Final Interpretation

### Confirmed Findings

1. Reproduced **{duplicate_count}** original-precision exact cross-split duplicate validation rows, with **{len(results['duplicate_detection']['cross_split_full_precision']['conflicting_validation_indices'])}** conflicting-label matches.
2. Both models' original, duplicate-only, non-duplicate, and actual-training-exposure metric subsets were evaluated from saved-model probabilities.
3. Baseline duplicate-subset performance implies: **{baseline_effect}**
4. The Ensemble's corresponding descriptive aggregate contribution is **{uplift['ensemble']:+.3f} pp** (vs. **{uplift['baseline']:+.3f} pp** for Baseline); both models score 100% on the 263 duplicates. The overlap impact on the aggregate is similar in size, not evidence of equivalent memorization.
5. On the shared **{val_count - duplicate_count}**-sample non-duplicate subset, Ensemble {'outperforms' if delta > 0 else 'does not outperform' if delta < 0 else 'matches'} Baseline by **{delta:+.3f} pp**. Its 95% CI includes zero and McNemar p={p_value:.4g}; this audit does not provide sufficient evidence by itself to change the submitted model.
6. All **{models['baseline']['full_validation']['error_count']}** Baseline and **{models['ensemble']['full_validation']['error_count']}** Ensemble validation errors are on non-duplicate rows. Therefore exact duplicates contribute modestly to the aggregate scores but do not account for the remaining accuracy bottleneck.

### Supported Hypotheses

- If duplicate-subset accuracy exceeds non-duplicate accuracy, exact overlap descriptively contributes to the aggregate score of that model. The same comparison is reported independently for Baseline and Ensemble; similar counts of overlapping rows do not imply equal performance impact.
- The aggregate score uplift is only a counterfactual reweighting description, not a causal estimate of training leakage or memorization.
- Duplicate overlap alone is not sufficient to explain the remaining non-duplicate errors; the measured common-subset error rate quantifies errors outside these exact overlaps.

### Unresolved Limitations

- Validation was used for repeated model selection, so this is not a fresh test-set generalization claim.
- Float32 training-input matching can in principle add matches not present at source precision, but it changed neither unique-row counts nor cross-split matches for these files (263 in both representations).
- Performance differences and distribution summaries are observational; they do not establish that overlap caused a model's accuracy.
- The Ensemble has a small observed common-subset accuracy advantage, but the paired interval crosses zero, the McNemar test is not statistically significant at conventional thresholds, and its macro F1 is lower; this audit alone is not sufficient reason to replace the submission model.

## Artifacts

- Results: `phase9_results.json`
- Figures: `../figures/phase9_*.png`
- Reproduction script: `../scripts/duplicate_validation_audit.py`
"""
    (REPORT_DIR / "phase9_duplicate_validation.md").write_text(report, encoding="utf-8")


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    train_archive = np.load(ROOT / "dataset" / "train.npz")
    val_archive = np.load(ROOT / "dataset" / "validation.npz")
    train_x_native = train_archive["data"]
    val_x_native = val_archive["data"]
    train_y = train_archive["label"].astype(np.int32).reshape(-1)
    val_y = val_archive["label"].astype(np.int32).reshape(-1)
    if train_x_native.ndim != 2 or val_x_native.ndim != 2:
        raise ValueError("Expected 2D feature arrays.")
    if train_x_native.shape[1] != 25 or val_x_native.shape[1] != 25:
        raise ValueError("Expected exactly 25 input features.")
    if len(train_y) != len(train_x_native) or len(val_y) != len(val_x_native):
        raise ValueError("Feature and label sample counts do not match.")

    raw_duplicates = duplicate_analysis(train_x_native, train_y, val_x_native, val_y)
    float32_duplicates = duplicate_analysis(
        train_x_native.astype(np.float32),
        train_y,
        val_x_native.astype(np.float32),
        val_y,
    )
    full_duplicate_mask = np.zeros(len(val_y), dtype=bool)
    full_duplicate_mask[raw_duplicates["cross_split_validation_indices"]] = True
    float32_full_mask = np.zeros(len(val_y), dtype=bool)
    float32_full_mask[float32_duplicates["cross_split_validation_indices"]] = True

    split_file = np.load(PHASE_DIR.parent / "phase7b" / "split_indices.npz")
    saved_train_indices = split_file["train_indices"]
    saved_dev_indices = split_file["dev_indices"]
    rebuilt_train_indices, rebuilt_dev_indices = reproduce_phase7b_split(train_y)
    split_matches = (
        np.array_equal(saved_train_indices, rebuilt_train_indices)
        and np.array_equal(saved_dev_indices, rebuilt_dev_indices)
    )
    if not split_matches:
        raise AssertionError(
            "Saved Phase 7B indices differ from the documented deterministic split."
        )
    if len(saved_train_indices) != 28000 or len(saved_dev_indices) != 7000:
        raise AssertionError("Unexpected Phase 7B train/dev split sizes.")

    train_x32 = train_x_native.astype(np.float32)
    val_x32 = val_x_native.astype(np.float32)
    baseline_exposure_analysis = duplicate_analysis(
        train_x32, train_y, val_x32, val_y
    )
    ensemble_exposure_analysis = duplicate_analysis(
        train_x32[saved_train_indices],
        train_y[saved_train_indices],
        val_x32,
        val_y,
    )
    baseline_exposure_mask = np.zeros(len(val_y), dtype=bool)
    baseline_exposure_mask[
        baseline_exposure_analysis["cross_split_validation_indices"]
    ] = True
    ensemble_exposure_mask = np.zeros(len(val_y), dtype=bool)
    ensemble_exposure_mask[
        ensemble_exposure_analysis["cross_split_validation_indices"]
    ] = True

    model_paths = {
        "baseline": ROOT / "models" / "baseline_standard.h5",
        "ensemble": PHASE_DIR.parent / "phase7b" / "models" / "ensemble_unified.h5",
    }
    predictions = {}
    for model_name, model_path in model_paths.items():
        if not model_path.is_file():
            raise FileNotFoundError(f"Required model does not exist: {model_path}")
        model = tf.keras.models.load_model(model_path, compile=False)
        probabilities = np.asarray(model.predict(val_x32, batch_size=512, verbose=0))
        if probabilities.shape != (len(val_y), 5):
            raise ValueError(
                f"{model_name} produced unexpected probability shape {probabilities.shape}."
            )
        if not np.all(np.isfinite(probabilities)):
            raise ValueError(f"{model_name} produced non-finite probability values.")
        if np.any(probabilities < 0) or not np.allclose(
            probabilities.sum(axis=1), 1.0, atol=1e-4
        ):
            raise ValueError(f"{model_name} output is not a valid probability matrix.")
        predictions[model_name] = probabilities
        del model

    masks = {
        "full_precision": full_duplicate_mask,
        "float32_full_train": float32_full_mask,
        "baseline_exposure": baseline_exposure_mask,
        "ensemble_exposure": ensemble_exposure_mask,
    }
    groups = {
        "full_validation": np.ones(len(val_y), dtype=bool),
        "cross_split_duplicate": full_duplicate_mask,
        "non_duplicate": ~full_duplicate_mask,
        "baseline_exposure": baseline_exposure_mask,
        "ensemble_exposure": ensemble_exposure_mask,
    }
    evaluations = {}
    for model_name, probabilities in predictions.items():
        evaluations[model_name] = {
            group: classification_metrics(val_y, probabilities, np.flatnonzero(mask))
            for group, mask in groups.items()
        }
    expected_original_accuracy = {"baseline": 0.9464, "ensemble": 0.9496}
    for model_name, expected_accuracy in expected_original_accuracy.items():
        actual_accuracy = evaluations[model_name]["full_validation"]["accuracy"]
        if not np.isclose(actual_accuracy, expected_accuracy, atol=1e-12):
            raise AssertionError(
                f"{model_name} original validation accuracy {actual_accuracy:.8f} "
                f"does not reproduce expected {expected_accuracy:.8f}."
            )

    duplicate_count = int(full_duplicate_mask.sum())
    nonduplicate_count = int((~full_duplicate_mask).sum())
    if duplicate_count + nonduplicate_count != len(val_y):
        raise AssertionError("Duplicate and non-duplicate masks do not partition validation.")

    weighted_reconstruction = {}
    for model_name in evaluations:
        full = evaluations[model_name]["full_validation"]
        dup = evaluations[model_name]["cross_split_duplicate"]
        nondup = evaluations[model_name]["non_duplicate"]
        reconstructed_correct = dup["correct_count"] + nondup["correct_count"]
        reconstructed_accuracy = reconstructed_correct / len(val_y)
        if reconstructed_correct != full["correct_count"] or not np.isclose(
            reconstructed_accuracy, full["accuracy"], atol=1e-15
        ):
            raise AssertionError(f"{model_name} weighted accuracy does not reconstruct.")
        weighted_reconstruction[model_name] = {
            "duplicate_correct": dup["correct_count"],
            "non_duplicate_correct": nondup["correct_count"],
            "total_correct_reconstructed": reconstructed_correct,
            "total_samples_reconstructed": duplicate_count + nonduplicate_count,
            "weighted_accuracy_reconstructed": reconstructed_accuracy,
            "original_accuracy": full["accuracy"],
            "exact_match": True,
        }

    baseline_common = evaluations["baseline"]["non_duplicate"]["correct_mask"]
    ensemble_common = evaluations["ensemble"]["non_duplicate"]["correct_mask"]
    bootstrap = bootstrap_accuracy_difference(baseline_common, ensemble_common)

    distribution = {}
    for subset_name, indices in {
        "exact_duplicate": np.flatnonzero(full_duplicate_mask),
        "non_duplicate": np.flatnonzero(~full_duplicate_mask),
    }.items():
        subset_distribution = {
            "label_counts": label_counts(val_y, indices),
            "features": feature_distribution(val_x_native, indices),
            "models": {},
        }
        for model_name in predictions:
            subset_metrics = classification_metrics(
                val_y, predictions[model_name], indices
            )
            confidences = np.max(predictions[model_name][indices], axis=1)
            subset_distribution["models"][model_name] = {
                "mean_confidence": float(confidences.mean()),
                "confidence_percentiles": {
                    "p10": float(np.percentile(confidences, 10)),
                    "p50": float(np.percentile(confidences, 50)),
                    "p90": float(np.percentile(confidences, 90)),
                },
                "error_count": subset_metrics["error_count"],
                "error_rate": 1.0 - subset_metrics["accuracy"],
                "accuracy": subset_metrics["accuracy"],
            }
        distribution[subset_name] = subset_distribution

    split_results = {
        "saved_train_indices_count": int(len(saved_train_indices)),
        "saved_dev_indices_count": int(len(saved_dev_indices)),
        "reconstructed_train_indices_exact_match": bool(
            np.array_equal(saved_train_indices, rebuilt_train_indices)
        ),
        "reconstructed_dev_indices_exact_match": bool(
            np.array_equal(saved_dev_indices, rebuilt_dev_indices)
        ),
        "split_matches_saved_indices": bool(split_matches),
        "training_indices": saved_train_indices.tolist(),
        "dev_indices": saved_dev_indices.tolist(),
        "source": "experiments/phase7b/split_indices.npz, independently checked against train_ensemble.py logic",
    }

    results = {
        "phase": 9,
        "environment": {
            "container": "roundabout-midterm",
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "execution_assertion": "This script is intended to be run inside the project Docker container.",
        },
        "dataset": {
            "train_samples": int(len(train_y)),
            "validation_samples": int(len(val_y)),
            "feature_count": int(train_x_native.shape[1]),
            "native_feature_dtype": str(train_x_native.dtype),
            "native_validation_feature_dtype": str(val_x_native.dtype),
            "train_label_counts": label_counts(train_y, np.arange(len(train_y))),
            "validation_label_counts": label_counts(val_y, np.arange(len(val_y))),
        },
        "duplicate_detection": {
            "method": "Exact numeric equality across all 25 coordinates; no rounding; Python tuple keys at stated dtype.",
            "native_dtype": str(train_x_native.dtype),
            "cross_split_full_precision": {
                **raw_duplicates,
                "match_count": len(raw_duplicates["cross_split_validation_indices"]),
                "conflicting_validation_indices": raw_duplicates[
                    "cross_split_conflicting_label_validation_indices"
                ],
                "validation_match_rate": len(raw_duplicates[
                    "cross_split_validation_indices"
                ]) / len(val_y),
            },
            "float32_sensitivity": {
                **float32_duplicates,
                "match_count": len(float32_duplicates["cross_split_validation_indices"]),
                "conflicting_validation_indices": float32_duplicates[
                    "cross_split_conflicting_label_validation_indices"
                ],
                "match_indices_added_vs_native": sorted(
                    set(float32_duplicates["cross_split_validation_indices"])
                    - set(raw_duplicates["cross_split_validation_indices"])
                ),
                "native_matches_not_found_after_cast": sorted(
                    set(raw_duplicates["cross_split_validation_indices"])
                    - set(float32_duplicates["cross_split_validation_indices"])
                ),
            },
        },
        "split_reconstruction": split_results,
        "exposure": {
            "definitions": {
                "full_precision_match": "Native precision exact feature match against all train.npz rows.",
                "baseline_training_match": "Float32 exact feature match against all train.npz rows, matching the Baseline training script's input cast.",
                "ensemble_training_match": "Float32 exact feature match against the 28,000 saved Phase 7B training indices; excludes the internal dev set.",
            },
            "full_train_native_precision_match_count": duplicate_count,
            "baseline_float32_training_match_count": int(baseline_exposure_mask.sum()),
            "baseline_float32_only_match_count": int(
                np.count_nonzero(baseline_exposure_mask & ~full_duplicate_mask)
            ),
            "baseline_native_full_matches_absent_after_float32": int(
                np.count_nonzero(full_duplicate_mask & ~baseline_exposure_mask)
            ),
            "ensemble_float32_training_match_count": int(ensemble_exposure_mask.sum()),
            "ensemble_float32_only_match_count": int(
                np.count_nonzero(ensemble_exposure_mask & ~full_duplicate_mask)
            ),
            "ensemble_native_full_matches_absent_after_float32": int(
                np.count_nonzero(full_duplicate_mask & ~ensemble_exposure_mask)
            ),
            "baseline_exposure_validation_indices": np.flatnonzero(
                baseline_exposure_mask
            ).tolist(),
            "ensemble_exposure_validation_indices": np.flatnonzero(
                ensemble_exposure_mask
            ).tolist(),
            "ensemble_cross_split_match_train_indices": ensemble_exposure_analysis[
                "cross_split_train_indices"
            ],
            "ensemble_rows_matching_only_internal_dev_not_ensemble_train": sorted(
                set(baseline_exposure_analysis["cross_split_validation_indices"])
                - set(ensemble_exposure_analysis["cross_split_validation_indices"])
            ),
        },
        "model_evaluation": {
            model_name: {
                group: to_json_metrics(metrics)
                for group, metrics in group_results.items()
            }
            for model_name, group_results in evaluations.items()
        },
        "metric_conventions": {
            "macro_f1": "Unweighted mean over all five action classes; absent classes in a subset contribute F1=0.",
            "cross_entropy_loss": "Mean negative log probability of the true class; probabilities clipped at 1e-7.",
            "accuracy": "Correct argmax predictions divided by subset sample count.",
        },
        "weighted_accuracy_reconstruction": weighted_reconstruction,
        "paired_comparison_common_non_duplicate": {
            key: value for key, value in bootstrap.items()
            if key != "bootstrap_distribution"
        },
        "distribution_analysis": distribution,
        "model_files": {
            name: str(path.relative_to(ROOT)) for name, path in model_paths.items()
        },
        "bootstrap_configuration": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "resampling_unit": "validation rows, paired across both model predictions",
        },
    }

    save_figures(results, val_y, val_x_native, masks, evaluations, bootstrap)
    (REPORT_DIR / "phase9_results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_report(results)
    print(f"Native precision unique train rows: {raw_duplicates['train_unique_features']}")
    print(f"Native precision unique validation rows: {raw_duplicates['validation_unique_features']}")
    print(f"Native precision cross-split matches: {duplicate_count}")
    print(f"Float32 cross-split matches: {len(float32_duplicates['cross_split_validation_indices'])}")
    print(f"Phase 7B saved split reproduced exactly: {split_matches}")
    print(
        "Baseline / Ensemble duplicate exposure matches:",
        int(baseline_exposure_mask.sum()),
        int(ensemble_exposure_mask.sum()),
    )
    print(
        "Common subset n / accuracy difference (pp) / McNemar p:",
        nonduplicate_count,
        f"{bootstrap['accuracy_difference_percentage_points']:+.6f}",
        f"{bootstrap['mcnemar_exact_binomial']['p_value']:.8g}",
    )
    print("Reports:", REPORT_DIR)
    print("Figures:", FIGURE_DIR)


if __name__ == "__main__":
    main()
