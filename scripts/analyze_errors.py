#!/usr/bin/env python3
"""
scripts/analyze_errors.py
Comprehensive Error Analysis and Environment Audit for Phase 3B:
- Task A: Audit Environment Evaluation (bootstrap paired CI, invalid action breakdown)
- Task B: Validation Error Analysis (confidence, misclassification pairs, actions 0/2 errors, actions 3/4 confusion, feature distribution audit)
"""

import json
import os
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

CLASS_NAMES = [
    "0: LANE_LEFT",
    "1: IDLE",
    "2: LANE_RIGHT",
    "3: FASTER",
    "4: SLOWER"
]
SHORT_NAMES = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]


def task_a_audit(phase3a_json_path):
    print("=" * 70)
    print(" TASK A: AUDIT ENVIRONMENT EVALUATION & BOOTSTRAP ANALYSIS")
    print("=" * 70)

    with open(phase3a_json_path, "r") as f:
        data = json.load(f)

    eps_std = data["episodes_standard"]
    eps_wgt = data["episodes_weighted"]

    rewards_std = np.array([e["reward"] for e in eps_std])
    rewards_wgt = np.array([e["reward"] for e in eps_wgt])

    crashes_std = np.array([int(e["crashed"]) for e in eps_std])
    crashes_wgt = np.array([int(e["crashed"]) for e in eps_wgt])

    # Paired differences
    diff_rewards = rewards_wgt - rewards_std
    diff_crashes = crashes_wgt - crashes_std

    # Bootstrap 10,000 resamples
    np.random.seed(42)
    n_boot = 10000
    n_samples = len(diff_rewards)

    boot_diff_rewards = np.array([
        np.mean(np.random.choice(diff_rewards, size=n_samples, replace=True))
        for _ in range(n_boot)
    ])
    r_ci_lower = float(np.percentile(boot_diff_rewards, 2.5))
    r_ci_upper = float(np.percentile(boot_diff_rewards, 97.5))

    boot_diff_crashes = np.array([
        np.mean(np.random.choice(diff_crashes, size=n_samples, replace=True)) * 100.0
        for _ in range(n_boot)
    ])
    c_ci_lower = float(np.percentile(boot_diff_crashes, 2.5))
    c_ci_upper = float(np.percentile(boot_diff_crashes, 97.5))

    print(f"Paired 100 Episodes Comparison (Weighted - Standard):")
    print(f"  - Mean Reward Difference:       {np.mean(diff_rewards):+.4f}")
    print(f"    * 95% Bootstrap CI:           [{r_ci_lower:+.4f}, {r_ci_upper:+.4f}]")
    print(f"    * Contains 0 (Not Sig at 5%): {r_ci_lower <= 0 <= r_ci_upper}")
    print(f"  - Collision Rate Difference:    {np.mean(diff_crashes)*100:+.2f}% ({np.sum(crashes_wgt)} vs {np.sum(crashes_std)})")
    print(f"    * 95% Bootstrap CI:           [{c_ci_lower:+.2f}%, {c_ci_upper:+.2f}%]")
    print(f"    * Contains 0 (Not Sig at 5%): {c_ci_lower <= 0 <= c_ci_upper}")

    # Action Infeasibility Breakdown (Pre-computed in environment audit)
    # Standard: Total 1059 steps: 7 Lane Change (0.66%), 457 Speed Action (43.15%)
    # Weighted: Total 1066 steps: 14 Lane Change (1.31%), 473 Speed Action (44.37%)
    infeasible_breakdown = {
        "standard": {
            "total_steps": 1059,
            "lane_change_infeasible": 7,
            "lane_change_infeasible_pct": 0.66,
            "speed_infeasible": 457,
            "speed_infeasible_pct": 43.15,
            "total_infeasible": 464,
            "total_infeasible_pct": 43.81,
        },
        "weighted": {
            "total_steps": 1066,
            "lane_change_infeasible": 14,
            "lane_change_infeasible_pct": 1.31,
            "speed_infeasible": 473,
            "speed_infeasible_pct": 44.37,
            "total_infeasible": 487,
            "total_infeasible_pct": 45.68,
        }
    }

    print("\nInfeasible Action Detailed Breakdown:")
    for m in ["standard", "weighted"]:
        b = infeasible_breakdown[m]
        print(f"  [{m.upper()}] Total steps: {b['total_steps']}")
        print(f"    - Infeasible Lane Changes (Actions 0, 2): {b['lane_change_infeasible']:3d} ({b['lane_change_infeasible_pct']:.2f}%)")
        print(f"    - Infeasible Speed Changes (Actions 3, 4): {b['speed_infeasible']:3d} ({b['speed_infeasible_pct']:.2f}%)")
        print(f"    - Total Infeasible:                        {b['total_infeasible']:3d} ({b['total_infeasible_pct']:.2f}%)")

    return {
        "reward_diff_mean": float(np.mean(diff_rewards)),
        "reward_diff_ci_95": [r_ci_lower, r_ci_upper],
        "reward_diff_sig": not (r_ci_lower <= 0 <= r_ci_upper),
        "collision_diff_pct": float(np.mean(diff_crashes) * 100),
        "collision_diff_ci_95": [c_ci_lower, c_ci_upper],
        "collision_diff_sig": not (c_ci_lower <= 0 <= c_ci_upper),
        "infeasible_breakdown": infeasible_breakdown,
    }


def task_b_validation_error_analysis(val_path, model_path):
    print("\n" + "=" * 70)
    print(" TASK B: VALIDATION ERROR & DATASET FEATURE ANALYSIS")
    print("=" * 70)

    val_data = np.load(val_path)
    x_val = val_data["data"].astype(np.float32)
    y_val = val_data["label"].astype(np.int32).flatten()
    n_samples = len(y_val)

    model = tf.keras.models.load_model(model_path)
    probs = model(x_val, training=False).numpy()
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    is_correct = (preds == y_val)
    n_errors = int((~is_correct).sum())
    val_acc = float(is_correct.mean() * 100)

    print(f"Validation Samples: {n_samples}, Correct: {is_correct.sum()}, Errors: {n_errors} (Acc: {val_acc:.2f}%)")

    # 1. Prediction Confidence & Entropy Analysis
    eps = 1e-12
    entropies = -np.sum(probs * np.log(probs + eps), axis=1)

    conf_correct = confidences[is_correct]
    conf_error = confidences[~is_correct]
    ent_correct = entropies[is_correct]
    ent_error = entropies[~is_correct]

    print("\n1. Prediction Confidence & Uncertainty:")
    print(f"  - Correct Samples:   Mean Conf = {conf_correct.mean():.4f} +/- {conf_correct.std():.4f}, Mean Entropy = {ent_correct.mean():.4f}")
    print(f"  - Error Samples:     Mean Conf = {conf_error.mean():.4f} +/- {conf_error.std():.4f}, Mean Entropy = {ent_error.mean():.4f}")
    print(f"  - Low Conf (< 0.50): {(confidences < 0.5).sum()} samples ({(confidences < 0.5).mean()*100:.2f}%)")
    print(f"  - Mid Conf (0.5-0.8):{((confidences >= 0.5) & (confidences < 0.8)).sum()} samples ({((confidences >= 0.5) & (confidences < 0.8)).mean()*100:.2f}%)")
    print(f"  - High Conf (>= 0.8):{(confidences >= 0.8).sum()} samples ({(confidences >= 0.8).mean()*100:.2f}%)")

    # 2. Confusion Pairs Ranking
    cm = np.zeros((5, 5), dtype=int)
    for t, p in zip(y_val, preds):
        cm[t, p] += 1

    pairs = []
    for t in range(5):
        for p in range(5):
            if t != p and cm[t, p] > 0:
                pairs.append({
                    "true_class": t,
                    "pred_class": p,
                    "pair_name": f"{SHORT_NAMES[t]} -> {SHORT_NAMES[p]}",
                    "count": int(cm[t, p]),
                    "pct_of_true_class": float(cm[t, p] / cm[t].sum() * 100),
                    "pct_of_all_errors": float(cm[t, p] / n_errors * 100),
                })
    pairs.sort(key=lambda x: x["count"], reverse=True)

    print("\n2. Top Misclassification Pairs:")
    for idx, pair in enumerate(pairs[:8]):
        print(f"  #{idx+1}: {pair['pair_name']:<22} | Count: {pair['count']:3d} | {pair['pct_of_all_errors']:5.1f}% of errors | {pair['pct_of_true_class']:5.1f}% of True class")

    # 3. Actions 3 & 4 Mutual Misclassification
    f3_to_4 = cm[3, 4]  # FASTER -> SLOWER
    f4_to_3 = cm[4, 3]  # SLOWER -> FASTER
    print("\n3. Mutual Confusion Between FASTER (3) and SLOWER (4):")
    print(f"  - True FASTER misclassified as SLOWER: {f3_to_4} / {cm[3].sum()} ({f3_to_4/cm[3].sum()*100:.2f}%)")
    print(f"  - True SLOWER misclassified as FASTER: {f4_to_3} / {cm[4].sum()} ({f4_to_3/cm[4].sum()*100:.2f}%)")
    print(f"  - Total Mutual F/S Confusion:          {f3_to_4 + f4_to_3} / {n_errors} ({((f3_to_4+f4_to_3)/n_errors)*100:.2f}% of all errors)")

    # 4. Actions 0 & 2 Error Analysis
    err_0_idx = np.where((y_val == 0) & (preds != 0))[0]
    err_2_idx = np.where((y_val == 2) & (preds != 2))[0]
    print("\n4. Error Analysis on Minor Turn Actions (0 & 2):")
    print(f"  - Class 0 (LANE_LEFT) Errors:  {len(err_0_idx)} / {cm[0].sum()} (Recall = {(1 - len(err_0_idx)/cm[0].sum())*100:.2f}%)")
    for p in range(5):
        if p != 0 and cm[0, p] > 0:
            print(f"    * Misclassified as {SHORT_NAMES[p]}: {cm[0, p]} ({cm[0, p]/cm[0].sum()*100:.1f}%)")

    print(f"  - Class 2 (LANE_RIGHT) Errors: {len(err_2_idx)} / {cm[2].sum()} (Recall = {(1 - len(err_2_idx)/cm[2].sum())*100:.2f}%)")
    for p in range(5):
        if p != 2 and cm[2, p] > 0:
            print(f"    * Misclassified as {SHORT_NAMES[p]}: {cm[2, p]} ({cm[2, p]/cm[2].sum()*100:.1f}%)")

    # Feature inspection on Class 0 errors
    if len(err_0_idx) > 0:
        err_0_feats = x_val[err_0_idx]
        corr_0_idx = np.where((y_val == 0) & (preds == 0))[0]
        corr_0_feats = x_val[corr_0_idx]
        print(f"    Feature Contrast for Class 0 (Error vs Correct):")
        print(f"      * Ego speed vx: Error mean = {err_0_feats[:, 3].mean():.3f} vs Correct mean = {corr_0_feats[:, 3].mean():.3f}")
        print(f"      * Ego speed vy: Error mean = {err_0_feats[:, 4].mean():.3f} vs Correct mean = {corr_0_feats[:, 4].mean():.3f}")
        print(f"      * Mean Confidence on False predictions: {confidences[err_0_idx].mean():.3f}")

    # 5. Dataset Feature Distribution Audit
    print("\n5. Dataset Feature Distribution Audit (25 Features):")
    feat_names = []
    vehicle_names = ["Ego", "Veh1", "Veh2", "Veh3", "Veh4"]
    attr_names = ["presence", "x", "y", "vx", "vy"]
    for v in vehicle_names:
        for a in attr_names:
            feat_names.append(f"{v}_{a}")

    train_data = np.load("/app/dataset/train.npz")
    x_train = train_data["data"]

    feature_audit = []
    print(f"  {'Index':<5} | {'Feature Name':<15} | {'Min':<8} | {'Max':<8} | {'Mean':<8} | {'Std':<8} | {'Zero Var / Dead?'}")
    print("  " + "-" * 75)
    for i in range(25):
        f_min = float(x_train[:, i].min())
        f_max = float(x_train[:, i].max())
        f_mean = float(x_train[:, i].mean())
        f_std = float(x_train[:, i].std())
        is_dead = (f_std == 0.0)
        dead_str = "YES (DEAD)" if is_dead else "No"
        feature_audit.append({
            "index": i,
            "name": feat_names[i],
            "min": f_min,
            "max": f_max,
            "mean": f_mean,
            "std": f_std,
            "is_dead": is_dead,
        })
        if i in [0, 1, 2, 3, 4, 5, 10, 15, 20]:
            print(f"  {i:<5} | {feat_names[i]:<15} | {f_min:<8.2f} | {f_max:<8.2f} | {f_mean:<8.2f} | {f_std:<8.2f} | {dead_str}")

    # Vehicle Presence Rates
    presence_indices = [0, 5, 10, 15, 20]
    print("\n  Vehicle Presence Rates in Training Dataset:")
    for v_idx, p_idx in enumerate(presence_indices):
        rate = float(x_train[:, p_idx].mean() * 100)
        print(f"    - {vehicle_names[v_idx]} (Feature {p_idx:2d}): {rate:5.1f}% present")

    return {
        "val_accuracy": val_acc,
        "n_errors": n_errors,
        "confidence": {
            "correct_mean": float(conf_correct.mean()),
            "correct_std": float(conf_correct.std()),
            "error_mean": float(conf_error.mean()),
            "error_std": float(conf_error.std()),
            "low_conf_pct": float((confidences < 0.5).mean() * 100),
            "high_conf_pct": float((confidences >= 0.8).mean() * 100),
        },
        "top_misclassifications": pairs[:10],
        "mutual_faster_slower": {
            "faster_to_slower": int(f3_to_4),
            "slower_to_faster": int(f4_to_3),
            "total_mutual": int(f3_to_4 + f4_to_3),
            "pct_of_all_errors": float((f3_to_4 + f4_to_3) / n_errors * 100),
        },
        "feature_audit": feature_audit,
    }


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    phase3a_json = os.path.join(root_dir, "reports", "phase3a_results.json")
    val_path = os.path.join(root_dir, "dataset", "validation.npz")
    model_path = os.path.join(root_dir, "models", "baseline_standard.h5")

    res_a = task_a_audit(phase3a_json)
    res_b = task_b_validation_error_analysis(val_path, model_path)

    output = {
        "task_a_audit": res_a,
        "task_b_error_analysis": res_b,
    }

    out_path = os.path.join(root_dir, "reports", "phase3b_audit_error_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved analysis results to: {out_path}")
    print("=" * 70)
    print(" AUDIT AND ERROR ANALYSIS FINISHED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()

