#!/usr/bin/env python3
"""
scripts/evaluate_optimization.py
Unified evaluation for Phase 3B Task D (Offline Evaluation) and Task E (Environment Evaluation).
- Evaluates 3 models offline on validation.npz:
    1. baseline_standard.h5 (Control)
    2. optimized_wider.h5 (Exp B)
    3. optimized_low_lr.h5 (Exp C)
- Produces full comparison metrics & figures
- Selects the best optimized model and runs 100-episode paired evaluation in roundabout-v0 (Task E)
- Performs paired bootstrap analysis for Reward & Collision differences
- Generates all required figures in reports/figures/
- Saves complete results to reports/phase3b_results.json
"""

import json
import os
import sys
import time
import numpy as np

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gymnasium as gym
import highway_env
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = ["0: LANE_LEFT", "1: IDLE", "2: LANE_RIGHT", "3: FASTER", "4: SLOWER"]
SHORT_NAMES = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]


def compute_offline_metrics(model, x_val, y_val):
    """Compute offline validation metrics for a single model."""
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    probs = model(x_val, training=False).numpy()
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
    f3_to_4 = int(cm[3, 4])  # FASTER -> SLOWER
    f4_to_3 = int(cm[4, 3])  # SLOWER -> FASTER

    return {
        "accuracy": accuracy,
        "loss": loss,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "total_errors": total_errors,
        "faster_to_slower": f3_to_4,
        "slower_to_faster": f4_to_3,
        "mutual_fs_errors": f3_to_4 + f4_to_3,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_env_100_episodes(model, seeds):
    """
    Run 100 paired episodes in roundabout-v0.
    Tracks speed, target speed, availability, and arrival.
    """
    env = gym.make("roundabout-v0", render_mode="rgb_array")
    episodes_data = []

    total_actions = np.zeros(5, dtype=int)
    lat_infeasible_count = 0
    long_infeasible_count = 0
    total_steps_all = 0

    all_actual_speeds = []
    all_target_speeds = []
    stopped_steps_total = 0

    for idx, seed in enumerate(seeds):
        obs, info = env.reset(seed=seed)
        done = False
        truncated = False
        ep_reward = 0.0
        ep_steps = 0
        ep_actions = []

        ep_speeds = []
        ep_target_speeds = []

        while not (done or truncated):
            obs_input = obs.reshape(1, 25).astype(np.float32)
            action = int(np.argmax(model(obs_input, training=False).numpy()[0]))

            # Check action availability
            available = env.unwrapped.action_type.get_available_actions()
            if action not in available:
                if action in [0, 2]:
                    lat_infeasible_count += 1
                elif action in [3, 4]:
                    long_infeasible_count += 1

            total_actions[action] += 1
            ep_actions.append(action)

            # Record speeds
            act_spd = float(env.unwrapped.vehicle.speed)
            tgt_spd = float(env.unwrapped.vehicle.target_speed)
            ep_speeds.append(act_spd)
            ep_target_speeds.append(tgt_spd)

            if act_spd < 1.0:
                stopped_steps_total += 1

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_steps += 1
            done = bool(terminated)
            truncated = bool(truncated)

        crashed = bool(info.get("crashed", False) or env.unwrapped.vehicle.crashed)
        final_lane = str(env.unwrapped.vehicle.lane_index)
        reached_dest = ("nxs" in final_lane) and (not crashed)

        episodes_data.append({
            "episode": idx + 1,
            "seed": seed,
            "reward": ep_reward,
            "steps": ep_steps,
            "crashed": crashed,
            "success": reached_dest,
            "truncated": truncated,
            "final_lane": final_lane,
            "mean_actual_speed": float(np.mean(ep_speeds)),
            "mean_target_speed": float(np.mean(ep_target_speeds)),
        })
        all_actual_speeds.extend(ep_speeds)
        all_target_speeds.extend(ep_target_speeds)
        total_steps_all += ep_steps

    env.close()

    rewards = np.array([e["reward"] for e in episodes_data])
    crashes = sum(1 for e in episodes_data if e["crashed"])
    successes = sum(1 for e in episodes_data if e["success"])
    truncations = sum(1 for e in episodes_data if e["truncated"])

    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_median": float(np.median(rewards)),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "collision_count": int(crashes),
        "collision_rate": float(crashes / len(seeds) * 100),
        "success_count": int(successes),
        "success_rate": float(successes / len(seeds) * 100),
        "truncated_count": int(truncations),
        "truncated_rate": float(truncations / len(seeds) * 100),
        "mean_steps": float(total_steps_all / len(seeds)),
        "total_steps": int(total_steps_all),
        "mean_actual_speed": float(np.mean(all_actual_speeds)),
        "mean_target_speed": float(np.mean(all_target_speeds)),
        "stopped_steps_rate": float(stopped_steps_total / total_steps_all * 100),
        "lat_infeasible_count": int(lat_infeasible_count),
        "lat_infeasible_rate": float(lat_infeasible_count / total_steps_all * 100),
        "long_infeasible_count": int(long_infeasible_count),
        "long_infeasible_rate": float(long_infeasible_count / total_steps_all * 100),
        "total_infeasible_rate": float((lat_infeasible_count + long_infeasible_count) / total_steps_all * 100),
        "action_counts": total_actions.tolist(),
        "action_percentages": (total_actions / total_steps_all * 100).tolist(),
        "episodes": episodes_data,
    }


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(root_dir, "models")
    reports_dir = os.path.join(root_dir, "reports")
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 70)
    print(" PHASE 3B — TASK D: OFFLINE EVALUATION (ALL 3 MODELS)")
    print("=" * 70)

    val_path = os.path.join(root_dir, "dataset", "validation.npz")
    val_data = np.load(val_path)
    x_val = val_data["data"].astype(np.float32)
    y_val = val_data["label"].astype(np.int32).flatten()

    models_info = {
        "Baseline Standard (Control)": {
            "path": os.path.join(models_dir, "baseline_standard.h5"),
            "arch": "25 -> 128 -> 64 -> 5",
            "lr": 0.001,
            "params": 11909,
        },
        "Wider Network (Exp B)": {
            "path": os.path.join(models_dir, "optimized_wider.h5"),
            "arch": "25 -> 256 -> 128 -> 5",
            "lr": 0.001,
            "params": 40197,
        },
        "Lower Learning Rate (Exp C)": {
            "path": os.path.join(models_dir, "optimized_low_lr.h5"),
            "arch": "25 -> 128 -> 64 -> 5",
            "lr": 0.0003,
            "params": 11909,
        },
    }

    offline_results = {}
    loaded_models = {}

    for name, info in models_info.items():
        print(f"Evaluating {name} from {info['path']}...")
        model = tf.keras.models.load_model(info["path"])
        loaded_models[name] = model
        res = compute_offline_metrics(model, x_val, y_val)
        offline_results[name] = {**info, **res}

    print("\n" + "=" * 85)
    print(f"{'Model Name':<30} | {'Val Acc':<9} | {'Val Loss':<9} | {'Macro F1':<9} | {'Errors':<7} | {'F<->S Confusion'}")
    print("-" * 85)
    for name, r in offline_results.items():
        print(f"{name:<30} | {r['accuracy']*100:<8.2f}% | {r['loss']:<9.4f} | {r['macro_f1']*100:<8.2f}% | {r['total_errors']:<7d} | {r['mutual_fs_errors']} ({r['faster_to_slower']}/{r['slower_to_faster']})")
    print("=" * 85)

    # Determine best model by Validation Accuracy
    best_opt_name = max(
        ["Wider Network (Exp B)", "Lower Learning Rate (Exp C)"],
        key=lambda k: offline_results[k]["accuracy"]
    )
    best_opt_model = loaded_models[best_opt_name]
    baseline_model = loaded_models["Baseline Standard (Control)"]

    print(f"\nBest Optimized Model Selected: '{best_opt_name}'")
    print(f"  - Validation Accuracy: {offline_results[best_opt_name]['accuracy']*100:.2f}% (vs Baseline: {offline_results['Baseline Standard (Control)']['accuracy']*100:.2f}%)")
    acc_diff = (offline_results[best_opt_name]['accuracy'] - offline_results['Baseline Standard (Control)']['accuracy']) * 100
    print(f"  - Accuracy Delta:      {acc_diff:+.2f}%")

    # =========================================================================
    # TASK E: Environment Evaluation (100 Episodes)
    # =========================================================================
    print("\n" + "=" * 70)
    print(f" PHASE 3B — TASK E: ENVIRONMENT EVALUATION (Control vs Best: {best_opt_name})")
    print("=" * 70)
    SEEDS = list(range(100))

    print(f"Running 100 paired episodes for Control (Baseline Standard)...")
    env_control = evaluate_env_100_episodes(baseline_model, SEEDS)

    print(f"Running 100 paired episodes for Best Optimized ({best_opt_name})...")
    env_best = evaluate_env_100_episodes(best_opt_model, SEEDS)

    # Paired Bootstrap Comparison (10,000 resamples)
    diff_rewards = np.array([e["reward"] for e in env_best["episodes"]]) - np.array([e["reward"] for e in env_control["episodes"]])
    diff_crashes = np.array([int(e["crashed"]) for e in env_best["episodes"]]) - np.array([int(e["crashed"]) for e in env_control["episodes"]])

    np.random.seed(42)
    boot_r = [np.mean(np.random.choice(diff_rewards, size=100, replace=True)) for _ in range(10000)]
    boot_c = [np.mean(np.random.choice(diff_crashes, size=100, replace=True)) * 100 for _ in range(10000)]

    ci_r = [float(np.percentile(boot_r, 2.5)), float(np.percentile(boot_r, 97.5))]
    ci_c = [float(np.percentile(boot_c, 2.5)), float(np.percentile(boot_c, 97.5))]

    bootstrap_stats = {
        "mean_reward_diff": float(np.mean(diff_rewards)),
        "reward_diff_ci_95": ci_r,
        "reward_diff_sig": not (ci_r[0] <= 0 <= ci_r[1]),
        "collision_diff_pct": float(np.mean(diff_crashes) * 100),
        "collision_diff_ci_95": ci_c,
        "collision_diff_sig": not (ci_c[0] <= 0 <= ci_c[1]),
    }

    print("\n" + "=" * 75)
    print(f"{'Environment Metric':<30} | {'Baseline Standard':<18} | {best_opt_name[:18]:<18}")
    print("-" * 75)
    print(f"{'Mean Episode Reward':<30} | {env_control['reward_mean']:<18.2f} | {env_best['reward_mean']:<18.2f}")
    print(f"{'Reward Std Dev':<30} | {env_control['reward_std']:<18.2f} | {env_best['reward_std']:<18.2f}")
    print(f"{'Median Reward':<30} | {env_control['reward_median']:<18.2f} | {env_best['reward_median']:<18.2f}")
    print(f"{'Collision Rate':<30} | {env_control['collision_rate']:.1f}% ({env_control['collision_count']:2d})       | {env_best['collision_rate']:.1f}% ({env_best['collision_count']:2d})")
    print(f"{'Destination Arrival Rate':<30} | {env_control['success_rate']:.1f}% ({env_control['success_count']:2d})       | {env_best['success_rate']:.1f}% ({env_best['success_count']:2d})")
    print(f"{'Mean Actual Speed (m/s)':<30} | {env_control['mean_actual_speed']:<18.2f} | {env_best['mean_actual_speed']:<18.2f}")
    print(f"{'Mean Target Speed (m/s)':<30} | {env_control['mean_target_speed']:<18.2f} | {env_best['mean_target_speed']:<18.2f}")
    print(f"{'Time Stopped / Crawl (<1m/s)':<30} | {env_control['stopped_steps_rate']:.2f}%            | {env_best['stopped_steps_rate']:.2f}%")
    print(f"{'Infeasible Lane Changes':<30} | {env_control['lat_infeasible_rate']:.2f}% ({env_control['lat_infeasible_count']:2d})      | {env_best['lat_infeasible_rate']:.2f}% ({env_best['lat_infeasible_count']:2d})")
    print(f"{'Infeasible Speed Changes':<30} | {env_control['long_infeasible_rate']:.2f}% ({env_control['long_infeasible_count']:2d})     | {env_best['long_infeasible_rate']:.2f}% ({env_best['long_infeasible_count']:2d})")
    print(f"{'Total Invalid Action Rate':<30} | {env_control['total_infeasible_rate']:<17.2f}% | {env_best['total_infeasible_rate']:<17.2f}%")
    print("=" * 75)
    print(f"Bootstrap 95% CI for Reward Diff:    [{ci_r[0]:+.4f}, {ci_r[1]:+.4f}] (Significant: {bootstrap_stats['reward_diff_sig']})")
    print(f"Bootstrap 95% CI for Collision Diff: [{ci_c[0]:+.2f}%, {ci_c[1]:+.2f}%] (Significant: {bootstrap_stats['collision_diff_sig']})")

    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    print("\nGenerating Phase 3B Figures...")

    # Figure 1: Hyperparameter Comparison (Bar chart of Accuracy, Loss, Macro F1)
    m_keys = list(offline_results.keys())
    m_accs = [offline_results[k]["accuracy"] * 100 for k in m_keys]
    m_f1s = [offline_results[k]["macro_f1"] * 100 for k in m_keys]
    m_losses = [offline_results[k]["loss"] for k in m_keys]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    x_idx = np.arange(len(m_keys))
    w = 0.28
    r1 = ax1.bar(x_idx - w/2, m_accs, w, label="Validation Accuracy (%)", color="#4285F4")
    r2 = ax1.bar(x_idx + w/2, m_f1s, w, label="Macro F1-score (%)", color="#34A853")

    ax1.set_ylabel("Percentage (%)")
    ax1.set_ylim(85, 98)
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(["Baseline (Control)", "Wider Network", "Lower LR"], rotation=10)
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    for r in r1:
        h = r.get_height()
        ax1.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for r in r2:
        h = r.get_height()
        ax1.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.title("Phase 3B: Offline Validation Accuracy and Macro F1 Comparison", fontsize=12)
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "opt_hyperparameter_comparison.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")

    # Figure 2: Confusion Matrices for All 3 Models
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (m_name, m_res) in zip(axes, offline_results.items()):
        cm = np.array(m_res["confusion_matrix"])
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{m_name.split()[0]} ({m_res['accuracy']*100:.2f}%)", fontsize=11)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(SHORT_NAMES, rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(SHORT_NAMES, fontsize=8)
        thresh = cm.max() / 2.0
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.suptitle("Confusion Matrices: Baseline vs Wider vs Lower LR", fontsize=13)
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "opt_confusion_matrices.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")

    # Figure 3: Per-class Recall Comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.25
    x = np.arange(5)
    colors = ["#4285F4", "#EA4335", "#FBBC05"]
    for idx, (m_name, m_res) in enumerate(offline_results.items()):
        recalls = [m_res["per_class"][c]["recall"] * 100 for c in CLASS_NAMES]
        ax.bar(x + (idx - 1)*w, recalls, w, label=m_name.split()[0], color=colors[idx])
    ax.set_ylabel("Recall (%)")
    ax.set_title("Per-class Recall Comparison Across Hyperparameter Experiments")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_NAMES)
    ax.set_ylim(80, 102)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "opt_per_class_recall.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")

    # Figure 4: Training Curves from Histories
    hist_std_path = os.path.join(models_dir, "training_history.json")
    hist_opt_path = os.path.join(models_dir, "optimized_training_history.json")

    if os.path.exists(hist_std_path) and os.path.exists(hist_opt_path):
        with open(hist_std_path) as f1, open(hist_opt_path) as f2:
            h_std = json.load(f1)["exp_a"]["history"]
            h_opt = json.load(f2)
            h_wider = h_opt["exp_b_wider"]["history"]
            h_low_lr = h_opt["exp_c_low_lr"]["history"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # Loss
        ax1.plot(h_std["val_loss"], label="Baseline (lr=0.001)", color="#4285F4")
        ax1.plot(h_wider["val_loss"], label="Wider (256-128)", color="#EA4335")
        ax1.plot(h_low_lr["val_loss"], label="Lower LR (lr=0.0003)", color="#FBBC05")
        ax1.set_title("Validation Loss Curves")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Validation Loss")
        ax1.legend()
        ax1.grid(True, linestyle=":", alpha=0.6)

        # Accuracy
        ax2.plot([x * 100 for x in h_std["val_accuracy"]], label="Baseline", color="#4285F4")
        ax2.plot([x * 100 for x in h_wider["val_accuracy"]], label="Wider", color="#EA4335")
        ax2.plot([x * 100 for x in h_low_lr["val_accuracy"]], label="Lower LR", color="#FBBC05")
        ax2.set_title("Validation Accuracy Curves (%)")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Validation Accuracy (%)")
        ax2.legend()
        ax2.grid(True, linestyle=":", alpha=0.6)

        fig.suptitle("Training Convergence & Generalization Curves", fontsize=13)
        plt.tight_layout()
        fig4_path = os.path.join(figures_dir, "opt_training_curves.png")
        plt.savefig(fig4_path, dpi=300)
        plt.close()
        print(f"Saved: {fig4_path}")

    # Figure 5: Environment Comparison (Control vs Best Optimized)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    # Rewards distribution
    r_ctrl = [e["reward"] for e in env_control["episodes"]]
    r_best = [e["reward"] for e in env_best["episodes"]]
    ax1.boxplot([r_ctrl, r_best], tick_labels=["Baseline Control", best_opt_name[:12]], patch_artist=True,
                boxprops=dict(facecolor="#D2E3FC"), medianprops=dict(color="red", linewidth=2))
    ax1.set_title("100-Episode Cumulative Reward")
    ax1.set_ylabel("Reward")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Collision & Arrival
    cats = ["Collision Rate", "Arrival Rate"]
    c_vals = [env_control["collision_rate"], env_control["success_rate"]]
    b_vals = [env_best["collision_rate"], env_best["success_rate"]]
    x_c = np.arange(2)
    w_c = 0.35
    ax2.bar(x_c - w_c/2, c_vals, w_c, label="Baseline Control", color="#4285F4")
    ax2.bar(x_c + w_c/2, b_vals, w_c, label=best_opt_name[:12], color="#34A853")
    ax2.set_title("Collision vs Destination Arrival Rates")
    ax2.set_ylabel("Rate (%)")
    ax2.set_xticks(x_c)
    ax2.set_xticklabels(cats)
    ax2.set_ylim(0, 45)
    ax2.legend()
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    for i, (cv, bv) in enumerate(zip(c_vals, b_vals)):
        ax2.text(i - w_c/2, cv + 1, f"{cv:.1f}%", ha="center", fontsize=9)
        ax2.text(i + w_c/2, bv + 1, f"{bv:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    fig5_path = os.path.join(figures_dir, "opt_env_comparison.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"Saved: {fig5_path}")

    # Compile Final JSON Report
    final_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_optimized_model": best_opt_name,
        "offline_evaluation": offline_results,
        "environment_evaluation": {
            "control": env_control,
            "best_optimized": env_best,
            "paired_bootstrap": bootstrap_stats,
        }
    }

    results_json_path = os.path.join(reports_dir, "phase3b_results.json")
    with open(results_json_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nFinal Phase 3B results saved to: {results_json_path}")
    print("=" * 70)
    print(" PHASE 3B EVALUATION SCRIPT FINISHED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()

