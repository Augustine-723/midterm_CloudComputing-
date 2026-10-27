#!/usr/bin/env python3
"""
scripts/evaluate_environment.py
Evaluate Baseline Neural Network Models in highway-env roundabout-v0 environment.
- Evaluates models/baseline_standard.h5 and models/baseline_weighted.h5
- Runs 100 paired episodes using fixed seeds (0-99)
- Records episode rewards, collisions, actions, and terminations
- Generates 4 comparison visualizations in reports/figures/
- Saves complete results to reports/phase3a_results.json
"""

import json
import os
import sys
import time
import numpy as np

# Headless & TF logging suppression
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gymnasium as gym
import highway_env
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACTION_NAMES = [
    "0: LANE_LEFT",
    "1: IDLE",
    "2: LANE_RIGHT",
    "3: FASTER",
    "4: SLOWER"
]
SHORT_ACTION_NAMES = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]


def run_evaluation_for_model(model_path, model_name, seeds):
    """
    Evaluate a single model on roundabout-v0 across specified seeds.
    """
    print(f"\n" + "-" * 70)
    print(f" Evaluating: {model_name} ({os.path.basename(model_path)})")
    print("-" * 70)

    model = tf.keras.models.load_model(model_path)
    env = gym.make("roundabout-v0", render_mode="rgb_array")

    episodes_data = []
    total_actions_count = np.zeros(5, dtype=int)
    total_infeasible_actions = 0
    total_steps_all = 0

    start_time = time.time()

    for idx, seed in enumerate(seeds):
        obs, info = env.reset(seed=seed)
        done = False
        truncated = False
        ep_reward = 0.0
        ep_steps = 0
        ep_actions = []
        ep_infeasible_count = 0

        while not (done or truncated):
            obs_input = obs.reshape(1, 25).astype(np.float32)

            # Direct tensor call for fast inference
            probs = model(obs_input, training=False).numpy()[0]
            action = int(np.argmax(probs))

            # Check if action is currently available
            available_actions = env.unwrapped.action_type.get_available_actions()
            is_infeasible = (action not in available_actions)
            if is_infeasible:
                ep_infeasible_count += 1

            total_actions_count[action] += 1
            ep_actions.append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_steps += 1
            done = bool(terminated)
            truncated = bool(truncated)

        # Episode final state analysis
        crashed = bool(info.get("crashed", False) or env.unwrapped.vehicle.crashed)
        final_lane = env.unwrapped.vehicle.lane_index

        # Success definition: Reached the North exit destination lane without crashing
        # Route destination in roundabout_env is "nxs"
        reached_destination = ("nxs" in str(final_lane))
        success = reached_destination and (not crashed)

        ep_data = {
            "episode": idx + 1,
            "seed": seed,
            "reward": ep_reward,
            "steps": ep_steps,
            "crashed": crashed,
            "terminated": done,
            "truncated": truncated,
            "success": success,
            "final_lane": str(final_lane),
            "final_speed": float(env.unwrapped.vehicle.speed),
            "infeasible_actions": ep_infeasible_count,
            "action_counts": [int(ep_actions.count(a)) for a in range(5)],
        }
        episodes_data.append(ep_data)
        total_infeasible_actions += ep_infeasible_count
        total_steps_all += ep_steps

        if (idx + 1) % 25 == 0:
            print(f"  Processed {idx + 1}/100 episodes... (Current Mean Reward: {np.mean([e['reward'] for e in episodes_data]):.2f})")

    env.close()
    duration = time.time() - start_time

    # Aggregate Statistics
    rewards = np.array([e["reward"] for e in episodes_data])
    steps = np.array([e["steps"] for e in episodes_data])
    crashes = sum(1 for e in episodes_data if e["crashed"])
    successes = sum(1 for e in episodes_data if e["success"])
    truncations = sum(1 for e in episodes_data if e["truncated"])
    terminations = sum(1 for e in episodes_data if e["terminated"])

    action_percentages = (total_actions_count / total_steps_all * 100).tolist()
    infeasible_rate = (total_infeasible_actions / total_steps_all * 100) if total_steps_all > 0 else 0.0

    summary = {
        "model_name": model_name,
        "model_path": model_path,
        "n_episodes": len(seeds),
        "duration_seconds": duration,
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_median": float(np.median(rewards)),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "steps_mean": float(np.mean(steps)),
        "steps_total": int(total_steps_all),
        "collision_count": int(crashes),
        "collision_rate": float(crashes / len(seeds) * 100),
        "success_count": int(successes),
        "success_rate": float(successes / len(seeds) * 100),
        "terminated_count": int(terminations),
        "terminated_rate": float(terminations / len(seeds) * 100),
        "truncated_count": int(truncations),
        "truncated_rate": float(truncations / len(seeds) * 100),
        "infeasible_actions_count": int(total_infeasible_actions),
        "infeasible_actions_rate": float(infeasible_rate),
        "action_counts": total_actions_count.tolist(),
        "action_percentages": action_percentages,
    }

    print(f"\n  [Results for {model_name}]")
    print(f"    - Mean Reward:      {summary['reward_mean']:.2f} +/- {summary['reward_std']:.2f} (Median: {summary['reward_median']:.2f})")
    print(f"    - Collision Rate:   {summary['collision_rate']:.1f}% ({crashes}/{len(seeds)})")
    print(f"    - Success Rate:     {summary['success_rate']:.1f}% ({successes}/{len(seeds)})")
    print(f"    - Mean Steps:       {summary['steps_mean']:.1f}")
    print(f"    - Infeasible Acts:  {summary['infeasible_actions_rate']:.2f}%")
    print(f"    - Action Dist:      {', '.join([f'{SHORT_ACTION_NAMES[i]}: {action_percentages[i]:.1f}%' for i in range(5)])}")

    return {
        "summary": summary,
        "episodes": episodes_data,
    }


def generate_visualizations(res_a, res_b, figures_dir):
    """
    Generate all 4 required comparison visualizations.
    """
    sum_a = res_a["summary"]
    sum_b = res_b["summary"]
    eps_a = res_a["episodes"]
    eps_b = res_b["episodes"]

    rewards_a = [e["reward"] for e in eps_a]
    rewards_b = [e["reward"] for e in eps_b]
    ep_numbers = range(1, len(rewards_a) + 1)

    # 1. Episode Reward Comparison (Paired per-episode curve)
    plt.figure(figsize=(11, 5))
    plt.plot(ep_numbers, rewards_a, label=f"Standard Baseline (Mean: {sum_a['reward_mean']:.2f})", color="#4285F4", alpha=0.85, linewidth=1.5)
    plt.plot(ep_numbers, rewards_b, label=f"Class Weighted (Mean: {sum_b['reward_mean']:.2f})", color="#EA4335", alpha=0.85, linewidth=1.5)
    plt.axhline(sum_a["reward_mean"], color="#4285F4", linestyle="--", alpha=0.7)
    plt.axhline(sum_b["reward_mean"], color="#EA4335", linestyle="--", alpha=0.7)
    plt.title("Episode Reward Comparison across 100 Paired Episodes in roundabout-v0", fontsize=12)
    plt.xlabel("Episode Number (Seed 0–99)")
    plt.ylabel("Cumulative Reward")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "env_episode_rewards.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")

    # 2. Reward Distribution (Histogram & Boxplot)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    bins = np.linspace(min(min(rewards_a), min(rewards_b)), max(max(rewards_a), max(rewards_b)), 15)
    ax1.hist(rewards_a, bins=bins, alpha=0.6, label=f"Standard (Median: {sum_a['reward_median']:.2f})", color="#4285F4", edgecolor="black")
    ax1.hist(rewards_b, bins=bins, alpha=0.6, label=f"Weighted (Median: {sum_b['reward_median']:.2f})", color="#EA4335", edgecolor="black")
    ax1.set_title("Reward Histogram")
    ax1.set_xlabel("Cumulative Episode Reward")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.boxplot([rewards_a, rewards_b], tick_labels=["Standard Baseline", "Class Weighted"], patch_artist=True,
                boxprops=dict(facecolor="#D2E3FC"), medianprops=dict(color="red", linewidth=2))
    ax2.set_title("Reward Boxplot Comparison")
    ax2.set_ylabel("Cumulative Episode Reward")
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("Environment Reward Distribution (100 Episodes)", fontsize=13)
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "env_reward_distribution.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")

    # 3. Action Distribution Comparison (Bar chart)
    x = np.arange(len(SHORT_ACTION_NAMES))
    width = 0.35

    plt.figure(figsize=(9, 5))
    rects1 = plt.bar(x - width/2, sum_a["action_percentages"], width, label="Standard Baseline", color="#4285F4")
    rects2 = plt.bar(x + width/2, sum_b["action_percentages"], width, label="Class Weighted", color="#EA4335")

    plt.ylabel("Selection Frequency (%)")
    plt.title("Action Selection Distribution in roundabout-v0 Simulation")
    plt.xticks(x, SHORT_ACTION_NAMES)
    plt.ylim(0, max(max(sum_a["action_percentages"]), max(sum_b["action_percentages"])) * 1.15)
    plt.legend()
    plt.grid(axis="y", linestyle=":", alpha=0.6)

    for rect in rects1:
        h = rect.get_height()
        plt.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        plt.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "env_action_distribution.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")

    # 4. Collision Rate & Success Rate Comparison
    categories = ["Collision Rate", "Success Rate", "Truncation Rate"]
    val_a = [sum_a["collision_rate"], sum_a["success_rate"], sum_a["truncated_rate"]]
    val_b = [sum_b["collision_rate"], sum_b["success_rate"], sum_b["truncated_rate"]]

    x_cat = np.arange(len(categories))
    plt.figure(figsize=(8, 5))
    r1 = plt.bar(x_cat - width/2, val_a, width, label="Standard Baseline", color="#4285F4")
    r2 = plt.bar(x_cat + width/2, val_b, width, label="Class Weighted", color="#EA4335")

    plt.ylabel("Rate (%)")
    plt.title("Collision, Success, and Truncation Rates (100 Episodes)")
    plt.xticks(x_cat, categories)
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(axis="y", linestyle=":", alpha=0.6)

    for r in r1:
        h = r.get_height()
        plt.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for r in r2:
        h = r.get_height()
        plt.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, "env_collision_success_rate.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved: {fig4_path}")


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(root_dir, "models")
    reports_dir = os.path.join(root_dir, "reports")
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 70)
    print(" PHASE 3A — ROUNDABOUT-V0 ENVIRONMENT EVALUATION")
    print("=" * 70)
    print(f"Python:     {sys.version.split()[0]}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"highway-env: 1.8.2")

    # Use fixed seeds 0 to 99 for paired comparison
    SEEDS = list(range(100))
    print(f"Paired evaluation configuration: 100 Episodes with seeds 0 through 99.")

    path_standard = os.path.join(models_dir, "baseline_standard.h5")
    path_weighted = os.path.join(models_dir, "baseline_weighted.h5")

    assert os.path.exists(path_standard), f"Model not found: {path_standard}"
    assert os.path.exists(path_weighted), f"Model not found: {path_weighted}"

    # Evaluate Model A
    res_a = run_evaluation_for_model(path_standard, "Standard Baseline", SEEDS)

    # Evaluate Model B
    res_b = run_evaluation_for_model(path_weighted, "Class Weighted Baseline", SEEDS)

    # Generate Visualizations
    print("\n[3] Generating Visualizations...")
    generate_visualizations(res_a, res_b, figures_dir)

    # Compile Final JSON Report
    output_results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "environment": "roundabout-v0",
            "n_episodes": len(SEEDS),
            "seeds": SEEDS,
            "highway_env_version": "1.8.2",
            "gymnasium_version": gym.__version__,
            "tensorflow_version": tf.__version__,
        },
        "standard_baseline": res_a["summary"],
        "weighted_baseline": res_b["summary"],
        "episodes_standard": res_a["episodes"],
        "episodes_weighted": res_b["episodes"],
    }

    results_json_path = os.path.join(reports_dir, "phase3a_results.json")
    with open(results_json_path, "w") as f:
        json.dump(output_results, f, indent=2)
    print(f"\nSaved results JSON to: {results_json_path}")

    # Print Summary Comparison Table
    sum_a = res_a["summary"]
    sum_b = res_b["summary"]
    print("\n" + "=" * 70)
    print(f"{'Environment Metric':<30} | {'Standard Baseline':<18} | {'Class Weighted':<18}")
    print("-" * 70)
    print(f"{'Mean Episode Reward':<30} | {sum_a['reward_mean']:<18.2f} | {sum_b['reward_mean']:<18.2f}")
    print(f"{'Reward Std Dev':<30} | {sum_a['reward_std']:<18.2f} | {sum_b['reward_std']:<18.2f}")
    print(f"{'Median Reward':<30} | {sum_a['reward_median']:<18.2f} | {sum_b['reward_median']:<18.2f}")
    print(f"{'Min / Max Reward':<30} | {sum_a['reward_min']:.2f} / {sum_a['reward_max']:.2f}    | {sum_b['reward_min']:.2f} / {sum_b['reward_max']:.2f}")
    print(f"{'Mean Episode Steps':<30} | {sum_a['steps_mean']:<18.2f} | {sum_b['steps_mean']:<18.2f}")
    print(f"{'Collision Count (Rate)':<30} | {sum_a['collision_count']} ({sum_a['collision_rate']:.1f}%)         | {sum_b['collision_count']} ({sum_b['collision_rate']:.1f}%)")
    print(f"{'Destination Success Rate':<30} | {sum_a['success_count']} ({sum_a['success_rate']:.1f}%)         | {sum_b['success_count']} ({sum_b['success_rate']:.1f}%)")
    print(f"{'Truncation Rate (Max Time)':<30} | {sum_a['truncated_count']} ({sum_a['truncated_rate']:.1f}%)         | {sum_b['truncated_count']} ({sum_b['truncated_rate']:.1f}%)")
    print(f"{'Infeasible Actions Rate':<30} | {sum_a['infeasible_actions_rate']:<17.2f}% | {sum_b['infeasible_actions_rate']:<17.2f}%")
    print("=" * 70)
    print(" PHASE 3A EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()

