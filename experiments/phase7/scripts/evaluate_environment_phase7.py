#!/usr/bin/env python3
"""
experiments/phase7/scripts/evaluate_environment_phase7.py
Phase 7A: Real Environment (roundabout-v0) Evaluation of Experiments B and C vs Control.

Evaluates 100 episodes (10 rounds x 10 episodes) under deterministic paired seeds (seed_base=42):
- Control: Baseline Standard (models/baseline_standard.h5)
- Experiment B: Feature Engineering MLP (experiments/phase7/models/fe_mlp_best.h5)
- Experiment C: Vehicle-Aware Network (experiments/phase7/models/vehicle_aware_best.h5)

Computes:
- Mean Round Reward (float sum & integer truncated)
- Collision Rate
- North Exit Destination Arrival Rate (using spatial "nxs" in final_lane criterion)
- Action Distribution
- Speed Saturation Rate (SLOWER at speed 0 or FASTER at speed 16)
- Appends results to experiments/phase7/reports/phase7_results.json
"""

import os
import sys
import json
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import gymnasium as gym
import highway_env

CLASS_NAMES = ["0: LANE_LEFT", "1: IDLE", "2: LANE_RIGHT", "3: FASTER", "4: SLOWER"]
SHORT_NAMES = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
RESULTS_JSON_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase7", "reports", "phase7_results.json")

EVAL_MODELS = [
    {
        "key": "control_baseline",
        "name": "Control: Baseline Standard",
        "path": os.path.join(PROJECT_ROOT, "models", "baseline_standard.h5")
    },
    {
        "key": "experiment_b",
        "name": "Exp B: Feature Engineering MLP",
        "path": os.path.join(PROJECT_ROOT, "experiments", "phase7", "models", "fe_mlp_best.h5")
    },
    {
        "key": "experiment_c",
        "name": "Exp C: Vehicle-Aware Network",
        "path": os.path.join(PROJECT_ROOT, "experiments", "phase7", "models", "vehicle_aware_best.h5")
    }
]


def evaluate_in_environment(model_path, model_name, num_rounds=10, episodes_per_round=10, seed_base=42):
    print("\n" + "=" * 70)
    print(f" ENVIRONMENT EVALUATION: {model_name}")
    print(f" Model Path: {model_path}")
    print("=" * 70)

    model = tf.keras.models.load_model(model_path, compile=False)
    env = gym.make("roundabout-v0", render_mode="rgb_array")

    total_episodes = num_rounds * episodes_per_round
    round_rewards_float = []
    round_rewards_int = []
    all_ep_rewards = []
    collisions = 0
    spatial_arrivals = 0
    action_counts = {c: 0 for c in range(5)}
    speed_saturation_count = 0
    total_steps = 0

    for r in range(num_rounds):
        round_sum = 0.0
        for ep in range(episodes_per_round):
            ep_seed = seed_base + r * 100 + ep
            obs, info = env.reset(seed=ep_seed)
            done = False
            truncated = False
            ep_reward = 0.0

            while not (done or truncated):
                current_speed = float(env.unwrapped.vehicle.speed)
                obs_reshaped = obs.reshape(1, 25).astype(np.float32)

                # Direct forward pass
                probs = model(tf.constant(obs_reshaped), training=False).numpy()
                action = int(np.argmax(probs[0]))
                action_counts[action] += 1
                total_steps += 1

                # Check speed saturation: calling SLOWER at min speed (0) or FASTER at max speed (16)
                if (action == 4 and np.isclose(current_speed, 0.0, atol=0.5)) or \
                   (action == 3 and np.isclose(current_speed, 16.0, atol=0.5)):
                    speed_saturation_count += 1

                obs, reward, done, truncated, info = env.step(action)
                ep_reward += float(reward)

            round_sum += ep_reward
            all_ep_rewards.append(ep_reward)

            crashed = bool(info.get("crashed", False) or env.unwrapped.vehicle.crashed)
            final_lane = env.unwrapped.vehicle.lane_index
            reached_destination = ("nxs" in str(final_lane))
            
            if crashed:
                collisions += 1
            if reached_destination and not crashed:
                spatial_arrivals += 1

        round_rewards_float.append(round_sum)
        round_rewards_int.append(int(round_sum))
        print(f"  Round {r + 1:02d}/{num_rounds}: Float Sum = {round_sum:6.2f} | Int Truncated = {int(round_sum):3d}")

    env.close()

    mean_round_float = float(np.mean(round_rewards_float))
    std_round_float = float(np.std(round_rewards_float))
    mean_round_int = float(np.mean(round_rewards_int))
    mean_ep_reward = float(np.mean(all_ep_rewards))
    collision_rate = float(collisions / total_episodes * 100.0)
    arrival_rate = float(spatial_arrivals / total_episodes * 100.0)
    saturation_rate = float(speed_saturation_count / total_steps * 100.0) if total_steps > 0 else 0.0

    action_dist = {
        CLASS_NAMES[c]: round(count / sum(action_counts.values()) * 100.0, 2)
        for c, count in action_counts.items()
    }

    env_metrics = {
        "mean_round_reward_float": round(mean_round_float, 2),
        "std_round_reward_float": round(std_round_float, 2),
        "mean_round_reward_int": round(mean_round_int, 2),
        "mean_episode_reward": round(mean_ep_reward, 2),
        "teacher_round_rewards_int": round_rewards_int,
        "collision_count": collisions,
        "collision_rate_percent": round(collision_rate, 2),
        "north_exit_arrival_count": spatial_arrivals,
        "north_exit_arrival_rate_percent": round(arrival_rate, 2),
        "speed_saturation_rate_percent": round(saturation_rate, 2),
        "action_distribution": action_dist
    }

    print("\n" + "-" * 60)
    print(f" Summary for {model_name}:")
    print(f"   * Mean Round Reward (Float): {mean_round_float:.2f} +/- {std_round_float:.2f}")
    print(f"   * Mean Round Reward (Int):   {mean_round_int:.2f}")
    print(f"   * Collision Rate:            {collision_rate:.1f}% ({collisions}/{total_episodes})")
    print(f"   * North Exit Arrival Rate:   {arrival_rate:.1f}% ({spatial_arrivals}/{total_episodes})")
    print(f"   * Speed Saturation Rate:     {saturation_rate:.1f}%")
    print(f"   * Action Dist:               {action_dist}")
    print("-" * 60)

    return env_metrics


def main():
    print("=" * 70)
    print(" PHASE 7A: REAL SIMULATION ENVIRONMENT EVALUATION (ROUNDABOUT-V0)")
    print("=" * 70)

    env_results = {}
    for target in EVAL_MODELS:
        res = evaluate_in_environment(
            model_path=target["path"],
            model_name=target["name"],
            num_rounds=10,
            episodes_per_round=10,
            seed_base=42
        )
        env_results[target["key"]] = res

    # Append to phase7_results.json
    existing_data = {}
    if os.path.exists(RESULTS_JSON_PATH):
        with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
            existing_data = json.load(f)

    for key, data in env_results.items():
        if key in existing_data:
            existing_data[key]["environment_evaluation"] = data
        else:
            existing_data[key] = {"environment_evaluation": data}

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
    print(f"\nEnvironment evaluation appended to: {RESULTS_JSON_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()

