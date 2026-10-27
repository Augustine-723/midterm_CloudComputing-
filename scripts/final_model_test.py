#!/usr/bin/env python3
"""
scripts/final_model_test.py
Phase 6A: Final Model Integration and Teacher Test Compatibility.

Faithfully implements the evaluation loop required by the teacher's NN_test.py:
- 10 rounds x 10 episodes = 100 episodes total
- Direct forward pass: model(obs, training=False)
- Argmax action selection: np.argmax()
- Real highway-env roundabout-v0 environment
- Clean gym reset, termination handling, and env.close()
- Strictly NO action masking, NO rule overrides, NO reward manipulation

Usage:
  python scripts/final_model_test.py --model models/baseline_standard.h5
  python scripts/final_model_test.py --model models/distributed_cpu_best.h5
"""

import os
import sys
import json
import argparse
import numpy as np

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# Must import highway_env to register 'roundabout-v0' in gymnasium
import gymnasium as gym
import highway_env

CLASS_NAMES = ["0: LANE_LEFT", "1: IDLE", "2: LANE_RIGHT", "3: FASTER", "4: SLOWER"]
SHORT_NAMES = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]


def run_single_inference_test(model):
    """Verify single observation inference output shape, bounds, and Softmax sum."""
    dummy_obs = np.random.randn(1, 25).astype(np.float32)
    obs_tensor = tf.constant(dummy_obs)
    probs = model(obs_tensor, training=False).numpy()

    assert probs.shape == (1, 5), f"Expected shape (1, 5), got {probs.shape}"
    assert np.all(np.isfinite(probs)), "Inference output contains non-finite values (NaN/Inf)!"
    prob_sum = float(np.sum(probs))
    assert np.isclose(prob_sum, 1.0, atol=1e-4), f"Softmax probabilities do not sum to 1: {prob_sum}"
    action = int(np.argmax(probs[0]))
    assert 0 <= action <= 4, f"Predicted action {action} out of bounds [0, 4]!"

    return {
        "input_shape": list(dummy_obs.shape),
        "output_shape": list(probs.shape),
        "prob_sum": prob_sum,
        "sample_probs": probs[0].tolist(),
        "sample_action": action,
        "status": "PASSED"
    }


def evaluate_model(model_path, num_rounds=10, episodes_per_round=10, seed_base=42):
    """
    Evaluate model across num_rounds x episodes_per_round in roundabout-v0.
    Faithful to teacher's NN_test.py logic while collecting rigorous telemetry.
    """
    print("=" * 70)
    print(f" EVALUATING MODEL: {model_path}")
    print("=" * 70)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 1. Load Model
    print("[1] Loading Keras Model...")
    model = tf.keras.models.load_model(model_path, compile=False)
    print("    -> Model loaded successfully.")

    # 2. Verify Single Observation Inference
    print("[2] Running Single Observation Inference Sanity Check...")
    inf_test = run_single_inference_test(model)
    print(f"    -> Input Shape:     {inf_test['input_shape']}")
    print(f"    -> Output Shape:    {inf_test['output_shape']}")
    print(f"    -> Probabilities:   {[round(p, 4) for p in inf_test['sample_probs']]}")
    print(f"    -> Softmax Sum:     {inf_test['prob_sum']:.4f}")
    print(f"    -> Predicted Action: {inf_test['sample_action']} ({SHORT_NAMES[inf_test['sample_action']]})")

    # 3. Initialize Environment
    print("\n[3] Initializing roundabout-v0...")
    env = gym.make("roundabout-v0", render_mode="rgb_array")

    round_rewards_float = []
    round_rewards_int = []  # Teacher casts to int: int(total_reward)
    all_episode_rewards = []
    all_episode_lengths = []
    collisions = 0
    arrivals = 0
    timeouts = 0
    action_counts = {c: 0 for c in range(5)}

    print(f"\n[4] Executing {num_rounds} Rounds x {episodes_per_round} Episodes (Total: {num_rounds * episodes_per_round} Episodes)...")

    for r in range(num_rounds):
        round_total_reward = 0.0
        print(f"\n--- Round {r + 1}/{num_rounds} ---")

        for ep in range(episodes_per_round):
            # Seed strategy: deterministic paired seed across models
            ep_seed = seed_base + r * 100 + ep
            obs, info = env.reset(seed=ep_seed)
            done = False
            truncated = False
            ep_reward = 0.0
            ep_steps = 0

            while not (done or truncated):
                # Reshape to (1, 25) matching teacher's NN_test.py line 27
                obs_reshaped = obs.reshape(1, 25).astype(np.float32)
                obs_tensor = tf.constant(obs_reshaped)

                # Direct forward inference (strictly NO predict / evaluate)
                probs = model(obs_tensor, training=False).numpy()
                action = int(np.argmax(probs[0]))
                action_counts[action] += 1

                # Step environment
                obs, reward, done, truncated, info = env.step(action)
                ep_reward += float(reward)
                ep_steps += 1

            round_total_reward += ep_reward
            all_episode_rewards.append(ep_reward)
            all_episode_lengths.append(ep_steps)

            # Terminal state analysis
            if info.get("crashed", False):
                collisions += 1
            elif info.get("arrived", False):
                arrivals += 1
            else:
                timeouts += 1

            print(f"  Round {r + 1} | Ep {ep + 1:02d} (Seed: {ep_seed:04d}): Steps={ep_steps:02d}, Reward={ep_reward:6.2f}, Crashed={info.get('crashed', False)}")

        round_rewards_float.append(round_total_reward)
        round_rewards_int.append(int(round_total_reward))
        print(f"  >> Round {r + 1} Total Reward: {round_total_reward:.2f} (Teacher Integer Format: {int(round_total_reward)})")

    env.close()
    print("\n[5] Environment closed cleanly.")

    # 4. Compute Aggregate Statistics
    total_episodes = num_rounds * episodes_per_round
    mean_round_reward = float(np.mean(round_rewards_float))
    std_round_reward = float(np.std(round_rewards_float))
    mean_ep_reward = float(np.mean(all_episode_rewards))
    std_ep_reward = float(np.std(all_episode_rewards))
    mean_ep_steps = float(np.mean(all_episode_lengths))

    collision_rate = collisions / total_episodes
    arrival_rate = arrivals / total_episodes
    timeout_rate = timeouts / total_episodes

    action_distribution = {
        CLASS_NAMES[c]: round(count / sum(action_counts.values()) * 100, 2)
        for c, count in action_counts.items()
    }

    summary = {
        "model_path": model_path,
        "inference_test": inf_test,
        "num_rounds": num_rounds,
        "episodes_per_round": episodes_per_round,
        "total_episodes": total_episodes,
        "teacher_round_rewards_int": round_rewards_int,
        "round_rewards_float": [round(r, 2) for r in round_rewards_float],
        "mean_round_reward": round(mean_round_reward, 2),
        "std_round_reward": round(std_round_reward, 2),
        "mean_episode_reward": round(mean_ep_reward, 2),
        "std_episode_reward": round(std_ep_reward, 2),
        "mean_episode_steps": round(mean_ep_steps, 1),
        "collision_count": collisions,
        "collision_rate": round(collision_rate * 100, 2),
        "arrival_count": arrivals,
        "arrival_rate": round(arrival_rate * 100, 2),
        "timeout_count": timeouts,
        "timeout_rate": round(timeout_rate * 100, 2),
        "action_distribution": action_distribution,
        "status": "COMPLETED_SUCCESSFULLY"
    }

    print("\n" + "=" * 70)
    print(" EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Teacher 10-Round Integer Rewards: {round_rewards_int}")
    print(f"Mean Round Reward:                {mean_round_reward:.2f} +/- {std_round_reward:.2f}")
    print(f"Mean Episode Reward:              {mean_ep_reward:.2f} +/- {std_ep_reward:.2f}")
    print(f"Mean Episode Steps:               {mean_ep_steps:.1f}")
    print(f"Collision Rate:                   {collision_rate * 100:.1f}% ({collisions}/{total_episodes})")
    print(f"Arrival Rate:                     {arrival_rate * 100:.1f}% ({arrivals}/{total_episodes})")
    print(f"Action Distribution:              {action_distribution}")
    print("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Final Model Test in roundabout-v0")
    parser.add_argument("--model", type=str, default="/app/models/baseline_standard.h5", help="Path to .h5 model")
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds (default 10)")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per round (default 10)")
    parser.add_argument("--seed_base", type=int, default=42, help="Base random seed for reproduction")
    parser.add_argument("--output_json", type=str, default=None, help="Save evaluation summary to JSON")
    args = parser.parse_args()

    results = evaluate_model(
        model_path=args.model,
        num_rounds=args.rounds,
        episodes_per_round=args.episodes,
        seed_base=args.seed_base
    )

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved evaluation output to: {args.output_json}")


if __name__ == "__main__":
    main()

