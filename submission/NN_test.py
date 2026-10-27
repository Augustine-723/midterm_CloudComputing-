#!/usr/bin/env python3
"""
submission/NN_test.py
Teacher-Compatible Environment Testing Script for Highway-env Roundabout.

Implements the teacher's exact evaluation protocol:
- Loads YOURMODEL.h5
- Runs 10 rounds, each containing 10 episodes in roundabout-v0 (100 episodes total)
- Reshapes observation to (1, 25)
- Predicts discrete action via argmax
- Steps environment without any action masking or rule overrides
- Sums rewards per round, truncated to int: int(total_reward)
- Closes environment cleanly
- Prints the list of 10 round rewards: [r1, r2, ..., r10]
"""

import os
import sys
import argparse
import numpy as np

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras

# Register highway-env environments in gymnasium
import gymnasium as gym
import highway_env

root_path = os.path.abspath(os.path.dirname(__file__))
DEFAULT_MODEL_PATH = os.path.join(root_path, "YOURMODEL.h5")


def main(model=None, render=False):
    """Run one round of 10 episodes in roundabout-v0 and return integer total reward."""
    if model is None:
        model = keras.models.load_model(DEFAULT_MODEL_PATH, compile=False)

    total_reward = 0.0
    env = gym.make("roundabout-v0", render_mode="rgb_array")

    for _ in range(10):
        (obs, info) = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            if render:
                env.render()
            obs = obs.reshape(1, 25).astype(np.float32)

            # Predict action (direct tensor call)
            probs = model(obs, training=False).numpy()
            action = int(np.argmax(probs[0]))

            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)

    env.close()
    return int(total_reward)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Roundabout Model in Environment")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to .h5 model")
    parser.add_argument("--render", action="store_true", help="Render RGB array")
    args = parser.parse_args()

    # Load model once for evaluation
    loaded_model = keras.models.load_model(args.model, compile=False)

    rewards = []
    for round_idx in range(10):
        reward = main(model=loaded_model, render=args.render)
        rewards.append(reward)

    print(rewards)

