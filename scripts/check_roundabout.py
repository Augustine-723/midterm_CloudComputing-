#!/usr/bin/env python3
"""
scripts/check_roundabout.py
Inspect the roundabout-v0 environment and its compatibility with the project requirements.
"""

import importlib.metadata
import json
import os
import pprint
import sys
import numpy as np

# Suppress pygame and headless warnings
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gymnasium as gym
import highway_env
import tensorflow as tf


def format_value(val):
    if isinstance(val, np.ndarray):
        return f"array(shape={val.shape}, dtype={val.dtype})"
    return str(val)


def main():
    print("=" * 70)
    print(" ROUNDABOUT-V0 ENVIRONMENT COMPATIBILITY CHECK")
    print("=" * 70)

    # 1. Package Versions
    print("\n[1] Installed Package Versions:")
    py_ver = sys.version.split()[0]
    tf_ver = tf.__version__
    gym_ver = gym.__version__
    np_ver = np.__version__
    try:
        hw_ver = importlib.metadata.version("highway-env")
    except Exception:
        hw_ver = "unknown"

    print(f"  - Python:      {py_ver}")
    print(f"  - TensorFlow:  {tf_ver}")
    print(f"  - Gymnasium:   {gym_ver}")
    print(f"  - highway-env: {hw_ver}")
    print(f"  - NumPy:       {np_ver}")

    # 2. Make Environment
    print("\n[2] Instantiating 'roundabout-v0'...")
    env = gym.make("roundabout-v0", render_mode="rgb_array")
    unwrapped = env.unwrapped

    # 3. Observation Space & Action Space
    print("\n[3] Spaces:")
    print(f"  - Observation Space: {env.observation_space}")
    print(f"    * Shape: {env.observation_space.shape}")
    print(f"    * Dtype: {env.observation_space.dtype}")
    print(f"    * Low:\n{env.observation_space.low}")
    print(f"    * High:\n{env.observation_space.high}")

    print(f"\n  - Action Space: {env.action_space}")
    is_discrete_5 = isinstance(env.action_space, gym.spaces.Discrete) and env.action_space.n == 5
    print(f"    * Type: {type(env.action_space).__name__}")
    print(f"    * Number of actions (n): {getattr(env.action_space, 'n', 'N/A')}")
    print(f"    * Is Discrete(5): {is_discrete_5}")

    # 4. Observation Configuration
    print("\n[4] Environment Configuration:")
    obs_config = unwrapped.config.get("observation", {})
    action_config = unwrapped.config.get("action", {})
    print("  - Observation Config:")
    pprint.pprint(obs_config, indent=4)
    print("\n  - Action Config:")
    pprint.pprint(action_config, indent=4)

    # 5. Reset Environment
    print("\n[5] Resetting Environment:")
    reset_ret = env.reset(seed=42)
    if isinstance(reset_ret, tuple) and len(reset_ret) == 2:
        obs, info = reset_ret
        print("  - env.reset() returned (obs, info) tuple.")
    else:
        obs = reset_ret
        info = {}
        print("  - env.reset() returned single object.")

    print(f"  - Raw Observation Shape: {obs.shape}")
    print(f"  - Raw Observation Dtype: {obs.dtype}")
    print(f"  - Raw Observation Sample:\n{obs}")
    obs_flat = obs.reshape(-1)
    print(f"  - Flattened Shape: {obs_flat.shape}")
    print(f"  - Number of Features: {obs_flat.size}")
    print(f"  - Is Flattened 25 Features: {obs_flat.size == 25}")

    # 6. Step with Legal Action
    print("\n[6] Taking One Simulation Step:")
    action = 1  # IDLE
    step_ret = env.step(action)
    print(f"  - Action taken: {action} (Discrete)")
    if len(step_ret) == 5:
        next_obs, reward, terminated, truncated, step_info = step_ret
        print("  - Returned 5-tuple: (observation, reward, terminated, truncated, info)")
        print(f"    * next_obs shape: {next_obs.shape}")
        print(f"    * reward: {reward} (type: {type(reward).__name__})")
        print(f"    * terminated: {terminated} (type: {type(terminated).__name__})")
        print(f"    * truncated: {truncated} (type: {type(truncated).__name__})")
        print(f"    * step_info keys: {list(step_info.keys())}")
    else:
        print(f"  - Unexpected return length from step: {len(step_ret)}")

    # 7. Render Test
    print("\n[7] Render Test:")
    frame = env.render()
    if frame is not None:
        print(f"  - Render frame shape: {frame.shape}, dtype: {frame.dtype}")
    else:
        print("  - Render frame: None")

    env.close()
    print("\n" + "=" * 70)
    print(" ROUNDABOUT-V0 COMPATIBILITY CHECK COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()

