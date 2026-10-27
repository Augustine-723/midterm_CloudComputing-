#!/usr/bin/env python3
"""
Environment verification script for Cloud Computing Midterm Project (midterm_roundabout).
Checks:
- Python version (>= 3.10)
- TensorFlow (== 2.13.0)
- NumPy (== 1.24.3)
- Gymnasium (compatible version, e.g. 0.29.1)
- highway-env (== 1.8.2) and roundabout-v0 environment creation/reset/step/render
"""

import os
import sys

# Suppress Pygame welcome message and set headless video driver for clean verification
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    print("=" * 60)
    print(" Cloud Computing Midterm - Environment Diagnostic Check ")
    print("=" * 60)

    checks_passed = True

    # 1. Python Check
    py_version = sys.version.split()[0]
    print(f"\n[1/5] Python Version: {py_version}")
    if sys.version_info >= (3, 10):
        print("      Status: [PASS] Python >= 3.10 detected.")
    else:
        print("      Status: [FAIL] Requires Python >= 3.10.")
        checks_passed = False

    # 2. NumPy Check
    try:
        import numpy as np

        np_version = np.__version__
        print(f"\n[2/5] NumPy Version: {np_version}")
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.sum() == 15
        print(f"      Status: [PASS] NumPy is working properly.")
        if np_version != "1.24.3":
            print(f"      Notice: Recommended version is 1.24.3 (current: {np_version}).")
    except Exception as e:
        print(f"\n[2/5] NumPy Check FAILED: {e}")
        checks_passed = False

    # 3. TensorFlow Check
    try:
        import tensorflow as tf

        tf_version = tf.__version__
        print(f"\n[3/5] TensorFlow Version: {tf_version}")
        t = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        t_sum = float(tf.reduce_sum(t))
        assert t_sum == 10.0
        gpus = tf.config.list_physical_devices("GPU")
        device_msg = f"{len(gpus)} GPU(s) available" if gpus else "CPU mode (No GPU detected)"
        print(f"      Device: {device_msg}")
        print("      Status: [PASS] TensorFlow basic tensor operations succeeded.")
        if tf_version != "2.13.0":
            print(f"      Notice: Recommended version is 2.13.0 (current: {tf_version}).")
    except Exception as e:
        print(f"\n[3/5] TensorFlow Check FAILED: {e}")
        checks_passed = False

    # 4. Gymnasium Check
    try:
        import gymnasium as gym

        gym_version = gym.__version__
        print(f"\n[4/5] Gymnasium Version: {gym_version}")
        print("      Status: [PASS] Gymnasium imported successfully.")
    except Exception as e:
        print(f"\n[4/5] Gymnasium Check FAILED: {e}")
        checks_passed = False

    # 5. highway-env & roundabout-v0 Check
    try:
        import importlib.metadata
        try:
            hw_version = importlib.metadata.version("highway-env")
        except Exception:
            hw_version = "installed"

        print(f"\n[5/5] highway-env Version: {hw_version}")
        import highway_env  # noqa: F401

        print("      Initializing gymnasium environment: 'roundabout-v0'...")
        env = gym.make("roundabout-v0", render_mode="rgb_array")
        obs, info = env.reset()
        print(f"      Initial Observation shape: {obs.shape} (expected 5x5)")

        # Test single simulation step
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"      Simulation step passed (reward: {reward:.4f}, done: {done})")

        # Test render
        frame = env.render()
        if frame is not None:
            print(f"      Rendering passed (frame shape: {frame.shape})")

        env.close()
        print("      Status: [PASS] highway-env and roundabout-v0 are working normally.")
    except Exception as e:
        print(f"\n[5/5] highway-env / roundabout-v0 Check FAILED: {e}")
        checks_passed = False

    print("\n" + "=" * 60)
    if checks_passed:
        print(" [ALL CHECKS PASSED] Environment is fully ready for the project!")
        print("=" * 60)
        return 0
    else:
        print(" [WARNING] Some checks failed. Please inspect the logs above.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

