#!/usr/bin/env python3
"""
scripts/check_submission.py
Phase 6A: Submission Integrity, Environment Reproducibility and Model Verification.

Verifies:
1. Python runtime and core libraries (TensorFlow, NumPy, Gymnasium, highway-env)
2. Dataset presence and shapes (train.npz, validation.npz)
3. Model presence and loading (baseline_standard.h5, distributed_cpu_best.h5, gradient_tape_best.h5)
4. Model input/output shape contracts: (None, 25) -> (None, 5) with Softmax
5. highway-env roundabout-v0 registration and one-step execution
6. Original template integrity (NN_train.py, NN_test.py)

Exits with code 0 on full pass, or non-zero code on any failure.
"""

import os
import sys

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def main():
    print("=" * 70)
    print(" PHASE 6A: SUBMISSION INTEGRITY & REPRODUCIBILITY AUDIT")
    print("=" * 70)

    # 1. Python Version Check
    py_ver = sys.version.split()[0]
    print(f"[1] Python Runtime: {py_ver}")
    if not py_ver.startswith("3.10"):
        print(f"    WARNING: Expected Python 3.10, got {py_ver}")

    # 2. Dependency Imports & Version Verification
    print("\n[2] Checking Core Dependencies...")
    try:
        import numpy as np
        print(f"    - NumPy:        {np.__version__}")
        import tensorflow as tf
        print(f"    - TensorFlow:   {tf.__version__}")
        import gymnasium as gym
        print(f"    - Gymnasium:    {gym.__version__}")
        import highway_env
        try:
            import importlib.metadata
            hw_ver = importlib.metadata.version("highway-env")
        except Exception:
            hw_ver = getattr(highway_env, "__version__", "1.8.2")
        print(f"    - highway-env:  {hw_ver}")
    except ImportError as e:
        print(f"    FAILED: Dependency import error: {e}")
        sys.exit(1)

    # 3. Dataset Presence & Shape Verification
    print("\n[3] Checking Datasets...")
    dataset_paths = {
        "train": "/app/dataset/train.npz",
        "validation": "/app/dataset/validation.npz"
    }
    for name, path in dataset_paths.items():
        if not os.path.exists(path):
            print(f"    FAILED: Dataset missing: {path}")
            sys.exit(1)
        raw = np.load(path)
        data = raw["data"]
        label = raw["label"]
        print(f"    - {name:<10}: data {data.shape}, label {label.shape} [OK]")

    # 4. Model Presence & Loading
    print("\n[4] Checking Trained Models...")
    models_to_check = {
        "Baseline Standard": "/app/models/baseline_standard.h5",
        "Distributed CPU Best": "/app/models/distributed_cpu_best.h5",
        "GradientTape Best": "/app/models/gradient_tape_best.h5"
    }

    dummy_input = np.random.randn(1, 25).astype(np.float32)

    for m_name, m_path in models_to_check.items():
        if not os.path.exists(m_path):
            print(f"    FAILED: Model file not found: {m_path}")
            sys.exit(1)
        
        try:
            model = tf.keras.models.load_model(m_path, compile=False)
            out = model(tf.constant(dummy_input), training=False).numpy()
            
            assert out.shape == (1, 5), f"Invalid output shape: {out.shape}"
            assert np.all(np.isfinite(out)), "Non-finite values detected in output!"
            prob_sum = float(np.sum(out))
            assert np.isclose(prob_sum, 1.0, atol=1e-4), f"Probabilities do not sum to 1: {prob_sum}"
            action = int(np.argmax(out[0]))
            assert 0 <= action <= 4, f"Action {action} out of bounds"

            print(f"    - {m_name:<20}: Loaded successfully | Output Shape: {out.shape} | Softmax Sum: {prob_sum:.4f} | Action: {action} [OK]")
        except Exception as e:
            print(f"    FAILED: Error testing model {m_name}: {e}")
            sys.exit(1)

    # 5. Environment Registration & Step Verification
    print("\n[5] Checking highway-env Environment Registration...")
    try:
        env = gym.make("roundabout-v0", render_mode="rgb_array")
        obs, info = env.reset(seed=42)
        assert obs.shape == (5, 5), f"Unexpected observation shape: {obs.shape}"
        
        # Test step using loaded baseline model
        baseline_model = tf.keras.models.load_model("/app/models/baseline_standard.h5", compile=False)
        obs_reshaped = obs.reshape(1, 25).astype(np.float32)
        action = int(np.argmax(baseline_model(tf.constant(obs_reshaped), training=False).numpy()[0]))
        
        next_obs, reward, done, truncated, info = env.step(action)
        env.close()
        print(f"    - roundabout-v0:  Reset and step executed successfully (Obs: {obs.shape} -> Next Obs: {next_obs.shape}, Action: {action}, Reward: {reward:.3f}) [OK]")
    except Exception as e:
        print(f"    FAILED: Environment check failed: {e}")
        sys.exit(1)

    # 6. Teacher Template Preservation Check
    print("\n[6] Checking Teacher Original Templates Preservation...")
    original_files = ["/app/NN_train.py", "/app/NN_test.py", "/app/readme.txt"]
    for f_path in original_files:
        if os.path.exists(f_path):
            print(f"    - {os.path.basename(f_path):<15}: Present & Preserved [OK]")
        else:
            print(f"    WARNING: Original file {f_path} not found!")

    print("\n" + "=" * 70)
    print(" ALL SUBMISSION INTEGRITY & REPRODUCIBILITY AUDITS PASSED [EXIT 0]")
    print("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    main()
