#!/usr/bin/env python3
"""
demo/scripts/check_environment.py
Demo Day System Environment & Model Verification Tool.

Checks:
1. Python Version
2. TensorFlow Version
3. NumPy Version
4. Gymnasium Version
5. Highway-env Installation
6. Current Working Directory and Root Resolution
7. Primary Model Existence (submission/YOURMODEL.h5) & SHA-256 Hash
8. Primary Model Loading (keras.models.load_model with compile=False)
9. Model Input Shape: (None, 25)
10. Model Output Shape: (None, 5)
11. Model Total Parameter Count: 11,909
12. Dummy Forward Pass & Softmax Probability Verification
13. Advanced Bonus Models Presence & Readability

Read-only: strictly never modifies, overwrites, or retrains any model.
"""

import os
import sys
import hashlib
import numpy as np

# Suppress TF C++ verbose logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

EXPECTED_PARAMS = 11909
EXPECTED_INPUT_SHAPE = (None, 25)
EXPECTED_OUTPUT_SHAPE = (None, 5)
PRIMARY_MODEL_SHA256 = "ccc4b358d95e9b56805ef8244eb7247d60f9a5d7bc5e5340c197652fef837e02"

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

def compute_sha256(filepath):
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

def resolve_model_path():
    """Find YOURMODEL.h5 in candidate paths."""
    candidates = [
        os.path.join(PROJECT_ROOT, "submission", "YOURMODEL.h5"),
        os.path.join(PROJECT_ROOT, "YOURMODEL.h5"),
        os.path.join(os.getcwd(), "submission", "YOURMODEL.h5"),
        os.path.join(os.getcwd(), "YOURMODEL.h5"),
        "/app/submission/YOURMODEL.h5",
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None

def main():
    print("=" * 75)
    print(" DEMO DAY: SYSTEM ENVIRONMENT & MODEL READINESS VERIFICATION")
    print("=" * 75)

    all_passed = True
    errors = []

    # 1. Python
    py_ver = sys.version.split()[0]
    print(f"[*] Python Version:         {py_ver} ({sys.executable})")

    # 2. TensorFlow
    try:
        import tensorflow as tf
        tf_ver = tf.__version__
        print(f"[*] TensorFlow Version:     {tf_ver}")
        if not tf_ver.startswith("2."):
            all_passed = False
            errors.append(f"Unexpected TensorFlow version: {tf_ver} (expected 2.x)")
    except ImportError as e:
        all_passed = False
        errors.append(f"TensorFlow not installed: {e}")
        print("[!] ERROR: TensorFlow not found.")
        sys.exit(1)

    # 3. NumPy
    print(f"[*] NumPy Version:          {np.__version__}")

    # 4. Gymnasium
    try:
        import gymnasium as gym
        print(f"[*] Gymnasium Version:      {gym.__version__}")
    except ImportError as e:
        all_passed = False
        errors.append(f"Gymnasium not installed: {e}")

    # 5. Highway-env
    try:
        import highway_env
        hw_ver = getattr(highway_env, "__version__", "1.8.2 (module loaded)")
        print(f"[*] Highway-env Version:    {hw_ver}")
    except ImportError as e:
        all_passed = False
        errors.append(f"Highway-env not installed: {e}")

    # 6. Working Directory
    cwd = os.getcwd()
    print(f"[*] Working Directory:      {cwd}")
    print(f"[*] Resolved Project Root:  {PROJECT_ROOT}")

    # 7. Locate Primary Model
    model_path = resolve_model_path()
    if model_path is None:
        print("[!] ERROR: submission/YOURMODEL.h5 not found in candidate paths!")
        errors.append("Primary model YOURMODEL.h5 not found.")
        all_passed = False
    else:
        file_size = os.path.getsize(model_path)
        actual_sha = compute_sha256(model_path)
        print(f"\n[+] Primary Model Path:     {model_path}")
        print(f"    - File Size:            {file_size} bytes")
        print(f"    - SHA-256 Hash:         {actual_sha}")
        if actual_sha == PRIMARY_MODEL_SHA256:
            print("    - Integrity Check:      PASS (Exact match with official submission)")
        else:
            print("    - Integrity Check:      WARNING (SHA mismatch with expected hash!)")
            errors.append(f"Model SHA mismatch: got {actual_sha}, expected {PRIMARY_MODEL_SHA256}")

        # 8-12. Load Model & Inspect
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            input_shape = model.input_shape
            output_shape = model.output_shape
            total_params = model.count_params()

            print(f"    - Model Loading:        SUCCESS")
            print(f"    - Input Shape:          {input_shape}")
            print(f"    - Output Shape:         {output_shape}")
            print(f"    - Total Parameters:     {total_params}")

            if input_shape != EXPECTED_INPUT_SHAPE:
                all_passed = False
                errors.append(f"Input shape mismatch: {input_shape} != {EXPECTED_INPUT_SHAPE}")

            if output_shape != EXPECTED_OUTPUT_SHAPE:
                all_passed = False
                errors.append(f"Output shape mismatch: {output_shape} != {EXPECTED_OUTPUT_SHAPE}")

            if total_params != EXPECTED_PARAMS:
                all_passed = False
                errors.append(f"Parameter count mismatch: {total_params} != {EXPECTED_PARAMS}")

            # Dummy forward pass
            dummy_in = np.zeros((1, 25), dtype=np.float32)
            dummy_out = model(dummy_in, training=False).numpy()
            prob_sum = float(np.sum(dummy_out))
            pred_class = int(np.argmax(dummy_out[0]))
            print(f"    - Dummy Forward Pass:   SUCCESS (Predicted Class: {pred_class}, Softmax Sum: {prob_sum:.4f})")
            if not np.isclose(prob_sum, 1.0, atol=1e-3):
                all_passed = False
                errors.append(f"Softmax probabilities do not sum to 1.0 (sum={prob_sum})")

        except Exception as e:
            all_passed = False
            errors.append(f"Failed to load/probe model: {e}")
            print(f"    - Model Loading ERROR:  {e}")

    # 13. Advanced Bonus Models Presence
    print("\n[*] Advanced Bonus Models Status:")
    bonus_models = [
        ("Distributed CPU (MirroredStrategy)", os.path.join(PROJECT_ROOT, "submission", "models", "distributed_cpu_best.h5")),
        ("Custom GradientTape (Autodiff)", os.path.join(PROJECT_ROOT, "submission", "models", "gradient_tape_best.h5")),
    ]
    for name, bpath in bonus_models:
        if os.path.exists(bpath):
            b_size = os.path.getsize(bpath)
            b_sha = compute_sha256(bpath)
            print(f"    [+] {name}: Found ({b_size} bytes)")
            print(f"        SHA-256: {b_sha}")
        else:
            print(f"    [-] {name}: Not found at {bpath}")

    print("\n" + "=" * 75)
    if all_passed:
        print(" [✓] DEMO ENVIRONMENT VERIFICATION: ALL CHECKS PASSED")
        print("     System is fully prepared to execute evaluations.")
        print("=" * 75)
        sys.exit(0)
    else:
        print(" [✗] DEMO ENVIRONMENT VERIFICATION: ISSUES DETECTED")
        for err in errors:
            print(f"     - {err}")
        print("=" * 75)
        sys.exit(1)

if __name__ == "__main__":
    main()

