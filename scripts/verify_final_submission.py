#!/usr/bin/env python3
"""
scripts/verify_final_submission.py
Phase 6C: Comprehensive Final Verification of the submission/ Directory.

Performs rigorous checks:
1. File existence in submission/
2. Cryptographic SHA256 verification of YOURMODEL.h5
3. Input/Output tensor contract validation: (None, 25) -> (None, 5)
4. Exact validation accuracy reproduction (94.64%)
5. Standalone execution of submission/NN_test.py from within submission/
6. Verification of 10-round integer rewards output
7. Code hygiene: scanning for developer-specific absolute paths
8. Clean exit code 0 on full verification pass.
"""

import os
import sys
import hashlib
import subprocess
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "submission")
EXPECTED_SHA256 = "ccc4b358d95e9b56805ef8244eb7247d60f9a5d7bc5e5340c197652fef837e02"

REQUIRED_FILES = [
    "YOURMODEL.h5",
    "NN_train.py",
    "NN_test.py",
    "train_gradient_tape.py",
    "train_distributed.py",
    "models/distributed_cpu_best.h5",
    "models/gradient_tape_best.h5",
    "README.md",
    "RESULTS.md"
]


def check_file_existence():
    print("[1] Verifying File Existence in submission/...")
    missing = []
    for rel_path in REQUIRED_FILES:
        full_path = os.path.join(SUBMISSION_DIR, rel_path)
        if os.path.exists(full_path):
            size = os.path.getsize(full_path)
            print(f"    - {rel_path:<35}: Present ({size:,} bytes) [OK]")
        else:
            print(f"    - {rel_path:<35}: MISSING [FAILED]")
            missing.append(rel_path)
    if missing:
        raise FileNotFoundError(f"Missing required submission files: {missing}")


def check_model_integrity():
    print("\n[2] Verifying Primary Model (YOURMODEL.h5) Integrity...")
    model_path = os.path.join(SUBMISSION_DIR, "YOURMODEL.h5")
    
    with open(model_path, "rb") as f:
        actual_sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"    - Actual SHA256:   {actual_sha256}")
    print(f"    - Expected SHA256: {EXPECTED_SHA256}")
    assert actual_sha256 == EXPECTED_SHA256, f"SHA256 mismatch! Got {actual_sha256}"
    print("    -> Cryptographic SHA256 Match Verified [OK]")

    model = tf.keras.models.load_model(model_path, compile=False)
    in_shape = model.input_shape
    out_shape = model.output_shape
    print(f"    - Input Shape:     {in_shape} (Expected: (None, 25))")
    print(f"    - Output Shape:    {out_shape} (Expected: (None, 5))")
    assert in_shape == (None, 25), f"Invalid input shape: {in_shape}"
    assert out_shape == (None, 5), f"Invalid output shape: {out_shape}"

    # Verify accuracy on validation set
    val_path = os.path.join(PROJECT_ROOT, "dataset", "validation.npz")
    val_raw = np.load(val_path)
    x_val = val_raw["data"].astype(np.float32).reshape(-1, 25)
    y_val = val_raw["label"].astype(np.int32).flatten()

    probs = model(tf.constant(x_val), training=False).numpy()
    preds = np.argmax(probs, axis=1)
    acc = float(np.mean(preds == y_val) * 100.0)
    errs = int(np.sum(preds != y_val))
    print(f"    - Validation Acc:  {acc:.2f}% ({len(y_val)-errs}/{len(y_val)} correct, {errs} errors)")
    assert np.isclose(acc, 94.64, atol=0.01), f"Unexpected accuracy: {acc}"
    print("    -> Validation Accuracy 94.64% Verified [OK]")


def check_code_hygiene():
    print("\n[3] Checking Code Hygiene for Developer Hardcoded Paths...")
    python_files = [f for f in REQUIRED_FILES if f.endswith(".py")]
    disallowed_strings = ["/home/augustine", "C:\\", "Users/"]
    
    for rel_path in python_files:
        full_path = os.path.join(SUBMISSION_DIR, rel_path)
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        for s in disallowed_strings:
            if s in content:
                raise ValueError(f"Found forbidden hardcoded path '{s}' in {rel_path}!")
        print(f"    - {rel_path:<25}: Clean (No absolute developer paths) [OK]")


def check_nn_test_execution():
    print("\n[4] Executing submission/NN_test.py from within submission/...")
    # Run from within submission directory to ensure relative path resolution works
    cmd = [sys.executable, "NN_test.py"]
    res = subprocess.run(
        cmd,
        cwd=SUBMISSION_DIR,
        capture_output=True,
        text=True
    )
    print(f"    - Exit Code: {res.returncode}")
    if res.returncode != 0:
        print(f"    - STDERR: {res.stderr}")
        raise RuntimeError(f"submission/NN_test.py failed with exit code {res.returncode}")

    stdout_clean = res.stdout.strip().splitlines()[-1]
    print(f"    - Captured Output: {stdout_clean}")

    import ast
    rewards_list = ast.literal_eval(stdout_clean)
    assert isinstance(rewards_list, list), f"Expected list output, got {type(rewards_list)}"
    assert len(rewards_list) == 10, f"Expected 10 rounds, got {len(rewards_list)}"
    for r in rewards_list:
        assert isinstance(r, int), f"Expected integer reward, got {type(r)}"
    print(f"    -> Successfully verified 10 integer rewards: {rewards_list} [OK]")


def main():
    print("=" * 70)
    print(" PHASE 6C: FINAL SUBMISSION REPRODUCIBILITY AUDIT")
    print("=" * 70)

    try:
        check_file_existence()
        check_model_integrity()
        check_code_hygiene()
        check_nn_test_execution()
    except Exception as e:
        print(f"\n[AUDIT FAILED]: {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(" ALL FINAL SUBMISSION VERIFICATION CHECKS PASSED [EXIT 0]")
    print("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    main()

