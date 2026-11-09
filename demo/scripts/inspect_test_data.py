#!/usr/bin/env python3
"""
demo/scripts/inspect_test_data.py
Read-only Diagnostic Tool for Newly Received Official Test Datasets.

Features:
- Inspects any .npz file (e.g., test.npz) placed in demo/incoming/ or specified by path
- Displays: Array names, shapes, dtypes, sample counts, NaN/Inf counts, numerical ranges
- Checks compatibility with YOURMODEL.h5 input requirement (25-dim kinematics)
- Read-only: strictly never modifies, rewrites, or overwrites the input test file
- Privacy-safe: never dumps full test labels or raw sample values
"""

import os
import sys
import argparse
import hashlib
import numpy as np

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
INCOMING_DIR = os.path.join(PROJECT_ROOT, "demo", "incoming")

def compute_sha256(filepath):
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

def find_candidate_npz():
    """Auto-detect .npz files in demo/incoming/ or project root."""
    if os.path.exists(INCOMING_DIR):
        npz_files = [
            os.path.join(INCOMING_DIR, f)
            for f in os.listdir(INCOMING_DIR)
            if f.endswith(".npz")
        ]
        if npz_files:
            return npz_files[0]
    return None

def main():
    parser = argparse.ArgumentParser(description="Read-only inspection of test dataset NPZ files.")
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Path to .npz test dataset (default: auto-detect in demo/incoming/)"
    )
    args = parser.parse_args()

    target_file = args.file
    if target_file is None:
        target_file = find_candidate_npz()

    print("=" * 75)
    print(" DEMO DAY: OFFICIAL TEST DATASET READ-ONLY INSPECTOR")
    print("=" * 75)

    if target_file is None or not os.path.exists(target_file):
        print("[!] No test dataset found to inspect.")
        print(f"    Expected location: {INCOMING_DIR}/<filename>.npz")
        print("\n[ACTION REQUIRED]:")
        print("    1. When the TA provides the official test dataset (e.g., test.npz),")
        print(f"       copy or move it into: {INCOMING_DIR}/")
        print("    2. Re-run this inspector:")
        print("       python demo/scripts/inspect_test_data.py")
        print("       OR:")
        print("       python demo/scripts/inspect_test_data.py /path/to/test.npz")
        print("=" * 75)
        sys.exit(0)

    target_file = os.path.abspath(target_file)
    file_size = os.path.getsize(target_file)
    file_sha = compute_sha256(target_file)

    print(f"[+] Target File:        {target_file}")
    print(f"    - File Size:        {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
    print(f"    - SHA-256 Hash:     {file_sha}")

    try:
        data = np.load(target_file)
    except Exception as e:
        print(f"[!] ERROR: Failed to read {target_file} as NPZ archive: {e}")
        sys.exit(1)

    keys = list(data.files)
    print(f"\n[+] Arrays Found ({len(keys)} total): {keys}")

    feature_candidate = None
    label_candidate = None

    for key in keys:
        arr = data[key]
        shape = arr.shape
        dtype = arr.dtype
        n_samples = shape[0] if len(shape) > 0 else 0

        # NaN / Inf check
        has_nan = bool(np.isnan(arr).any()) if np.issubdtype(dtype, np.number) else False
        has_inf = bool(np.isinf(arr).any()) if np.issubdtype(dtype, np.floating) else False
        nan_count = int(np.isnan(arr).sum()) if has_nan else 0
        inf_count = int(np.isinf(arr).sum()) if has_inf else 0

        print(f"\n  • Array: '{key}'")
        print(f"    - Shape:            {shape}")
        print(f"    - Data Type:        {dtype}")
        print(f"    - Samples:          {n_samples}")
        print(f"    - NaN Count:        {nan_count} {'(OK)' if nan_count == 0 else '(WARNING: NaN DETECTED!)'}")
        print(f"    - Inf Count:        {inf_count} {'(OK)' if inf_count == 0 else '(WARNING: Inf DETECTED!)'}")

        if np.issubdtype(dtype, np.number) and arr.size > 0:
            arr_min = float(np.min(arr))
            arr_max = float(np.max(arr))
            arr_mean = float(np.mean(arr))
            print(f"    - Value Range:      min={arr_min:.4f}, max={arr_max:.4f}, mean={arr_mean:.4f}")

        # Guess role
        if len(shape) == 2 and shape[1] == 25:
            feature_candidate = key
        elif len(shape) == 3 and shape[1] * shape[2] == 25:
            feature_candidate = key
        elif len(shape) == 1 or (len(shape) == 2 and shape[1] == 1):
            if np.issubdtype(dtype, np.integer) or (np.issubdtype(dtype, np.floating) and np.all(arr == np.round(arr))):
                label_candidate = key

    # Compatibility Assessment
    print("\n" + "-" * 75)
    print("[*] Compatibility Assessment with YOURMODEL.h5 [Input: (None, 25), Output: (None, 5)]:")

    if feature_candidate:
        arr_feat = data[feature_candidate]
        print(f"    [✓] Observation Features Detected: Array '{feature_candidate}'")
        print(f"        Shape: {arr_feat.shape} -> Can be reshaped to ({arr_feat.shape[0]}, 25)")
    else:
        print("    [!] WARNING: Could not find an obvious 25-dimensional feature array.")
        print(f"        Available arrays: {[(k, data[k].shape) for k in keys]}")
        print("        Check if the array needs flattening or has a different dimension name.")

    if label_candidate:
        arr_label = data[label_candidate].flatten()
        unique_labels = sorted(list(set(arr_label.tolist())))
        print(f"    [✓] Action Labels Detected: Array '{label_candidate}'")
        print(f"        Shape: {data[label_candidate].shape}")
        print(f"        Unique Class IDs Observed: {unique_labels}")
        if set(unique_labels).issubset({0, 1, 2, 3, 4}):
            print("        Classes match highway-env 5 discrete actions: 0, 1, 2, 3, 4 (PASS)")
        else:
            print("        [!] Warning: Detected action IDs outside expected range [0, 1, 2, 3, 4]!")
    else:
        print("    [-] Note: No separate integer label array detected.")
        print("        (If this is an unlabeled test set for inference, this is normal.)")

    print("=" * 75)
    print(" Inspection complete. File remains 100% unaltered (read-only mode).")
    print("=" * 75)

if __name__ == "__main__":
    main()
