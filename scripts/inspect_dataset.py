#!/usr/bin/env python3
"""
scripts/inspect_dataset.py
Validate dataset/train.npz and dataset/validation.npz for the midterm project.
"""

import os
import sys
import numpy as np


def inspect_single_dataset(name, file_path):
    print("=" * 60)
    print(f" Inspecting {name}: {file_path}")
    print("=" * 60)

    if not os.path.exists(file_path):
        print(f"[ERROR] File does not exist: {file_path}")
        return None

    try:
        loaded = np.load(file_path)
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return None

    # 1. Dataset keys
    keys = list(loaded.keys())
    print(f"\n1. Dataset Keys: {keys}")

    # Identify data and label keys dynamically (do not guess)
    data_key = None
    label_key = None

    # Check common candidates
    for k in keys:
        kl = k.lower()
        if kl in ["data", "x", "obs", "observations", "features", "inputs"]:
            data_key = k
        elif kl in ["label", "labels", "y", "action", "actions", "targets"]:
            label_key = k

    # If not identified by name, try heuristics based on dimension
    if data_key is None and len(keys) >= 1:
        # data usually has ndim >= 2
        for k in keys:
            if loaded[k].ndim >= 2:
                data_key = k
                break
    if label_key is None and len(keys) >= 2:
        for k in keys:
            if k != data_key:
                label_key = k
                break

    print(f"   - Identified Feature Key: '{data_key}'")
    print(f"   - Identified Label Key:   '{label_key}'")

    if data_key is None:
        print("[ERROR] Could not identify feature array.")
        return None

    data_arr = loaded[data_key]
    label_arr = loaded[label_key] if label_key else None

    # 2. Feature Shape
    print(f"\n2. Feature Shape: {data_arr.shape}")

    # 3. Label Shape
    if label_arr is not None:
        print(f"\n3. Label Shape: {label_arr.shape}")
    else:
        print("\n3. Label Shape: [MISSING / NOT FOUND]")

    # 4. Data dtype
    print(f"\n4. Data Dtypes:")
    print(f"   - Feature Dtype: {data_arr.dtype}")
    if label_arr is not None:
        print(f"   - Label Dtype:   {label_arr.dtype}")

    # 5. Number of Samples
    n_samples = len(data_arr)
    print(f"\n5. Number of Samples: {n_samples}")
    if label_arr is not None:
        if len(label_arr) != n_samples:
            print(f"   [WARNING] Sample count mismatch! Features: {n_samples}, Labels: {len(label_arr)}")
        else:
            print(f"   [PASS] Feature and label sample counts match ({n_samples}).")

    # 6. Number of Features
    feature_size = data_arr[0].size
    can_reshape_25 = (feature_size == 25)
    print(f"\n6. Number of Features per Sample: {feature_size}")
    print(f"   - Can reshape to (N, 25): {can_reshape_25}")

    # 7 & 8. Unique Labels & Distribution
    unique_labels = None
    if label_arr is not None:
        # Flatten label if one-hot or 2D
        if label_arr.ndim == 2 and label_arr.shape[1] == 5:
            # One-hot encoded labels
            print("   - Note: Labels appear to be One-Hot encoded.")
            cat_labels = np.argmax(label_arr, axis=1)
        else:
            cat_labels = label_arr.flatten()

        unique_labels, counts = np.unique(cat_labels, return_counts=True)
        print(f"\n7. Unique Labels: {unique_labels.tolist()}")
        expected_classes = set(range(5))
        is_0_to_4 = set(unique_labels.tolist()).issubset(expected_classes)
        print(f"   - Are all labels in 0-4 classes: {is_0_to_4}")

        print(f"\n8. Label Distribution:")
        for ul, count in zip(unique_labels, counts):
            pct = (count / len(cat_labels)) * 100
            print(f"   - Class {ul}: {count:6d} ({pct:5.2f}%)")
    else:
        print("\n7 & 8. Labels not available for distribution check.")

    # 9. Missing Values (NaN)
    nan_in_data = np.isnan(data_arr).sum()
    print(f"\n9. Missing Values (NaN):")
    print(f"   - NaN in Features: {nan_in_data}")
    if label_arr is not None:
        nan_in_label = np.isnan(label_arr).sum() if np.issubdtype(label_arr.dtype, np.floating) else 0
        print(f"   - NaN in Labels:   {nan_in_label}")

    # 10. Infinite Values
    inf_in_data = np.isinf(data_arr).sum()
    print(f"\n10. Infinite Values (Inf):")
    print(f"   - Inf in Features: {inf_in_data}")

    # 11. Feature Min/Max/Mean/Std
    f_min = float(np.min(data_arr))
    f_max = float(np.max(data_arr))
    f_mean = float(np.mean(data_arr))
    f_std = float(np.std(data_arr))
    print(f"\n11. Feature Statistics:")
    print(f"   - Min:  {f_min:.6f}")
    print(f"   - Max:  {f_max:.6f}")
    print(f"   - Mean: {f_mean:.6f}")
    print(f"   - Std:  {f_std:.6f}")

    return {
        "file_path": file_path,
        "keys": keys,
        "data_key": data_key,
        "label_key": label_key,
        "feature_shape": data_arr.shape,
        "label_shape": label_arr.shape if label_arr is not None else None,
        "feature_dtype": str(data_arr.dtype),
        "label_dtype": str(label_arr.dtype) if label_arr is not None else None,
        "n_samples": n_samples,
        "feature_size": feature_size,
        "can_reshape_25": can_reshape_25,
        "unique_labels": unique_labels.tolist() if unique_labels is not None else None,
        "nan_features": int(nan_in_data),
        "inf_features": int(inf_in_data),
        "min": f_min,
        "max": f_max,
        "mean": f_mean,
        "std": f_std,
    }


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_path = os.path.join(root_dir, "dataset", "train.npz")
    val_path = os.path.join(root_dir, "dataset", "validation.npz")

    print(f"Checking dataset directory in {os.path.join(root_dir, 'dataset')}...")
    train_exists = os.path.exists(train_path)
    val_exists = os.path.exists(val_path)

    print(f"train.npz exists:      {train_exists}")
    print(f"validation.npz exists: {val_exists}")

    if not train_exists or not val_exists:
        print("\n[ALERT] One or more dataset files are missing!")
        if not train_exists:
            print(f"  - Missing: {train_path}")
        if not val_exists:
            print(f"  - Missing: {val_path}")
        print("Per instructions, fake data will NOT be generated.")
        print("Please place the official train.npz and validation.npz into the dataset/ directory.")

    train_info = inspect_single_dataset("Training Set", train_path) if train_exists else None
    val_info = inspect_single_dataset("Validation Set", val_path) if val_exists else None

    # 12. Training vs Validation Shape Consistency
    print("\n" + "=" * 60)
    print(" 12. Training vs Validation Consistency Check")
    print("=" * 60)
    if train_info and val_info:
        # Check feature dimension compatibility
        feat_match = (train_info["feature_shape"][1:] == val_info["feature_shape"][1:])
        print(f"Feature shape (excluding batch) match: {feat_match}")
        print(f"  Train: {train_info['feature_shape']}")
        print(f"  Val:   {val_info['feature_shape']}")

        label_match = (
            train_info["label_shape"] is not None
            and val_info["label_shape"] is not None
            and train_info["label_shape"][1:] == val_info["label_shape"][1:]
        )
        print(f"Label shape (excluding batch) match: {label_match}")
        print(f"  Train: {train_info['label_shape']}")
        print(f"  Val:   {val_info['label_shape']}")

        keys_match = (train_info["keys"] == val_info["keys"])
        print(f"Dataset keys match: {keys_match} ({train_info['keys']})")

        print("\nOverall Consistency Summary:")
        if feat_match and label_match and keys_match:
            print("[PASS] Training and Validation datasets are fully consistent.")
        else:
            print("[WARNING] Inconsistencies detected between train and val.")
    else:
        print("[PENDING / 尚待確認] Cannot perform consistency check because dataset file(s) are missing.")


if __name__ == "__main__":
    main()

