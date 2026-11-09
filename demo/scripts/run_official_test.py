#!/usr/bin/env python3
"""
demo/scripts/run_official_test.py
Official Evaluation Runner & Telemetry Logger for Demo Day.

Adaptive capabilities:
- Auto-detects official test script (*.py) in demo/incoming/ or accepts --script / -s
- Auto-detects official test dataset (*.npz) in demo/incoming/ or accepts --data / -d
- Accepts custom command invocations via --cmd "<command>" for unknown TA script patterns
- Records pre-run and post-run model SHA-256 to certify zero model tampering
- Saves timestamped execution logs (stdout, stderr, summary JSON) in demo/outputs/
- Read-only safe: guards submission/YOURMODEL.h5 and warns if missing

"""

import os
import sys
import time
import json
import shutil
import hashlib
import argparse
import subprocess

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
INCOMING_DIR = os.path.join(PROJECT_ROOT, "demo", "incoming")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "demo", "outputs")
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "submission", "YOURMODEL.h5")

def compute_sha256(filepath):
    """Compute SHA-256 hash of a file."""
    if not os.path.exists(filepath):
        return None
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

def find_incoming_files():
    """Auto-detect test script and test dataset in demo/incoming/."""
    script_file = None
    data_file = None
    if os.path.exists(INCOMING_DIR):
        for f in os.listdir(INCOMING_DIR):
            fpath = os.path.join(INCOMING_DIR, f)
            if f.endswith(".py") and script_file is None:
                script_file = fpath
            elif f.endswith(".npz") and data_file is None:
                data_file = fpath
    return script_file, data_file

def main():
    parser = argparse.ArgumentParser(description="Official Demo Evaluation Runner.")
    parser.add_argument("--script", "-s", type=str, default=None, help="Path to TA's test script (default: auto-detect in demo/incoming/)")
    parser.add_argument("--data", "-d", type=str, default=None, help="Path to TA's test dataset (default: auto-detect in demo/incoming/)")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained model .h5")
    parser.add_argument("--cmd", type=str, default=None, help="Direct custom command string to execute TA's test script")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and check prerequisites without executing")
    args = parser.parse_args()

    print("=" * 75)
    print(" DEMO DAY: OFFICIAL TEST EVALUATION RUNNER")
    print("=" * 75)

    detected_script, detected_data = find_incoming_files()
    test_script = args.script or detected_script
    test_data = args.data or detected_data
    model_path = os.path.abspath(args.model)

    # 1. Model Verification
    if not os.path.exists(model_path):
        print(f"[!] FATAL: Target model not found: {model_path}")
        sys.exit(1)

    pre_model_sha = compute_sha256(model_path)
    model_size = os.path.getsize(model_path)
    print(f"[*] Target Model:       {model_path}")
    print(f"    - Size:             {model_size} bytes")
    print(f"    - Pre-Run SHA-256:  {pre_model_sha}")

    # 2. Check Prerequisites
    missing_items = []
    if args.cmd is None:
        if test_script is None or not os.path.exists(test_script):
            missing_items.append("Official Test Script (e.g., demo/incoming/test.py)")
        if test_data is None or not os.path.exists(test_data):
            missing_items.append("Official Test Dataset (e.g., demo/incoming/test.npz)")

    if missing_items and args.cmd is None:
        print("\n[!] OFFICIAL TEST FILES NOT YET LOADED:")
        for item in missing_items:
            print(f"    [-] Missing: {item}")
        print("\n[INSTRUCTIONS FOR DEMO DAY]:")
        print(f"    1. When the TA provides the test script and dataset, copy them into:")
        print(f"       {INCOMING_DIR}/")
        print("    2. Re-run this evaluation runner:")
        print("       python demo/scripts/run_official_test.py")
        print("    3. If the TA specifies a custom command format (e.g., python test.py <args>),")
        print("       use the --cmd option:")
        print("       python demo/scripts/run_official_test.py --cmd \"python demo/incoming/test.py --model submission/YOURMODEL.h5 --data demo/incoming/test.npz\"")
        print("=" * 75)
        sys.exit(0)

    # 3. Construct Command
    if args.cmd:
        exec_cmd = args.cmd
    else:
        test_script = os.path.abspath(test_script)
        test_data = os.path.abspath(test_data)
        print(f"[*] Test Script:        {test_script}")
        print(f"[*] Test Dataset:       {test_data}")
        # Default flexible execution pattern
        exec_cmd = f"python {test_script} --model {model_path} --data {test_data}"

    print(f"\n[+] Prepared Execution Command:")
    print(f"    $ {exec_cmd}")

    if args.dry_run:
        print("\n[*] Dry-run mode enabled. Exiting without execution.")
        sys.exit(0)

    # 4. Prepare Output Directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUTS_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[*] Output Directory:   {run_dir}")

    # 5. Execute Command
    print("\n" + "=" * 75)
    print(" STARTING OFFICIAL EVALUATION EXECUTION")
    print("=" * 75)
    start_time = time.time()

    stdout_path = os.path.join(run_dir, "stdout.log")
    stderr_path = os.path.join(run_dir, "stderr.log")

    try:
        proc = subprocess.run(
            exec_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=PROJECT_ROOT
        )
        duration = time.time() - start_time
        exit_code = proc.returncode

        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write(proc.stdout)
        with open(stderr_path, "w", encoding="utf-8") as f:
            f.write(proc.stderr)

        print(proc.stdout)
        if proc.stderr:
            print("\n[STDERR]:")
            print(proc.stderr)

    except Exception as e:
        duration = time.time() - start_time
        exit_code = -1
        print(f"[!] Execution failed with exception: {e}")
        with open(stderr_path, "w", encoding="utf-8") as f:
            f.write(str(e))

    # 6. Post-Run Model Integrity Check
    post_model_sha = compute_sha256(model_path)
    print("\n" + "=" * 75)
    print(" EXECUTION COMPLETED")
    print("=" * 75)
    print(f"[*] Exit Code:          {exit_code}")
    print(f"[*] Elapsed Time:       {duration:.2f} seconds")
    print(f"[*] Post-Run Model SHA: {post_model_sha}")

    integrity_passed = (pre_model_sha == post_model_sha)
    if integrity_passed:
        print("[✓] Model Integrity:    PASS (Model was NOT modified during testing)")
    else:
        print("[!] CRITICAL ALARM:     Model SHA changed! Model file was altered during testing!")

    # 7. Write Summary JSON
    summary = {
        "timestamp": timestamp,
        "command": exec_cmd,
        "exit_code": exit_code,
        "duration_seconds": round(duration, 3),
        "model_path": model_path,
        "pre_model_sha256": pre_model_sha,
        "post_model_sha256": post_model_sha,
        "model_integrity_passed": integrity_passed,
        "test_script": test_script if args.cmd is None else None,
        "test_data": test_data if args.cmd is None else None,
        "stdout_log": stdout_path,
        "stderr_log": stderr_path,
    }
    summary_path = os.path.join(run_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[+] Full telemetry saved to: {summary_path}")
    print("=" * 75)

if __name__ == "__main__":
    main()

