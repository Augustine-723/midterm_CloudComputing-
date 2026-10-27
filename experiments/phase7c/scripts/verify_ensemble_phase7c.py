#!/usr/bin/env python3
"""
experiments/phase7c/scripts/verify_ensemble_phase7c.py
Phase 7C: Ensemble Final Verification & Paired Statistical Audit.

Models:
- Control:   models/baseline_standard.h5
- Candidate: experiments/phase7b/models/ensemble_unified.h5

Components:
1. Teacher Compatibility Verification
2. Classification Verification & McNemar Test on validation.npz (5,000 samples)
3. 100-Episode Paired Environment Evaluation on roundabout-v0 (10 rounds x 10 episodes)
4. Bootstrap 95% Confidence Intervals for paired differences (10,000 resamples)
5. Comprehensive Visualizations & JSON Export
"""

import os
import sys
import json
import time
import math
import numpy as np

# Headless & TF logging suppression
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import gymnasium as gym
import highway_env

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = ["0: LANE_LEFT", "1: IDLE", "2: LANE_RIGHT", "3: FASTER", "4: SLOWER"]
SHORT_NAMES = ["L_LEFT", "IDLE", "L_RIGHT", "FASTER", "SLOWER"]


def verify_model_compatibility(model_path, model_name):
    """Verify strict teacher test requirements without custom objects."""
    print(f"\n[Compatibility Check] {model_name} ({model_path})")
    
    # 1. Load without custom_objects
    t0 = time.perf_counter()
    model = tf.keras.models.load_model(model_path, compile=False)
    load_time_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  - Loaded via tf.keras.models.load_model() in {load_time_ms:.2f} ms")

    # 2. Input/Output Shape & Types
    input_shape = list(model.input_shape)
    output_shape = list(model.output_shape)
    print(f"  - Input shape:  {input_shape}")
    print(f"  - Output shape: {output_shape}")
    assert input_shape == [None, 25], f"Invalid input shape: {input_shape}"
    assert output_shape == [None, 5], f"Invalid output shape: {output_shape}"

    # 3. Softmax sum and argmax check
    dummy_obs = np.random.randn(10, 25).astype(np.float32)
    probs = model(tf.constant(dummy_obs), training=False).numpy()
    row_sums = np.sum(probs, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)
    actions = np.argmax(probs, axis=-1)
    assert np.all((actions >= 0) & (actions <= 4)), f"Actions out of range: {actions}"
    print(f"  - Softmax probability row sums: min={row_sums.min():.6f}, max={row_sums.max():.6f}")
    print(f"  - np.argmax() output dtype={actions.dtype}, values={actions[:5]}")
    print(f"  [PASS] Teacher compatibility verified for {model_name}.")

    return {
        "status": "PASS",
        "load_time_ms": round(load_time_ms, 2),
        "input_shape": input_shape,
        "output_shape": output_shape,
        "param_count": int(model.count_params()),
    }


def compute_mcnemar_test(y_true, p_base, p_ens):
    """
    Compute McNemar Test for paired binary outcomes on the exact same dataset.
    Contingency table:
      n11: Base Correct, Ens Correct
      n10: Base Correct, Ens Wrong (b)
      n01: Base Wrong, Ens Correct (c)
      n00: Base Wrong, Ens Wrong
    """
    base_correct = (p_base == y_true)
    ens_correct = (p_ens == y_true)

    n11 = int(np.sum(base_correct & ens_correct))
    n10 = int(np.sum(base_correct & (~ens_correct)))
    n01 = int(np.sum((~base_correct) & ens_correct))
    n00 = int(np.sum((~base_correct) & (~ens_correct)))

    total = len(y_true)
    b = n10
    c = n01
    discordant_total = b + c

    # Chi-square without continuity correction
    chi2_uncorrected = float((b - c) ** 2 / discordant_total) if discordant_total > 0 else 0.0

    # Chi-square with Edwards continuity correction
    chi2_corrected = float((abs(b - c) - 1.0) ** 2 / discordant_total) if discordant_total > 0 else 0.0

    # Compute p-values via survival function of chi2 with 1 degree of freedom
    # For df=1, chi2 = z^2, so p-value = 2 * (1 - norm_cdf(sqrt(chi2))) = erfc(sqrt(chi2 / 2))
    p_val_uncorrected = float(math.erfc(math.sqrt(chi2_uncorrected / 2.0)))
    p_val_corrected = float(math.erfc(math.sqrt(chi2_corrected / 2.0)))

    # Exact two-sided binomial p-value: Binomial(discordant_total, 0.5)
    k_min = min(b, c)
    log_half = math.log(0.5)
    binom_p = 0.0
    for k in range(k_min + 1):
        log_comb = math.lgamma(discordant_total + 1) - math.lgamma(k + 1) - math.lgamma(discordant_total - k + 1)
        binom_p += math.exp(log_comb + discordant_total * log_half)
    exact_binomial_p = min(1.0, 2.0 * binom_p)

    return {
        "contingency_table": {
            "n11_both_correct": n11,
            "n10_base_correct_ens_wrong": n10,
            "n01_base_wrong_ens_correct": n01,
            "n00_both_wrong": n00,
            "total_samples": total,
        },
        "base_accuracy": round(float(np.mean(base_correct)), 4),
        "ens_accuracy": round(float(np.mean(ens_correct)), 4),
        "base_errors": int(n01 + n00),
        "ens_errors": int(n10 + n00),
        "error_difference": int((n01 + n00) - (n10 + n00)),
        "discordant_pairs": discordant_total,
        "chi2_uncorrected": round(chi2_uncorrected, 4),
        "p_value_uncorrected": round(p_val_uncorrected, 4),
        "chi2_corrected": round(chi2_corrected, 4),
        "p_value_corrected": round(p_val_corrected, 4),
        "exact_binomial_p_value": round(exact_binomial_p, 4),
        "statistically_significant_alpha_005": bool(p_val_corrected < 0.05),
    }


def evaluate_model_in_environment(model_path, model_name, num_rounds=10, episodes_per_round=10, seed_base=42):
    """
    Run 100 paired episodes in roundabout-v0 matching teacher NN_test.py.
    """
    print(f"\n[Environment Simulation] Evaluating {model_name}...")
    model = tf.keras.models.load_model(model_path, compile=False)
    env = gym.make("roundabout-v0", render_mode="rgb_array")

    total_episodes = num_rounds * episodes_per_round
    round_rewards_float = []
    round_rewards_int = []
    episodes_data = []

    collisions = 0
    spatial_arrivals = 0
    action_counts = {c: 0 for c in range(5)}
    speed_saturation_count = 0
    total_steps = 0

    for r in range(num_rounds):
        round_sum = 0.0
        for ep in range(episodes_per_round):
            ep_seed = seed_base + r * 100 + ep
            obs, info = env.reset(seed=ep_seed)
            done = False
            truncated = False
            ep_reward = 0.0
            ep_steps = 0
            ep_actions = []

            while not (done or truncated):
                current_speed = float(env.unwrapped.vehicle.speed)
                obs_reshaped = obs.reshape(1, 25).astype(np.float32)

                probs = model(tf.constant(obs_reshaped), training=False).numpy()
                action = int(np.argmax(probs[0]))
                action_counts[action] += 1
                ep_actions.append(action)
                total_steps += 1
                ep_steps += 1

                # Speed saturation: SLOWER at min speed (<=0.5) or FASTER at max speed (>=15.5)
                if (action == 4 and current_speed <= 0.5) or (action == 3 and current_speed >= 15.5):
                    speed_saturation_count += 1

                obs, reward, done, truncated, info = env.step(action)
                ep_reward += float(reward)

            round_sum += ep_reward
            crashed = bool(info.get("crashed", False) or env.unwrapped.vehicle.crashed)
            final_lane = env.unwrapped.vehicle.lane_index
            reached_destination = ("nxs" in str(final_lane))
            is_arrival = reached_destination and (not crashed)

            if crashed:
                collisions += 1
            if is_arrival:
                spatial_arrivals += 1

            episodes_data.append({
                "round": r + 1,
                "episode": ep + 1,
                "seed": ep_seed,
                "reward": float(ep_reward),
                "steps": ep_steps,
                "crashed": crashed,
                "arrival": is_arrival,
                "final_lane": str(final_lane),
                "final_speed": float(env.unwrapped.vehicle.speed),
            })

        round_rewards_float.append(float(round_sum))
        round_rewards_int.append(int(round_sum))
        print(f"  Round {r + 1:02d}: Float Sum = {round_sum:6.2f} | Teacher Int = {int(round_sum):3d}")

    env.close()

    all_ep_rewards = np.array([e["reward"] for e in episodes_data])
    all_ep_crashes = np.array([1 if e["crashed"] else 0 for e in episodes_data])
    all_ep_arrivals = np.array([1 if e["arrival"] else 0 for e in episodes_data])

    action_percentages = {
        CLASS_NAMES[c]: round(count / total_steps * 100.0, 2)
        for c, count in action_counts.items()
    }

    return {
        "model_name": model_name,
        "model_path": model_path,
        "teacher_round_rewards_int": round_rewards_int,
        "round_rewards_float": [round(v, 2) for v in round_rewards_float],
        "mean_round_reward_float": round(float(np.mean(round_rewards_float)), 2),
        "std_round_reward_float": round(float(np.std(round_rewards_float)), 2),
        "mean_round_reward_int": round(float(np.mean(round_rewards_int)), 2),
        "std_round_reward_int": round(float(np.std(round_rewards_int)), 2),
        "mean_episode_reward": round(float(np.mean(all_ep_rewards)), 2),
        "std_episode_reward": round(float(np.std(all_ep_rewards)), 2),
        "collision_count": int(collisions),
        "collision_rate_percent": round(float(collisions / total_episodes * 100.0), 2),
        "arrival_count": int(spatial_arrivals),
        "arrival_rate_percent": round(float(spatial_arrivals / total_episodes * 100.0), 2),
        "speed_saturation_count": int(speed_saturation_count),
        "speed_saturation_rate_percent": round(float(speed_saturation_count / total_steps * 100.0), 2) if total_steps > 0 else 0.0,
        "action_counts": action_counts,
        "action_percentages": action_percentages,
        "episodes": episodes_data,
        "raw_rewards": all_ep_rewards,
        "raw_crashes": all_ep_crashes,
        "raw_arrivals": all_ep_arrivals,
    }


def compute_paired_bootstrap_ci(base_vals, ens_vals, n_resamples=10000, seed=42):
    """
    Compute 95% Bootstrap Confidence Interval for paired differences (ens - base).
    Returns mean diff, median diff, 2.5% CI, 97.5% CI, and distribution.
    """
    rng = np.random.RandomState(seed)
    diffs = ens_vals - base_vals
    n = len(diffs)

    boot_means = np.zeros(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sample_indices = rng.randint(0, n, size=n)
        boot_means[i] = np.mean(diffs[sample_indices])

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))

    return {
        "observed_mean_difference": round(mean_diff, 4),
        "observed_median_difference": round(median_diff, 4),
        "ci_95_lower": round(ci_lower, 4),
        "ci_95_upper": round(ci_upper, 4),
        "ci_contains_zero": bool(ci_lower <= 0.0 <= ci_upper),
        "bootstrap_std": round(float(np.std(boot_means)), 4),
        "boot_samples": boot_means,
    }


def plot_action_distribution(base_acts, ens_acts, output_path):
    """Plot grouped bar chart comparing action distributions."""
    x = np.arange(len(SHORT_NAMES))
    width = 0.35

    base_pcts = [base_acts[CLASS_NAMES[i]] for i in range(5)]
    ens_pcts = [ens_acts[CLASS_NAMES[i]] for i in range(5)]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    r1 = ax.bar(x - width/2, base_pcts, width, label="Control (Baseline Standard)", color="#1f77b4")
    r2 = ax.bar(x + width/2, ens_pcts, width, label="Candidate (Unified Ensemble)", color="#2ca02c")

    ax.set_ylabel("Execution Percentage (%)", fontsize=11)
    ax.set_title("Simulator Action Distribution Comparison (100 Episodes, roundabout-v0)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_NAMES, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for r in r1:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for r in r2:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved action distribution figure to: {output_path}")


def plot_reward_and_bootstrap(reward_boot, crash_boot, ep_diffs, output_path):
    """Plot paired difference histogram and bootstrap sampling distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Subplot 1: Episode Reward Paired Difference & Bootstrap CI
    ax1.hist(reward_boot["boot_samples"], bins=50, color="#2ca02c", alpha=0.7, edgecolor="black", density=True)
    ax1.axvline(reward_boot["observed_mean_difference"], color="red", linestyle="-", linewidth=2,
                label=f"Mean Diff: {reward_boot['observed_mean_difference']:.2f}")
    ax1.axvline(reward_boot["ci_95_lower"], color="black", linestyle="--", linewidth=1.5,
                label=f"95% CI Lower: {reward_boot['ci_95_lower']:.2f}")
    ax1.axvline(reward_boot["ci_95_upper"], color="black", linestyle="--", linewidth=1.5,
                label=f"95% CI Upper: {reward_boot['ci_95_upper']:.2f}")
    ax1.axvline(0.0, color="gray", linestyle=":", linewidth=1.5, label="Zero Difference (Reference)")
    ax1.set_title("Bootstrap Distribution: Paired Episode Reward Difference\n(Ensemble - Baseline, B=10,000)",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("Mean Difference in Episode Reward", fontsize=10)
    ax1.set_ylabel("Probability Density", fontsize=10)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(axis="x", linestyle="--", alpha=0.5)

    # Subplot 2: Crash Rate Paired Difference Bootstrap CI
    # Scaled to percentage points (difference in crash probability)
    crash_pct_boot = crash_boot["boot_samples"] * 100.0
    ax2.hist(crash_pct_boot, bins=50, color="#d62728", alpha=0.7, edgecolor="black", density=True)
    ax2.axvline(crash_boot["observed_mean_difference"] * 100.0, color="blue", linestyle="-", linewidth=2,
                label=f"Mean Diff: {crash_boot['observed_mean_difference']*100:.1f}%")
    ax2.axvline(crash_boot["ci_95_lower"] * 100.0, color="black", linestyle="--", linewidth=1.5,
                label=f"95% CI Lower: {crash_boot['ci_95_lower']*100:.1f}%")
    ax2.axvline(crash_boot["ci_95_upper"] * 100.0, color="black", linestyle="--", linewidth=1.5,
                label=f"95% CI Upper: {crash_boot['ci_95_upper']*100:.1f}%")
    ax2.axvline(0.0, color="gray", linestyle=":", linewidth=1.5, label="Zero Difference (Reference)")
    ax2.set_title("Bootstrap Distribution: Paired Collision Rate Difference\n(Ensemble - Baseline, B=10,000)",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("Difference in Collision Rate (Percentage Points)", fontsize=10)
    ax2.set_ylabel("Probability Density", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved reward & bootstrap figure to: {output_path}")


def plot_mcnemar_contingency(mcnemar_res, output_path):
    """Plot 2x2 contingency matrix and McNemar discordant breakdown."""
    tbl = mcnemar_res["contingency_table"]
    mat = np.array([
        [tbl["n11_both_correct"], tbl["n10_base_correct_ens_wrong"]],
        [tbl["n01_base_wrong_ens_correct"], tbl["n00_both_wrong"]]
    ])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # 1. Heatmap
    im = ax1.imshow(mat, cmap="YlGnBu", interpolation="nearest")
    ax1.set_title("McNemar Paired Classification Contingency Matrix\n(5,000 Validation Samples)", fontsize=11, fontweight="bold")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Ensemble Correct", "Ensemble Wrong"], fontsize=10)
    ax1.set_yticklabels(["Baseline Correct", "Baseline Wrong"], fontsize=10)

    thresh = mat.max() / 2.0
    for i in range(2):
        for j in range(2):
            color = "white" if mat[i, j] > thresh else "black"
            ax1.text(j, i, f"{mat[i, j]:,}\n({mat[i, j]/5000*100:.2f}%)", ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    # 2. Discordant Pair Breakdown Bar Chart
    disc_labels = ["Baseline Correct,\nEnsemble Wrong (b)", "Baseline Wrong,\nEnsemble Correct (c)"]
    disc_vals = [tbl["n10_base_correct_ens_wrong"], tbl["n01_base_wrong_ens_correct"]]
    colors = ["#ff7f0e", "#2ca02c"]
    bars = ax2.bar(disc_labels, disc_vals, color=colors, width=0.5)
    ax2.set_ylabel("Number of Samples", fontsize=10)
    ax2.set_title(f"Discordant Pairs Analysis (Total Discordant: {tbl['n10_base_correct_ens_wrong'] + tbl['n01_base_wrong_ens_correct']})\n"
                  f"McNemar χ²={mcnemar_res['chi2_corrected']:.3f}, p={mcnemar_res['p_value_corrected']:.4f}",
                  fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for b in bars:
        h = b.get_height()
        ax2.annotate(f"{h} samples\n({h/200*100:.1f}%)", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved McNemar contingency figure to: {output_path}")


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    phase_dir = os.path.join(repo_root, "experiments", "phase7c")
    reports_dir = os.path.join(phase_dir, "reports")
    figures_dir = os.path.join(phase_dir, "figures")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 75)
    print(" PHASE 7C — ENSEMBLE FINAL VERIFICATION & STATISTICAL AUDIT")
    print("=" * 75)
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Gymnasium Version:  {gym.__version__}")

    control_path = os.path.join(repo_root, "models", "baseline_standard.h5")
    candidate_path = os.path.join(repo_root, "experiments", "phase7b", "models", "ensemble_unified.h5")

    # 1. Teacher Compatibility Verification
    print("\n" + "=" * 75)
    print(" [1] TEACHER COMPATIBILITY AUDIT")
    print("=" * 75)
    control_compat = verify_model_compatibility(control_path, "Control (Baseline Standard)")
    candidate_compat = verify_model_compatibility(candidate_path, "Candidate (Unified Ensemble)")

    # 2. Classification Verification & McNemar Test
    print("\n" + "=" * 75)
    print(" [2] CLASSIFICATION VERIFICATION & MCNEMAR TEST (dataset/validation.npz)")
    print("=" * 75)
    val_npz = np.load(os.path.join(repo_root, "dataset", "validation.npz"))
    x_val = val_npz["data"].astype(np.float32)
    y_val = val_npz["label"].astype(np.int32).flatten()

    m_control = tf.keras.models.load_model(control_path, compile=False)
    m_candidate = tf.keras.models.load_model(candidate_path, compile=False)

    p_control = np.argmax(m_control.predict(x_val, verbose=0), axis=1)
    p_candidate = np.argmax(m_candidate.predict(x_val, verbose=0), axis=1)

    mcnemar_res = compute_mcnemar_test(y_val, p_control, p_candidate)
    print(f"  Baseline Validation Accuracy: {mcnemar_res['base_accuracy']*100:.2f}% ({mcnemar_res['base_errors']} errors)")
    print(f"  Ensemble Validation Accuracy: {mcnemar_res['ens_accuracy']*100:.2f}% ({mcnemar_res['ens_errors']} errors)")
    print(f"  Net Error Reduction:          {mcnemar_res['error_difference']} fewer errors")
    print(f"  Both Correct (n11):           {mcnemar_res['contingency_table']['n11_both_correct']}")
    print(f"  Baseline Correct, Ens Wrong:  {mcnemar_res['contingency_table']['n10_base_correct_ens_wrong']}")
    print(f"  Baseline Wrong, Ens Correct:  {mcnemar_res['contingency_table']['n01_base_wrong_ens_correct']}")
    print(f"  Both Wrong (n00):             {mcnemar_res['contingency_table']['n00_both_wrong']}")
    print(f"  McNemar Chi-Square (Corr):    {mcnemar_res['chi2_corrected']:.4f} (p-value: {mcnemar_res['p_value_corrected']:.4f})")
    print(f"  Exact Binomial p-value:       {mcnemar_res['exact_binomial_p_value']:.4f}")
    print(f"  Significant at alpha=0.05:    {mcnemar_res['statistically_significant_alpha_005']}")

    # 3. Paired Environment Simulation
    print("\n" + "=" * 75)
    print(" [3] PAIRED ENVIRONMENT EVALUATION (100 Episodes, roundabout-v0)")
    print("=" * 75)
    env_control = evaluate_model_in_environment(control_path, "Control (Baseline Standard)")
    env_candidate = evaluate_model_in_environment(candidate_path, "Candidate (Unified Ensemble)")

    # 4. Bootstrap 95% Confidence Intervals
    print("\n" + "=" * 75)
    print(" [4] BOOTSTRAP 95% CONFIDENCE INTERVALS (10,000 Resamples)")
    print("=" * 75)
    reward_boot = compute_paired_bootstrap_ci(env_control["raw_rewards"], env_candidate["raw_rewards"])
    crash_boot = compute_paired_bootstrap_ci(env_control["raw_crashes"], env_candidate["raw_crashes"])
    arrival_boot = compute_paired_bootstrap_ci(env_control["raw_arrivals"], env_candidate["raw_arrivals"])

    print(f"  Paired Episode Reward Difference (Ensemble - Baseline):")
    print(f"    Observed Mean Diff: {reward_boot['observed_mean_difference']:.4f}")
    print(f"    95% Bootstrap CI:   [{reward_boot['ci_95_lower']:.4f}, {reward_boot['ci_95_upper']:.4f}]")
    print(f"    Contains Zero:      {reward_boot['ci_contains_zero']}")

    print(f"\n  Paired Collision Rate Difference (Ensemble - Baseline):")
    print(f"    Observed Mean Diff: {crash_boot['observed_mean_difference']*100:.2f}%")
    print(f"    95% Bootstrap CI:   [{crash_boot['ci_95_lower']*100:.2f}%, {crash_boot['ci_95_upper']*100:.2f}%]")
    print(f"    Contains Zero:      {crash_boot['ci_contains_zero']}")

    print(f"\n  Paired Arrival Rate Difference (Ensemble - Baseline):")
    print(f"    Observed Mean Diff: {arrival_boot['observed_mean_difference']*100:.2f}%")
    print(f"    95% Bootstrap CI:   [{arrival_boot['ci_95_lower']*100:.2f}%, {arrival_boot['ci_95_upper']*100:.2f}%]")
    print(f"    Contains Zero:      {arrival_boot['ci_contains_zero']}")

    # 5. Visualizations
    print("\n" + "=" * 75)
    print(" [5] GENERATING VERIFICATION FIGURES")
    print("=" * 75)
    fig_action = os.path.join(figures_dir, "phase7c_action_distribution.png")
    plot_action_distribution(env_control["action_percentages"], env_candidate["action_percentages"], fig_action)

    fig_boot = os.path.join(figures_dir, "phase7c_reward_and_bootstrap.png")
    plot_reward_and_bootstrap(reward_boot, crash_boot, env_candidate["raw_rewards"] - env_control["raw_rewards"], fig_boot)

    fig_mcnemar = os.path.join(figures_dir, "phase7c_mcnemar_contingency.png")
    plot_mcnemar_contingency(mcnemar_res, fig_mcnemar)

    # 6. JSON Export (clean non-serializable elements)
    def clean_dict(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                continue
            elif isinstance(v, dict):
                out[k] = clean_dict(v)
            elif isinstance(v, (np.float32, np.float64)):
                out[k] = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                out[k] = int(v)
            else:
                out[k] = v
        return out

    clean_env_control = clean_dict(env_control)
    clean_env_candidate = clean_dict(env_candidate)
    clean_reward_boot = clean_dict(reward_boot)
    clean_crash_boot = clean_dict(crash_boot)
    clean_arrival_boot = clean_dict(arrival_boot)

    final_results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tf_version": tf.__version__,
            "gymnasium_version": gym.__version__,
            "highway_env_version": "1.8.2",
            "num_episodes": 100,
            "seed_base": 42,
        },
        "model_compatibility": {
            "control": control_compat,
            "candidate": candidate_compat,
        },
        "classification_mcnemar": mcnemar_res,
        "environment_control": clean_env_control,
        "environment_candidate": clean_env_candidate,
        "paired_bootstrap_ci": {
            "reward_difference": clean_reward_boot,
            "collision_rate_difference": clean_crash_boot,
            "arrival_rate_difference": clean_arrival_boot,
        }
    }

    json_path = os.path.join(reports_dir, "phase7c_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nFinal Phase 7C results saved to: {json_path}")
    print("=" * 75)
    print(" PHASE 7C VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    main()
