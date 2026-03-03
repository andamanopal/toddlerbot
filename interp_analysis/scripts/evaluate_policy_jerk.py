#!/usr/bin/env python3
"""Evaluate policy-output jerk from recorded rollouts.

Loads NPZ files produced by record_policy.py and computes policy-level
metrics: jerk, tracking RMSE, and spectral energy.

Usage:
    python -m interp_analysis.scripts.evaluate_policy_jerk \
        --rollout-dir rollouts/ \
        --output-dir evaluation_results/
"""

import argparse
import json
import os
from glob import glob

import numpy as np

from interp_analysis.evaluation.metrics import (
    compute_jerk,
    compute_mean_jerk,
    compute_per_joint_rmse,
    compute_spectral_energy,
    compute_tracking_rmse,
)


def evaluate_rollout(npz_path: str) -> dict:
    """Compute metrics for a single rollout NPZ file.

    Args:
        npz_path: Path to the NPZ file.

    Returns:
        Dict with computed metrics.
    """

    data = np.load(npz_path, allow_pickle=True)
    dt = float(data["dt"])
    actual_pos = data["actual_motor_pos"]

    result = {
        "dt": dt,
        "n_steps": int(data["n_steps"]),
        "n_motors": int(data["n_motors"]),
        "interp_method": str(data["interp_method"]),
    }

    jerk_vals = compute_jerk(actual_pos, dt)
    result["policy_mean_jerk"] = float(compute_mean_jerk(actual_pos, dt))
    result["policy_max_jerk"] = float(np.max(jerk_vals)) if len(jerk_vals) > 0 else 0.0

    spectral = compute_spectral_energy(actual_pos, dt)
    result["policy_high_freq_ratio"] = spectral["high_freq_ratio"]

    if "ref_motor_pos" in data:
        ref_pos = data["ref_motor_pos"]
        min_len = min(actual_pos.shape[0], ref_pos.shape[0])
        actual_trimmed = actual_pos[:min_len]
        ref_trimmed = ref_pos[:min_len]

        result["ref_mean_jerk"] = float(compute_mean_jerk(ref_pos, dt))
        ref_jerk_vals = compute_jerk(ref_pos, dt)
        result["ref_max_jerk"] = (
            float(np.max(ref_jerk_vals)) if len(ref_jerk_vals) > 0 else 0.0
        )

        result["tracking_rmse"] = float(
            compute_tracking_rmse(actual_trimmed, ref_trimmed)
        )
        per_joint = compute_per_joint_rmse(actual_trimmed, ref_trimmed)
        result["per_joint_rmse"] = per_joint.tolist()

        ref_spectral = compute_spectral_energy(ref_pos, dt)
        result["ref_high_freq_ratio"] = ref_spectral["high_freq_ratio"]

        if result["ref_mean_jerk"] > 0:
            result["jerk_transfer_ratio"] = (
                result["policy_mean_jerk"] / result["ref_mean_jerk"]
            )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate policy jerk from recorded rollouts"
    )
    parser.add_argument(
        "--rollout-dir", type=str, default="rollouts",
        help="Directory containing rollout NPZ files.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluation_results",
        help="Output directory for evaluation results.",
    )
    args = parser.parse_args()

    npz_files = sorted(glob(os.path.join(args.rollout_dir, "*.npz")))
    if not npz_files:
        print(f"No NPZ files found in {args.rollout_dir}")
        return

    print(f"Found {len(npz_files)} rollout files")

    # Group by method
    method_results: dict = {}
    for npz_path in npz_files:
        print(f"  Evaluating {os.path.basename(npz_path)}...")
        metrics = evaluate_rollout(npz_path)
        method = metrics["interp_method"]
        if method not in method_results:
            method_results[method] = []
        method_results[method].append(metrics)

    # Aggregate per method (mean across episodes)
    summary = {}
    for method, episodes in method_results.items():
        agg = {
            "n_episodes": len(episodes),
            "policy_mean_jerk": float(
                np.mean([e["policy_mean_jerk"] for e in episodes])
            ),
            "policy_max_jerk": float(
                np.mean([e["policy_max_jerk"] for e in episodes])
            ),
            "policy_high_freq_ratio": float(
                np.mean([e["policy_high_freq_ratio"] for e in episodes])
            ),
        }

        # Aggregate ref-related fields using only episodes that have them
        ref_episodes = [e for e in episodes if "ref_mean_jerk" in e]
        if ref_episodes:
            agg["ref_mean_jerk"] = float(
                np.mean([e["ref_mean_jerk"] for e in ref_episodes])
            )
            agg["ref_max_jerk"] = float(
                np.mean([e["ref_max_jerk"] for e in ref_episodes])
            )
            agg["tracking_rmse"] = float(
                np.mean([e["tracking_rmse"] for e in ref_episodes])
            )
            agg["ref_high_freq_ratio"] = float(
                np.mean([e["ref_high_freq_ratio"] for e in ref_episodes])
            )

            transfer_episodes = [
                e for e in ref_episodes if "jerk_transfer_ratio" in e
            ]
            if transfer_episodes:
                agg["jerk_transfer_ratio"] = float(
                    np.mean([e["jerk_transfer_ratio"] for e in transfer_episodes])
                )

            per_joint_arrays = [
                np.array(e["per_joint_rmse"]) for e in ref_episodes
                if "per_joint_rmse" in e
            ]
            if per_joint_arrays:
                agg["per_joint_rmse"] = np.mean(
                    np.stack(per_joint_arrays), axis=0
                ).tolist()

        summary[method] = agg

        print(f"\n  {method}:")
        print(f"    Policy mean jerk: {agg['policy_mean_jerk']:.2f}")
        print(f"    Policy max jerk:  {agg['policy_max_jerk']:.2f}")
        if "ref_mean_jerk" in agg:
            print(f"    Ref mean jerk:    {agg['ref_mean_jerk']:.2f}")
            print(f"    Tracking RMSE:    {agg['tracking_rmse']:.4f}")
            if "jerk_transfer_ratio" in agg:
                print(f"    Jerk transfer:    {agg['jerk_transfer_ratio']:.2f}x")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "policy_jerk_results.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
