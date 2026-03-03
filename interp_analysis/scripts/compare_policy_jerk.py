#!/usr/bin/env python3
"""Cross-method comparison v2: policy-output jerk figures.

Generates publication-quality figures comparing policy jerk vs reference
jerk across interpolation methods. This answers the key question:
does interpolation smoothness actually reach the trained policy?

Usage:
    python -m interp_analysis.scripts.compare_policy_jerk \\
        --eval-dir evaluation_results/ \\
        --output-dir figures/
"""

import argparse
import json
import os

from interp_analysis.evaluation.metrics import parse_reward_log


def _print_summary(policy_jerk_data: dict) -> None:
    """Print a summary table of policy jerk data to stdout."""
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    linear_pol_jerk = policy_jerk_data.get("linear", {}).get("policy_mean_jerk", 0.0)
    for method, data in policy_jerk_data.items():
        pol_jerk = data["policy_mean_jerk"]
        ref_jerk = data.get("ref_mean_jerk", 0)
        transfer = pol_jerk / ref_jerk if ref_jerk > 0 else float("nan")
        ratio_to_linear = pol_jerk / linear_pol_jerk if linear_pol_jerk > 0 else float("nan")
        print(f"  {method:22s} | policy={pol_jerk:>10.1f} | ref={ref_jerk:>10.1f} "
              f"| transfer={transfer:>5.2f}x | vs_linear={ratio_to_linear:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Generate policy vs reference jerk figures"
    )
    parser.add_argument(
        "--eval-dir", type=str, default="evaluation_results",
        help="Directory with evaluation results.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="figures",
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--ref-smoothness", type=str, default="",
        help="Path to reference_smoothness.json (optional, for overlay).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from interp_analysis.evaluation.visualize import (
        plot_jerk_comparison,
        plot_jerk_transfer_ratio,
        plot_policy_vs_reference_jerk,
        plot_reward_curves,
        plot_summary_table_policy,
        plot_tracking_error,
    )

    # Load policy jerk results
    policy_jerk_path = os.path.join(args.eval_dir, "policy_jerk_results.json")
    if not os.path.exists(policy_jerk_path):
        print(f"No policy jerk results at {policy_jerk_path}")
        print("Run evaluate_policy_jerk.py first.")
        return

    with open(policy_jerk_path) as f:
        policy_jerk_data = json.load(f)

    print(f"Loaded policy jerk data for methods: {list(policy_jerk_data.keys())}")

    # Figure 1: Policy vs Reference Jerk (the key figure)
    print("\n1. Generating policy vs reference jerk figure...")
    plot_policy_vs_reference_jerk(
        policy_jerk_data,
        title="Policy vs Reference Jerk by Interpolation Method",
        output_path=os.path.join(args.output_dir, "policy_vs_ref_jerk.png"),
    )
    print(f"   Saved to {args.output_dir}/policy_vs_ref_jerk.png")

    # Figure 2: Jerk Transfer Ratio
    has_transfer = any(
        "ref_mean_jerk" in d for d in policy_jerk_data.values()
    )
    if has_transfer:
        print("2. Generating jerk transfer ratio figure...")
        plot_jerk_transfer_ratio(
            policy_jerk_data,
            title="Jerk Transfer Ratio (Policy / Reference)",
            output_path=os.path.join(args.output_dir, "jerk_transfer_ratio.png"),
        )
        print(f"   Saved to {args.output_dir}/jerk_transfer_ratio.png")
    else:
        print("2. Skipping jerk transfer ratio (no reference data).")

    # Figure 3: Per-joint Tracking RMSE
    has_per_joint = any(
        "per_joint_rmse" in d for d in policy_jerk_data.values()
    )
    if has_per_joint:
        print("3. Generating per-joint tracking error figure...")
        tracking_data = {
            m: d["per_joint_rmse"]
            for m, d in policy_jerk_data.items()
            if "per_joint_rmse" in d
        }
        plot_tracking_error(
            tracking_data,
            title="Per-Joint Tracking RMSE (Policy vs Reference)",
            output_path=os.path.join(args.output_dir, "tracking_rmse.png"),
        )
        print(f"   Saved to {args.output_dir}/tracking_rmse.png")
    else:
        print("3. Skipping tracking RMSE (no per-joint data).")

    # Figure 4: Summary Table v2
    print("4. Generating extended summary table...")
    plot_summary_table_policy(
        policy_jerk_data,
        output_path=os.path.join(args.output_dir, "summary_table_policy.png"),
    )
    print(f"   Saved to {args.output_dir}/summary_table_policy.png")

    # Figure 5: Policy-only jerk comparison (normalized to linear)
    print("5. Generating policy jerk comparison (normalized)...")
    policy_only_jerk = {
        m: {
            "mean_jerk": d["policy_mean_jerk"],
            "max_jerk": d["policy_max_jerk"],
        }
        for m, d in policy_jerk_data.items()
    }
    plot_jerk_comparison(
        policy_only_jerk,
        title="Policy-Output Jerk (Normalized to Linear)",
        output_path=os.path.join(args.output_dir, "policy_jerk_normalized.png"),
    )
    print(f"   Saved to {args.output_dir}/policy_jerk_normalized.png")

    # Optionally overlay with reference smoothness data
    ref_path = args.ref_smoothness or os.path.join(
        args.eval_dir, "reference_smoothness.json"
    )
    if os.path.exists(ref_path):
        print("6. Generating reference-only jerk comparison...")
        with open(ref_path) as f:
            ref_smoothness = json.load(f)
        plot_jerk_comparison(
            ref_smoothness,
            title="Reference Trajectory Jerk (Normalized to Linear)",
            output_path=os.path.join(args.output_dir, "ref_jerk_normalized.png"),
        )
        print(f"   Saved to {args.output_dir}/ref_jerk_normalized.png")
    else:
        print("6. Skipping reference-only jerk (no reference_smoothness.json).")

    # Reward curves
    runs_path = os.path.join(args.eval_dir, "training_runs.json")
    if os.path.exists(runs_path):
        with open(runs_path) as f:
            runs = json.load(f)

        reward_data = {}
        for method, run_dirs in runs.items():
            for run_dir in run_dirs:
                log_path = os.path.join(run_dir, "output.log")
                if os.path.exists(log_path):
                    steps, rewards = parse_reward_log(log_path)
                    if len(steps) > 0:
                        reward_data[method] = {
                            "steps": steps, "rewards": rewards,
                        }

        if reward_data:
            print("7. Generating reward curves...")
            plot_reward_curves(
                reward_data,
                title="Training Reward by Interpolation Method",
                output_path=os.path.join(args.output_dir, "reward_curves.png"),
            )
            print(f"   Saved to {args.output_dir}/reward_curves.png")
        else:
            print("7. Skipping reward curves (no log data found).")
    else:
        print("7. Skipping reward curves (no training_runs.json).")

    _print_summary(policy_jerk_data)
    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
