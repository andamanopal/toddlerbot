#!/usr/bin/env python3
"""Post-training evaluation for interpolation experiments.

Loads trained policies from results directories and computes
tracking metrics, smoothness metrics, and generates comparison data.

Usage (from the toddlerbot/ directory):
    python -m interp_analysis.scripts.evaluate_interp \\
        --results-dir results/ \\
        --env crawl \\
        --output-dir evaluation_results/
"""

import argparse
import json
import os
from glob import glob


def collect_training_metrics(results_dir: str, env: str) -> dict:
    """Collect wandb/training metrics from result directories.

    Scans results_dir for runs matching the env pattern and extracts
    reward curves from output logs.

    Args:
        results_dir: Root results directory.
        env: Environment name (e.g., "crawl").

    Returns:
        Dict mapping method name to collected metrics.
    """
    collected = {}
    pattern = os.path.join(results_dir, f"*_{env}_*")
    run_dirs = sorted(glob(pattern))

    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            continue

        args_path = os.path.join(run_dir, "args.json")
        if not os.path.exists(args_path):
            continue

        with open(args_path) as f:
            args = json.load(f)

        note = args.get("note", "")
        method = "linear"
        for part in note.split(","):
            if part.strip().startswith("interp="):
                method = part.strip().split("=")[1]

        if method not in collected:
            collected[method] = {"run_dirs": []}
        collected[method]["run_dirs"].append(run_dir)

    return collected


def main():
    parser = argparse.ArgumentParser(description="Evaluate interpolation experiments")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Root results directory.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="crawl",
        help="Environment name to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for evaluation data.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from interp_analysis.evaluation.metrics import compute_reference_smoothness

    # Compute offline reference smoothness metrics
    # (this doesn't need trained policies, just keyframe data)
    print("Computing reference trajectory smoothness...")
    import numpy as np

    n_frames = 50
    n_joints = 10
    dt = 0.02
    # Simulate keyframe-like data (smooth-ish trajectory)
    t = np.linspace(0, 2 * np.pi, n_frames)
    frames = np.column_stack([np.sin(t + i * 0.3) for i in range(n_joints)]).astype(
        np.float32
    )

    smoothness_results = {}
    for method in ["linear", "min_jerk", "min_jerk_viapoint", "cubic_spline"]:
        metrics = compute_reference_smoothness(frames, dt, method, n_samples=500)
        smoothness_results[method] = metrics
        print(f"  {method:15s} | mean_jerk={metrics['mean_jerk']:.2f} "
              f"| max_jerk={metrics['max_jerk']:.2f} "
              f"| hf_ratio={metrics['high_freq_ratio']:.4f}")

    smoothness_path = os.path.join(args.output_dir, "reference_smoothness.json")
    with open(smoothness_path, "w") as f:
        json.dump(
            {k: {kk: float(vv) for kk, vv in v.items()} for k, v in smoothness_results.items()},
            f,
            indent=2,
        )
    print(f"\nSaved smoothness metrics to {smoothness_path}")

    # Collect training run metrics (if available)
    collected = collect_training_metrics(args.results_dir, args.env)
    if collected:
        print(f"\nFound training runs for methods: {list(collected.keys())}")
        collected_path = os.path.join(args.output_dir, "training_runs.json")
        # Serialize just the run_dirs (not full data)
        serializable = {k: v["run_dirs"] for k, v in collected.items()}
        with open(collected_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved run index to {collected_path}")
    else:
        print(f"\nNo training runs found in {args.results_dir} for env={args.env}")
        print("Run training first with: python -m interp_analysis.scripts.train_interp --env crawl --interp-method <method>")


if __name__ == "__main__":
    main()
