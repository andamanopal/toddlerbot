#!/usr/bin/env python3
"""Training wrapper for interpolation comparison experiments.

Sets the motion file path to a pre-generated .lz4 file (produced by
generate_motion.py), then delegates to the standard train_mjx pipeline.

Usage (from the toddlerbot/ directory):
    python -m interp_analysis.scripts.train_interp \\
        --env crawl \\
        --interp-method min_jerk \\
        --seed 0

This script:
1. Resolves the motion file path for the chosen interpolation method
2. Sets MOTION_FILE_OVERRIDE so CrawlReference loads the right file
3. Seeds all RNGs for reproducibility
4. Injects interp method and seed into the run name
5. Runs the standard train_mjx.main() with remaining args
"""

import argparse
import functools
import os
import random
import sys
import time

os.environ["USE_JAX"] = "true"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np

METHODS = ["linear", "min_jerk", "min_jerk_viapoint", "cubic_spline"]

METHOD_TAGS = {
    "linear": "linear",
    "min_jerk": "minjerk",
    "min_jerk_viapoint": "minjerkviapoint",
    "cubic_spline": "cubicspline",
}


def _strftime_with_suffix(original_fn, suffix, fmt, *args, **kwargs):
    """Patched strftime that appends a suffix to run-name timestamps."""
    result = original_fn(fmt, *args, **kwargs)
    if fmt == "%Y%m%d_%H%M%S":
        return f"{result}_{suffix}"
    return result


def _parse_interp_args():
    """Pre-parse interpolation-specific args before passing rest to train_mjx."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--interp-method",
        type=str,
        default="linear",
        choices=METHODS,
        help="Keyframe interpolation method.",
    )
    pre_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    pre_parser.add_argument(
        "--no-regularization",
        action="store_true",
        default=False,
        help="Disable regularization rewards (motor_torque, energy, action_rate).",
    )
    pre_parser.add_argument(
        "--motion-dir",
        type=str,
        default="motion",
        help="Directory containing the generated .lz4 motion files.",
    )
    return pre_parser.parse_known_args()


def _merge_gin_config(remaining, seed, no_reg):
    """Merge seed and regularization settings into gin-config CLI args."""
    gin_parts = [f"PPOConfig.seed = {seed}"]
    if no_reg:
        gin_parts.append("RewardsConfig.add_regularization = False")
    seed_gin = "; ".join(gin_parts)

    existing_gin = ""
    new_remaining = []
    i = 0
    while i < len(remaining):
        if remaining[i] == "--gin-config" and i + 1 < len(remaining):
            existing_gin = remaining[i + 1]
            i += 2
        else:
            new_remaining.append(remaining[i])
            i += 1

    combined_gin = f"{existing_gin}; {seed_gin}" if existing_gin else seed_gin
    new_remaining.extend(["--gin-config", combined_gin])
    return new_remaining


def main():
    our_args, remaining = _parse_interp_args()

    interp_method = our_args.interp_method
    seed = our_args.seed
    no_reg = our_args.no_regularization
    reg_tag = "_noreg" if no_reg else ""
    method_tag = METHOD_TAGS[interp_method]

    # Resolve the motion file for this method.
    motion_file = os.path.join(
        our_args.motion_dir, f"crawl_2xc_{method_tag}.lz4"
    )
    if not os.path.exists(motion_file):
        print(f"Error: motion file not found: {motion_file}")
        print(f"Run generate_motion.py --all first to create it.")
        sys.exit(1)

    os.environ["MOTION_FILE_OVERRIDE"] = motion_file

    print(f"\n{'='*60}")
    print(f"  Interpolation Experiment")
    print(f"  Method:         {interp_method}")
    print(f"  Motion file:    {motion_file}")
    print(f"  Seed:           {seed}")
    print(f"  Regularization: {'DISABLED' if no_reg else 'enabled (default)'}")
    print(f"{'='*60}\n")

    # Seed all RNGs for reproducibility.
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    # Patch strftime to inject interp method + seed into run name.
    suffix = f"{method_tag}{reg_tag}_s{seed}"
    original_strftime = time.strftime
    time.strftime = functools.partial(_strftime_with_suffix, original_strftime, suffix)

    # Merge gin-config with seed and regularization settings.
    remaining = _merge_gin_config(remaining, seed, no_reg)

    if not any(a.startswith("--note") for a in remaining):
        note_parts = [f"interp={interp_method}", f"seed={seed}"]
        if no_reg:
            note_parts.append("noreg=true")
        remaining.extend(["--note", ",".join(note_parts)])

    # Delegate to the standard training pipeline.
    sys.argv = [sys.argv[0]] + remaining
    from toddlerbot.locomotion.train_mjx import main as train_main

    try:
        train_main()
    finally:
        time.strftime = original_strftime


if __name__ == "__main__":
    main()
