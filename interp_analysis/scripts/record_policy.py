#!/usr/bin/env python3
"""Record policy rollouts for policy-output jerk analysis.

Loads a trained checkpoint, runs rollouts in MJX, and records trajectories
including actual motor positions, reference motor positions, velocities,
and torques.

Usage (via the wrapper script from toddlerbot/ directory):
    bash interp_analysis/scripts/run_policy_jerk.sh

Or directly (from toddlerbot/ directory):
    python -m interp_analysis.scripts.record_policy \\
        --run-dir results/toddlerbot_2xc_crawl_rsl_20260222_112733_linear_s0 \\
        --env crawl \\
        --output-dir rollouts/ \\
        --num-episodes 4
"""

import argparse
import json
import os

os.environ["USE_JAX"] = "true"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np


def detect_interp_method(run_dir: str) -> str:
    """Detect interpolation method from a run's args.json.

    Args:
        run_dir: Path to the training run directory.

    Returns:
        Interpolation method name.
    """
    args_path = os.path.join(run_dir, "args.json")
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"args.json not found in {run_dir}")

    with open(args_path) as f:
        args = json.load(f)

    note = args.get("note", "")
    for part in note.split(","):
        if part.strip().startswith("interp="):
            return part.strip().split("=")[1]

    return "linear"


def find_policy_path(run_dir: str) -> str:
    """Find the best model checkpoint in a run directory.

    Args:
        run_dir: Path to the training run directory.

    Returns:
        Path to the best policy checkpoint.

    Raises:
        FileNotFoundError: If no checkpoint is found.
    """
    candidates = [
        os.path.join(run_dir, "model_best.pt"),
        os.path.join(run_dir, "model_best"),
        os.path.join(run_dir, "model_last.pt"),
        os.path.join(run_dir, "model_last"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No model checkpoint found in {run_dir}. "
        f"Looked for: {[os.path.basename(c) for c in candidates]}"
    )


def extract_trajectory_data(states, env) -> dict:
    """Extract motor positions, velocities, torques from rollout states.

    Args:
        states: List of MJX State objects from rollout.
        env: The MJX environment instance.

    Returns:
        Dict with numpy arrays for each trajectory component.
    """
    actual_motor_pos = []
    ref_motor_pos = []
    actual_motor_vel = []
    torques = []

    q_start = int(env.q_start_idx)
    qd_start = int(env.qd_start_idx)
    motor_idx = np.array(env.motor_indices)

    for state in states:
        ps = state.pipeline_state
        actual_motor_pos.append(np.array(ps.q[q_start + motor_idx]))
        actual_motor_vel.append(np.array(ps.qd[qd_start + motor_idx]))
        torques.append(np.array(ps.qfrc_actuator[qd_start + motor_idx]))

        state_ref = state.info.get("state_ref")
        if state_ref is not None and "motor_pos" in state_ref:
            ref_motor_pos.append(np.array(state_ref["motor_pos"]))

    result = {
        "actual_motor_pos": np.stack(actual_motor_pos, axis=0),
        "actual_motor_vel": np.stack(actual_motor_vel, axis=0),
        "torques": np.stack(torques, axis=0),
    }
    if ref_motor_pos:
        if len(ref_motor_pos) != len(actual_motor_pos):
            print(f"  WARNING: ref_motor_pos has {len(ref_motor_pos)} entries "
                  f"vs {len(actual_motor_pos)} actual steps. Trimming.")
        result["ref_motor_pos"] = np.stack(ref_motor_pos, axis=0)

    return result


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Record policy rollouts for jerk analysis"
    )
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Path to the training run directory.",
    )
    parser.add_argument(
        "--env", type=str, default="crawl",
        help="Environment name.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="rollouts",
        help="Output directory for NPZ files.",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=4,
        help="Number of rollout episodes (different seeds).",
    )
    parser.add_argument(
        "--robot", type=str, default="toddlerbot_2xc",
        help="Robot name.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    interp_method = detect_interp_method(args.run_dir)
    print(f"Detected interpolation method: {interp_method}")

    # Point CrawlReference at the correct motion file for this method.
    method_tag = interp_method.replace("_", "")
    motion_file = os.path.join("motion", f"crawl_2xc_{method_tag}.lz4")
    if os.path.exists(motion_file):
        os.environ["MOTION_FILE_OVERRIDE"] = motion_file
        print(f"Using motion file: {motion_file}")
    else:
        print(f"WARNING: {motion_file} not found, using default motion file")

    import gin
    import jax
    import torch

    from toddlerbot.locomotion.mjx_config import MJXConfig
    from toddlerbot.locomotion.mjx_env import get_env_class
    from toddlerbot.locomotion.on_policy_runner import OnPolicyRunner
    from toddlerbot.locomotion.ppo_config import PPOConfig
    from toddlerbot.locomotion.train_mjx import (
        RSLRLWrapper,
        load_jax_ckpt_to_torch,
        load_runner_config,
        rollout,
    )
    from toddlerbot.sim.robot import Robot

    device = "cpu"
    try:
        if torch.cuda.is_available():
            device = "cuda"
    except Exception:
        pass

    _toddlerbot_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    gin_file_path = os.path.join(
        _toddlerbot_root, "toddlerbot", "locomotion",
        f"{args.env}.gin",
    )
    gin.parse_config_file(gin_file_path)

    robot = Robot(args.robot)
    EnvClass = get_env_class(args.env.replace("_fixed", ""))
    env_cfg = MJXConfig()
    train_cfg = PPOConfig()

    env = EnvClass(
        args.env, robot, env_cfg,
        fixed_base="fixed" in args.env,
        add_domain_rand=False,
    )

    policy_path = find_policy_path(args.run_dir)
    print(f"Loading policy from: {policy_path}")

    is_torch = policy_path.endswith(".pt")

    rsl_env = RSLRLWrapper(env, device, train_cfg, eval=True)
    runner_config = load_runner_config(train_cfg)

    if is_torch:
        policy_params = torch.load(policy_path, weights_only=False)
    else:
        from brax.training import model

        jax_params = model.load_params(policy_path)
        policy_params = load_jax_ckpt_to_torch(jax_params)

    runner = OnPolicyRunner(
        rsl_env, runner_config, restore_params=policy_params, device=device,
    )

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    inference_fn = runner.get_inference_policy(device=device)

    os.makedirs(args.output_dir, exist_ok=True)
    run_name = os.path.basename(args.run_dir)
    dt = float(env.dt)

    print(f"\nRecording {args.num_episodes} episodes...")
    for seed in range(args.num_episodes):
        rng = jax.random.PRNGKey(seed)
        states = rollout(
            jit_reset, jit_step, inference_fn,
            train_cfg, is_torch, False, rng,
        )

        data = extract_trajectory_data(states, env)
        n_steps = data["actual_motor_pos"].shape[0]
        n_motors = data["actual_motor_pos"].shape[1]

        output_path = os.path.join(
            args.output_dir, f"{run_name}_rollout_s{seed}.npz"
        )
        np.savez(
            output_path,
            dt=dt,
            interp_method=interp_method,
            n_steps=n_steps,
            n_motors=n_motors,
            **data,
        )
        print(f"  Seed {seed}: {n_steps} steps, {n_motors} motors -> {output_path}")

    print(f"\nDone. Rollouts saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
