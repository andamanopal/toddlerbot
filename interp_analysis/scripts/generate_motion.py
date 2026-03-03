#!/usr/bin/env python3
"""Generate dense motion files from sparse keyframes using different interpolation.

Reads the embedded sparse keyframes and timed sequence from the original
crawl .lz4 file, re-interpolates the trajectory using the specified method,
runs MuJoCo forward kinematics for body/site data, and saves a new .lz4 file
compatible with the unmodified training pipeline.

This mirrors the original keyframe editor pipeline exactly — the only
difference is the interpolation method used between keyframes.

Usage (from the toddlerbot/ directory):
    python -m interp_analysis.scripts.generate_motion \\
        --input motion/crawl_2xc.lz4 \\
        --method min_jerk \\
        --output motion/crawl_2xc_minjerk.lz4

    # Generate all methods at once:
    python -m interp_analysis.scripts.generate_motion \\
        --input motion/crawl_2xc.lz4 \\
        --all
"""

import argparse
import os

import joblib
import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

METHODS = ["linear", "min_jerk", "min_jerk_viapoint", "cubic_spline"]

# End-effector sites recorded by the keyframe editor (in this order).
SITE_NAMES = [
    "left_hand_center",
    "left_foot_center",
    "right_hand_center",
    "right_foot_center",
]


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def min_jerk_profile(tau):
    """Min-jerk rest-to-rest blend profile: 10t^3 - 15t^4 + 6t^5.

    Returns 0 at tau=0, 1 at tau=1, with zero velocity and acceleration
    at both endpoints.
    """
    tau = np.clip(tau, 0.0, 1.0)
    return tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau ** 2)


def catmull_rom_velocities(kf_times, kf_values):
    """Estimate velocities at non-uniformly spaced keyframes using Catmull-Rom.

    At interior points, the velocity is the average of the left and right
    finite-difference slopes. At endpoints, a one-sided difference is used.

    Args:
        kf_times: (K,) keyframe times.
        kf_values: (K, D) keyframe values.

    Returns:
        (K, D) velocity estimates.
    """
    n_kf = len(kf_times)
    velocities = np.zeros_like(kf_values)

    for i in range(n_kf):
        if i == 0:
            dt = kf_times[1] - kf_times[0]
            velocities[i] = (kf_values[1] - kf_values[0]) / max(dt, 1e-12)
        elif i == n_kf - 1:
            dt = kf_times[-1] - kf_times[-2]
            velocities[i] = (kf_values[-1] - kf_values[-2]) / max(dt, 1e-12)
        else:
            dt_bwd = kf_times[i] - kf_times[i - 1]
            dt_fwd = kf_times[i + 1] - kf_times[i]
            v_bwd = (kf_values[i] - kf_values[i - 1]) / max(dt_bwd, 1e-12)
            v_fwd = (kf_values[i + 1] - kf_values[i]) / max(dt_fwd, 1e-12)
            velocities[i] = 0.5 * (v_bwd + v_fwd)

    return velocities


def interpolate_trajectory(dense_times, kf_times, kf_values, method):
    """Interpolate keyframe values at dense time points.

    Args:
        dense_times: (T,) output time points.
        kf_times: (K,) keyframe arrival times.
        kf_values: (K, D) keyframe values.
        method: One of METHODS.

    Returns:
        (T, D) interpolated values as float32.
    """
    n_dense = len(dense_times)
    n_dim = kf_values.shape[1]
    result = np.zeros((n_dense, n_dim), dtype=np.float32)
    n_seg = len(kf_times) - 1

    if method == "linear":
        for d in range(n_dim):
            result[:, d] = np.interp(dense_times, kf_times, kf_values[:, d])

    elif method == "min_jerk":
        for i in range(n_seg):
            mask = _segment_mask(dense_times, kf_times, i, n_seg)
            if not np.any(mask):
                continue
            h = kf_times[i + 1] - kf_times[i]
            if h < 1e-12:
                result[mask] = kf_values[i]
                continue
            tau = (dense_times[mask] - kf_times[i]) / h
            alpha = min_jerk_profile(tau)[:, None]
            result[mask] = kf_values[i] + alpha * (kf_values[i + 1] - kf_values[i])

    elif method == "cubic_spline":
        for d in range(n_dim):
            cs = CubicSpline(kf_times, kf_values[:, d], bc_type="natural")
            result[:, d] = cs(dense_times).astype(np.float32)

    elif method == "min_jerk_viapoint":
        velocities = catmull_rom_velocities(kf_times, kf_values)

        for i in range(n_seg):
            mask = _segment_mask(dense_times, kf_times, i, n_seg)
            if not np.any(mask):
                continue
            h = kf_times[i + 1] - kf_times[i]
            if h < 1e-12:
                result[mask] = kf_values[i]
                continue

            tau = ((dense_times[mask] - kf_times[i]) / h)[:, None]
            x0 = kf_values[i]
            x1 = kf_values[i + 1]
            # Scale velocities from physical time to normalized [0,1] domain.
            v0 = velocities[i] * h
            v1 = velocities[i + 1] * h
            dx = x1 - x0

            # 5th-order polynomial coefficients (a0=x0, a1=v0, a2=0 from x''(0)=0)
            a3 = 10.0 * dx - 6.0 * v0 - 4.0 * v1
            a4 = -15.0 * dx + 8.0 * v0 + 7.0 * v1
            a5 = 6.0 * dx - 3.0 * v0 - 3.0 * v1

            result[mask] = (
                x0
                + v0 * tau
                + a3 * tau ** 3
                + a4 * tau ** 4
                + a5 * tau ** 5
            )

    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: {METHODS}")

    return result


def _segment_mask(dense_times, kf_times, seg_idx, n_seg):
    """Boolean mask selecting dense times that fall within segment seg_idx."""
    t0 = kf_times[seg_idx]
    t1 = kf_times[seg_idx + 1]
    if seg_idx < n_seg - 1:
        return (dense_times >= t0) & (dense_times < t1)
    return (dense_times >= t0) & (dense_times <= t1)


# ---------------------------------------------------------------------------
# Forward Kinematics
# ---------------------------------------------------------------------------


def mat_to_quat_wxyz(mat):
    """Convert a 3x3 rotation matrix to a wxyz quaternion (MuJoCo convention)."""
    return Rotation.from_matrix(mat).as_quat(scalar_first=True).astype(np.float32)


def compute_forward_kinematics(model_path, qpos_traj, dt):
    """Run MuJoCo forward kinematics for each frame.

    Sets qpos and computes qvel via finite differences, then calls mj_forward
    to populate body positions, orientations, velocities, and site data.

    Args:
        model_path: Path to the MuJoCo XML model.
        qpos_traj: (T, nq) generalized positions.
        dt: Timestep between frames.

    Returns:
        Tuple of (body_pos, body_quat, body_lin_vel, body_ang_vel,
                  site_pos, site_quat), all as float32 arrays.
    """
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    n_frames = qpos_traj.shape[0]
    n_bodies = model.nbody
    n_sites = len(SITE_NAMES)

    body_pos = np.zeros((n_frames, n_bodies, 3), dtype=np.float32)
    body_quat = np.zeros((n_frames, n_bodies, 4), dtype=np.float32)
    body_lin_vel = np.zeros((n_frames, n_bodies, 3), dtype=np.float32)
    body_ang_vel = np.zeros((n_frames, n_bodies, 3), dtype=np.float32)
    site_pos = np.zeros((n_frames, n_sites, 3), dtype=np.float32)
    site_quat = np.zeros((n_frames, n_sites, 4), dtype=np.float32)

    # Resolve site IDs once.
    site_ids = []
    for name in SITE_NAMES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise ValueError(f"Site '{name}' not found in model")
        site_ids.append(sid)

    for t in range(n_frames):
        data.qpos[:] = qpos_traj[t]

        if t > 0:
            mujoco.mj_differentiatePos(
                model, data.qvel, dt, qpos_traj[t - 1], qpos_traj[t]
            )
        else:
            data.qvel[:] = 0.0

        mujoco.mj_forward(model, data)

        body_pos[t] = data.xpos
        body_quat[t] = data.xquat
        body_lin_vel[t] = data.cvel[:, 3:]
        body_ang_vel[t] = data.cvel[:, :3]

        for s, sid in enumerate(site_ids):
            site_pos[t, s] = data.site_xpos[sid]
            xmat = data.site_xmat[sid].reshape(3, 3)
            site_quat[t, s] = mat_to_quat_wxyz(xmat)

    return body_pos, body_quat, body_lin_vel, body_ang_vel, site_pos, site_quat


def transform_to_relative_frame(
    body_pos, body_quat, body_lin_vel, body_ang_vel, site_pos, site_quat
):
    """Transform world-frame FK data to robot-relative frame (torso = body 1).

    Mirrors the transformation in the keyframe editor tool.

    Returns the same tuple of arrays, transformed in-place.
    """
    n_frames = body_pos.shape[0]

    for t in range(n_frames):
        torso_pos = body_pos[t, 1].copy()
        torso_quat_wxyz = body_quat[t, 1].copy()

        # scipy expects xyzw
        torso_rot = Rotation.from_quat(
            [torso_quat_wxyz[1], torso_quat_wxyz[2],
             torso_quat_wxyz[3], torso_quat_wxyz[0]]
        )
        torso_inv = torso_rot.inv()

        # Transform bodies
        for b in range(body_pos.shape[1]):
            body_pos[t, b] = torso_inv.apply(body_pos[t, b] - torso_pos)
            bq = body_quat[t, b]
            body_rot = Rotation.from_quat([bq[1], bq[2], bq[3], bq[0]])
            rel_rot = body_rot * torso_inv
            body_quat[t, b] = rel_rot.as_quat(scalar_first=True)
            body_lin_vel[t, b] = torso_inv.apply(body_lin_vel[t, b])
            body_ang_vel[t, b] = torso_inv.apply(body_ang_vel[t, b])

        # Transform sites
        for s in range(site_pos.shape[1]):
            site_pos[t, s] = torso_inv.apply(site_pos[t, s] - torso_pos)
            sq = site_quat[t, s]
            site_rot = Rotation.from_quat([sq[1], sq[2], sq[3], sq[0]])
            rel_rot = site_rot * torso_inv
            site_quat[t, s] = rel_rot.as_quat(scalar_first=True)

    return body_pos, body_quat, body_lin_vel, body_ang_vel, site_pos, site_quat


# ---------------------------------------------------------------------------
# Motion file generation
# ---------------------------------------------------------------------------


def extract_keyframe_arrays(motion_data):
    """Extract keyframe times and value arrays from the motion file metadata.

    Args:
        motion_data: Dict loaded from the .lz4 file.

    Returns:
        Tuple of (kf_times, kf_motor, kf_qpos):
            kf_times: (K,) arrival times.
            kf_motor: (K, n_motors) motor positions.
            kf_qpos: (K, nq) generalized positions.
    """
    keyframes = motion_data["keyframes"]
    timed_sequence = motion_data["timed_sequence"]

    kf_lookup = {kf["name"]: kf for kf in keyframes}

    kf_times = np.array([t for _, t in timed_sequence], dtype=np.float64)
    kf_motor = np.array(
        [kf_lookup[name]["motor_pos"] for name, _ in timed_sequence],
        dtype=np.float32,
    )
    kf_qpos = np.array(
        [kf_lookup[name]["qpos"] for name, _ in timed_sequence],
        dtype=np.float32,
    )

    return kf_times, kf_motor, kf_qpos


def generate_dense_motion(input_path, method, output_path, robot_name, dt=0.02):
    """Generate a dense motion .lz4 file using the specified interpolation method.

    Mirrors the original keyframe editor pipeline:
    1. Extract sparse keyframes and timed sequence from the original file.
    2. Interpolate motor positions and qpos at the control rate.
    3. Run MuJoCo FK for body/site data.
    4. Save in the same format as the original.

    Args:
        input_path: Path to the original .lz4 motion file.
        method: Interpolation method name.
        output_path: Where to save the new .lz4 file.
        robot_name: Robot name (e.g., "toddlerbot_2xc").
        dt: Control timestep (default 0.02s = 50 Hz).
    """
    original = joblib.load(input_path)
    kf_times, kf_motor, kf_qpos = extract_keyframe_arrays(original)
    is_relative = original.get("is_robot_relative_frame", False)

    print(f"  Sparse keyframes: {len(kf_times)} entries, "
          f"{len(original['keyframes'])} unique poses")
    print(f"  Time range: {kf_times[0]:.2f}s – {kf_times[-1]:.2f}s")
    print(f"  Robot-relative frame: {is_relative}")

    # Dense time array (matching the original tool: np.arange(0, end, dt))
    dense_times = np.arange(0, kf_times[-1], dt)

    print(f"  Interpolating {len(dense_times)} frames with '{method}'...")
    action = interpolate_trajectory(dense_times, kf_times, kf_motor, method)
    qpos = interpolate_trajectory(dense_times, kf_times, kf_qpos, method)

    # Renormalize base quaternion (qpos[:, 3:7]) after interpolation.
    quat_norm = np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
    qpos[:, 3:7] /= np.maximum(quat_norm, 1e-8)

    # Run MuJoCo forward kinematics.
    model_path = os.path.join(
        "toddlerbot", "descriptions", robot_name, "scene.xml"
    )
    print(f"  Running forward kinematics ({len(dense_times)} frames)...")
    fk_results = compute_forward_kinematics(model_path, qpos, dt)
    body_pos, body_quat, body_lin_vel, body_ang_vel, site_pos, site_quat = fk_results

    if is_relative:
        print("  Transforming to robot-relative frame...")
        body_pos, body_quat, body_lin_vel, body_ang_vel, site_pos, site_quat = (
            transform_to_relative_frame(
                body_pos, body_quat, body_lin_vel, body_ang_vel,
                site_pos, site_quat,
            )
        )

    # Build output dict (same structure as the keyframe editor saves).
    result = {
        "time": dense_times,
        "qpos": qpos,
        "action": action,
        "body_pos": body_pos,
        "body_quat": body_quat,
        "body_lin_vel": body_lin_vel,
        "body_ang_vel": body_ang_vel,
        "site_pos": site_pos,
        "site_quat": site_quat,
        "keyframes": original["keyframes"],
        "timed_sequence": original["timed_sequence"],
        "is_robot_relative_frame": is_relative,
        "interp_method": method,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    joblib.dump(result, output_path, compress="lz4")
    print(f"  Saved {len(dense_times)} frames to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate dense motion files with different interpolation methods"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the original .lz4 motion file (e.g., motion/crawl_2xc.lz4).",
    )
    parser.add_argument(
        "--method", type=str, choices=METHODS,
        help="Interpolation method to use.",
    )
    parser.add_argument(
        "--output", type=str,
        help="Output path for the new .lz4 file.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate motion files for all interpolation methods.",
    )
    parser.add_argument(
        "--robot", type=str, default="toddlerbot_2xc",
        help="Robot name (default: toddlerbot_2xc).",
    )
    parser.add_argument(
        "--dt", type=float, default=0.02,
        help="Control timestep in seconds (default: 0.02).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}")
        return

    if args.all:
        base, ext = os.path.splitext(args.input)
        for method in METHODS:
            tag = method.replace("_", "")
            output = f"{base}_{tag}{ext}"
            print(f"\n{'='*60}")
            print(f"  Method: {method}")
            print(f"{'='*60}")
            generate_dense_motion(args.input, method, output, args.robot, args.dt)
    elif args.method:
        output = args.output
        if not output:
            base, ext = os.path.splitext(args.input)
            tag = args.method.replace("_", "")
            output = f"{base}_{tag}{ext}"
        print(f"\n{'='*60}")
        print(f"  Method: {args.method}")
        print(f"{'='*60}")
        generate_dense_motion(args.input, args.method, output, args.robot, args.dt)
    else:
        parser.error("Specify --method or --all")

    print("\nDone.")


if __name__ == "__main__":
    main()
