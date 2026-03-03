#!/usr/bin/env python3
"""Visualize interpolated trajectories from sparse keyframes.

Generates a multi-panel figure with representative joints showing
position, velocity, and jerk for all interpolation methods overlaid.
Sparse keyframes are marked as dots for reference.

Usage (from the toddlerbot/ directory):
    python -m interp_analysis.scripts.visualize_interpolation \
        --input motion/crawl_2xc.lz4 \
        --output figures/interpolation_comparison.png

    # Zoom into a specific time window:
    python -m interp_analysis.scripts.visualize_interpolation \
        --input motion/crawl_2xc.lz4 \
        --t-start 4.0 --t-end 8.0
"""

import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np

from interp_analysis.evaluation.visualize import METHOD_COLORS, METHOD_LABELS
from interp_analysis.scripts.generate_motion import (
    METHODS,
    extract_keyframe_arrays,
    interpolate_trajectory,
)

# Motor names for the toddlerbot_2xc (30 motors, order from XML).
MOTOR_NAMES = [
    "neck_yaw_drive", "neck_pitch_act",
    "waist_act_1", "waist_act_2",
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw_drive",
    "left_knee", "left_ankle_roll", "left_ankle_pitch",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw_drive",
    "right_knee", "right_ankle_roll", "right_ankle_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw_drive",
    "left_elbow_roll", "left_elbow_yaw_drive", "left_wrist_pitch_drive",
    "left_wrist_roll",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw_drive",
    "right_elbow_roll", "right_elbow_yaw_drive", "right_wrist_pitch_drive",
    "right_wrist_roll",
]

# Indices of representative joints to plot (chosen for variety of motion).
DEFAULT_JOINT_INDICES = [4, 7, 16, 19]  # L hip pitch, L knee, L shoulder pitch, L elbow


def finite_diff(signal, dt):
    """First-order central finite difference (forward/backward at edges)."""
    deriv = np.zeros_like(signal)
    deriv[1:-1] = (signal[2:] - signal[:-2]) / (2 * dt)
    deriv[0] = (signal[1] - signal[0]) / dt
    deriv[-1] = (signal[-1] - signal[-2]) / dt
    return deriv


def compute_derivatives(positions, dt):
    """Compute velocity (1st derivative) and jerk (3rd derivative)."""
    velocity = finite_diff(positions, dt)
    acceleration = finite_diff(velocity, dt)
    jerk = finite_diff(acceleration, dt)
    return velocity, jerk


def build_figure(
    dense_times, trajectories, kf_times, kf_motor, joint_indices, dt,
    t_start=None, t_end=None,
):
    """Build the multi-panel comparison figure.

    Args:
        dense_times: (T,) time array.
        trajectories: Dict mapping method name -> (T, D) interpolated motor positions.
        kf_times: (K,) keyframe times.
        kf_motor: (K, D) keyframe motor positions.
        joint_indices: List of motor indices to plot.
        dt: Timestep.
        t_start: Optional start time for zoom window.
        t_end: Optional end time for zoom window.

    Returns:
        matplotlib Figure.
    """
    n_joints = len(joint_indices)
    n_rows = 3  # position, velocity, jerk
    fig, axes = plt.subplots(
        n_joints * n_rows, 1,
        figsize=(14, 3.2 * n_joints * n_rows),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    # Time mask for zoom window.
    if t_start is not None or t_end is not None:
        ts = t_start if t_start is not None else dense_times[0]
        te = t_end if t_end is not None else dense_times[-1]
        t_mask = (dense_times >= ts) & (dense_times <= te)
        kf_mask = (kf_times >= ts) & (kf_times <= te)
    else:
        t_mask = np.ones(len(dense_times), dtype=bool)
        kf_mask = np.ones(len(kf_times), dtype=bool)

    t_plot = dense_times[t_mask]

    for j_idx, motor_idx in enumerate(joint_indices):
        joint_name = MOTOR_NAMES[motor_idx] if motor_idx < len(MOTOR_NAMES) else f"joint_{motor_idx}"
        ax_pos = axes[j_idx * n_rows + 0]
        ax_vel = axes[j_idx * n_rows + 1]
        ax_jrk = axes[j_idx * n_rows + 2]

        for method in METHODS:
            color = METHOD_COLORS[method]
            label = METHOD_LABELS[method]
            traj = trajectories[method][:, motor_idx]
            vel, jrk = compute_derivatives(traj, dt)

            ax_pos.plot(t_plot, traj[t_mask], color=color, label=label, linewidth=1.2)
            ax_vel.plot(t_plot, vel[t_mask], color=color, linewidth=1.0, alpha=0.85)
            ax_jrk.plot(t_plot, jrk[t_mask], color=color, linewidth=0.8, alpha=0.75)

        # Sparse keyframes as gray dots on position panel.
        ax_pos.scatter(
            kf_times[kf_mask], kf_motor[kf_mask, motor_idx],
            color="black", s=30, zorder=5, marker="o", label="Keyframes",
        )

        ax_pos.set_ylabel("Position (rad)")
        ax_vel.set_ylabel("Velocity (rad/s)")
        ax_jrk.set_ylabel("Jerk (rad/s\u00b3)")

        ax_pos.set_title(f"{joint_name}", fontsize=11, fontweight="bold", loc="left")
        ax_jrk.set_xlabel("Time (s)")

        # Vertical lines at keyframe times on all 3 panels.
        for t_kf in kf_times[kf_mask]:
            for ax in [ax_pos, ax_vel, ax_jrk]:
                ax.axvline(t_kf, color="gray", linewidth=0.3, alpha=0.5)

        for ax in [ax_pos, ax_vel, ax_jrk]:
            ax.grid(True, alpha=0.3)

    # Single legend at the top.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, 1.0))

    fig.suptitle("Interpolation Method Comparison: Joint Trajectories",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize interpolated trajectories from sparse keyframes"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the original .lz4 motion file.",
    )
    parser.add_argument(
        "--output", type=str, default="figures/interpolation_comparison.png",
        help="Output path for the figure.",
    )
    parser.add_argument(
        "--joints", type=int, nargs="+", default=DEFAULT_JOINT_INDICES,
        help="Motor indices to plot (default: 4 7 16 19).",
    )
    parser.add_argument(
        "--dt", type=float, default=0.02,
        help="Control timestep in seconds (default: 0.02).",
    )
    parser.add_argument(
        "--t-start", type=float, default=None,
        help="Start time for zoom window (default: full range).",
    )
    parser.add_argument(
        "--t-end", type=float, default=None,
        help="End time for zoom window (default: full range).",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Figure DPI (default: 150).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}")
        return

    print(f"Loading {args.input}...")
    motion_data = joblib.load(args.input)
    kf_times, kf_motor, _kf_qpos = extract_keyframe_arrays(motion_data)

    print(f"Sparse keyframes: {len(kf_times)} entries")
    print(f"Time range: {kf_times[0]:.2f}s - {kf_times[-1]:.2f}s")

    dense_times = np.arange(0, kf_times[-1], args.dt)
    print(f"Dense frames: {len(dense_times)} at {1/args.dt:.0f} Hz")

    # Interpolate all methods.
    trajectories = {}
    for method in METHODS:
        print(f"  Interpolating with '{method}'...")
        trajectories[method] = interpolate_trajectory(
            dense_times, kf_times, kf_motor, method
        )

    print(f"Plotting joints: {[MOTOR_NAMES[i] for i in args.joints]}")
    fig = build_figure(
        dense_times, trajectories, kf_times, kf_motor,
        args.joints, args.dt, args.t_start, args.t_end,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved to {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
