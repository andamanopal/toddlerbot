"""Crawling motion reference implementation for toddlerbot.

Provides crawling motion references using keyframe-based motion data.
"""

import os
from typing import Dict, Tuple

import jax
import joblib

from toddlerbot.reference.motion_ref import MotionReference
from toddlerbot.utils.array_utils import ArrayType
from toddlerbot.utils.array_utils import array_lib as np


class CrawlReference(MotionReference):
    """Motion reference for crawling movements using keyframe data."""

    def __init__(self, robot, dt: float, fixed_base: bool = False):
        """Initialize crawling motion reference.

        Args:
            robot: Robot instance for motion generation.
            dt: Time step for motion reference.
            fixed_base: Whether to use fixed base mode.
        """
        super().__init__("crawl", "keyframe", robot, dt, fixed_base)

        # Load crawling motion data (env var override for interpolation experiments)
        robot_suffix = "_2xc" if "2xc" in robot.name else "_2xm"
        motion_file_path = os.environ.get(
            "MOTION_FILE_OVERRIDE",
            os.path.join("motion", f"crawl{robot_suffix}.lz4"),
        )
        self.motion_ref = joblib.load(motion_file_path)

        # Extract basic data
        self.time_array = np.array(self.motion_ref["time"])
        self.n_frames = len(self.time_array)

        # Check if motion data is in robot-relative frame (default: False for backward compatibility)
        self.is_robot_relative_frame = self.motion_ref.get(
            "is_robot_relative_frame", False
        )

        print("\n=== MOTION DATA ===")
        print(f"  - path         : {motion_file_path}")
        print(f"  - n_frames     : {self.n_frames}")
        print(f"  - qpos         : {self.motion_ref['qpos'].shape}")
        print(f"  - action       : {self.motion_ref['action'].shape}")
        print(f"  - body_pos     : {self.motion_ref['body_pos'].shape}")
        print(f"  - body_quat    : {self.motion_ref['body_quat'].shape}")
        print(f"  - body_lin_vel : {self.motion_ref['body_lin_vel'].shape}")
        print(f"  - body_ang_vel : {self.motion_ref['body_ang_vel'].shape}")
        print(f"  - site_pos     : {self.motion_ref['site_pos'].shape}")
        print(f"  - site_quat    : {self.motion_ref['site_quat'].shape}")
        print("  - Data format: New (separate pos/quat arrays)")

        frame_type = "Robot-relative" if self.is_robot_relative_frame else "Global"
        print(f"  - Frame reference: {frame_type}")
        if self.is_robot_relative_frame:
            print("  - Global torso position: Available in qpos[:3]")

        if self.use_jax:
            # Keep large arrays on CPU, only convert small arrays to JAX
            self.time_array = jax.device_put(self.time_array)
            # Convert only the numeric arrays to JAX, not the entire dictionary
            # This avoids issues with string keys/values that JAX cannot handle
            jax_motion_ref = {}
            for key, value in self.motion_ref.items():
                try:
                    # Try to convert to JAX array - only works for numeric data
                    if isinstance(value, (str, bool)) or value is None:
                        # Skip strings, booleans, and None values
                        jax_motion_ref[key] = value
                    else:
                        # Attempt JAX conversion for numeric data
                        jax_motion_ref[key] = jax.device_put(value)
                except (TypeError, ValueError):
                    # If conversion fails, keep original value
                    jax_motion_ref[key] = value
            self.motion_ref = jax_motion_ref

    def get_phase_signal(self, time_curr: float, init_idx: int = 0) -> ArrayType:
        """Get the phase signal for the current time."""
        # Calculate the index based on time and init_idx
        time_idx = np.floor(time_curr / self.dt).astype(np.int32)
        total_idx = (init_idx + time_idx) % self.n_frames

        # Calculate phase based on total_idx
        phase = (total_idx / self.n_frames) * 2 * np.pi
        phase_signal = np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)

        return phase_signal

    def get_vel(self, command: ArrayType) -> Tuple[ArrayType, ArrayType]:
        """Get the desired linear and angular velocities."""
        # Set to zero for now
        lin_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        ang_vel = np.zeros(3, dtype=np.float32)  # No rotation
        return lin_vel, ang_vel

    def get_state_ref(
        self,
        time_curr: float,
        command: ArrayType,
        last_state: Dict[str, ArrayType],
        init_idx: int = 0,
    ) -> Dict[str, ArrayType]:
        """Get the reference state for the current time. Supports RIS if fed init_idx

        Args:
            time_curr (float): The current time.
            command (ArrayType): Command inputs for the robot's movement.
            last_state (Dict[str, ArrayType]): The last state of the robot.
            init_idx (int, optional): Starting initial state index for RIS. Defaults to 0.

        Returns:
            Dict[str, ArrayType]: A dictionary containing the path state, motor positions, joint positions, body poses, and other reference data.
        """
        # Calculate the index based on time and init_idx
        time_idx = np.floor(time_curr / self.dt).astype(np.int32)
        total_idx = (init_idx + time_idx) % self.n_frames

        # Get reference qpos from keyframes
        qpos = self.motion_ref["qpos"][total_idx]
        if self.fixed_base:
            qpos = qpos[7:]  # Skip first 7 elements for fixed base

        joint_pos = qpos[self.q_start_idx + self.mj_joint_indices]
        # Get motor positions from action data
        motor_pos = self.motion_ref["action"][total_idx]

        # Get body poses (robot-relative frame)
        body_pos = self.motion_ref["body_pos"][total_idx]
        body_quat = self.motion_ref["body_quat"][total_idx]
        body_lin_vel = self.motion_ref["body_lin_vel"][total_idx]
        body_ang_vel = self.motion_ref["body_ang_vel"][total_idx]

        # Get reference site poses (robot-relative frame)
        site_pos = self.motion_ref["site_pos"][total_idx]
        site_quat = self.motion_ref["site_quat"][total_idx]

        # For crawling, both feet are typically in contact
        stance_mask = np.ones(2, dtype=np.float32)

        # Get path state using the new interface
        path_state = self.integrate_path_state(command, last_state)

        return {
            **path_state,
            "motor_pos": motor_pos,
            "joint_pos": joint_pos,
            "qpos": qpos,
            "body_pos": body_pos,
            "body_quat": body_quat,
            "body_lin_vel": body_lin_vel,
            "body_ang_vel": body_ang_vel,
            "site_pos": site_pos,
            "site_quat": site_quat,
            "stance_mask": stance_mask,
        }
