"""Interpolated motion reference classes.

Drop-in replacements for CrawlReference and CartwheelReference that
use smooth interpolation between keyframes instead of direct integer
frame indexing.

Usage from the training script:
    import toddlerbot.locomotion.crawl_env as crawl_env_module
    crawl_env_module.CrawlReference = InterpolatedCrawlReference
    # Now CrawlEnv will use interpolated references automatically.
"""

import functools

import jax.numpy as jnp
import numpy as np

from interp_analysis.interpolation.base import KeyframeInterpolator
from interp_analysis.interpolation.cubic_spline import (
    CubicSplineInterpolator,
    SplineCoeffs,
    evaluate_spline,
    precompute_coeffs,
)
from interp_analysis.interpolation.linear import LinearInterpolator
from interp_analysis.interpolation.min_jerk import MinJerkInterpolator
from interp_analysis.interpolation.min_jerk_viapoint import MinJerkViapointInterpolator

# Keys in motion_ref dict that contain per-frame trajectory data
TRAJECTORY_KEYS = [
    "qpos",
    "action",
    "body_pos",
    "body_quat",
    "body_lin_vel",
    "body_ang_vel",
    "site_pos",
    "site_quat",
]


def make_interpolator(method: str) -> KeyframeInterpolator:
    """Create an interpolator instance from a method name.

    Args:
        method: One of "linear", "min_jerk", "min_jerk_viapoint", "cubic_spline".

    Returns:
        KeyframeInterpolator instance.

    Raises:
        ValueError: If method is not recognized.
    """
    methods = {
        "linear": LinearInterpolator,
        "min_jerk": MinJerkInterpolator,
        "min_jerk_viapoint": MinJerkViapointInterpolator,
        "cubic_spline": CubicSplineInterpolator,
    }
    if method not in methods:
        raise ValueError(
            f"Unknown interpolation method '{method}'. "
            f"Choose from: {list(methods.keys())}"
        )
    return methods[method]()


def _interp_field(
    interpolator: KeyframeInterpolator,
    frames: jnp.ndarray,
    t: jnp.ndarray,
    periodic: bool,
    spline_coeffs: "dict[str, SplineCoeffs] | None" = None,
    key: str = "",
) -> jnp.ndarray:
    """Interpolate a single motion field at continuous index t.

    For cubic spline with precomputed coefficients, uses the fast path.
    For linear and min-jerk, calls the standard interpolate method.
    """
    if spline_coeffs is not None and key in spline_coeffs:
        return evaluate_spline(spline_coeffs[key], t)
    return interpolator.interpolate(frames, t, periodic)


def _precompute_spline_coefficients(
    motion_ref: dict,
    keys: list,
    periodic: bool,
) -> dict:
    """Precompute cubic spline coefficients for all trajectory fields.

    Args:
        motion_ref: Dict of motion data arrays (numpy).
        keys: Which keys to precompute for.
        periodic: Whether the motion is cyclic.

    Returns:
        Dict mapping field key to SplineCoeffs.
    """
    coeffs = {}
    for key in keys:
        if key in motion_ref:
            frames_np = np.asarray(motion_ref[key])
            if frames_np.ndim >= 1 and frames_np.shape[0] >= 2:
                coeffs[key] = precompute_coeffs(frames_np, periodic)
    return coeffs


def _interpolate_common_fields(
    interpolator: KeyframeInterpolator,
    motion_ref: dict,
    t: jnp.ndarray,
    periodic: bool,
    spline_coeffs: "dict[str, SplineCoeffs] | None",
    fixed_base: bool,
    q_start_idx: int,
    mj_joint_indices,
) -> dict:
    """Interpolate trajectory fields shared between crawl and cartwheel.

    Returns a dict with qpos, joint_pos, and all body/site fields.
    """
    sc = spline_coeffs
    qpos = _interp_field(interpolator, motion_ref["qpos"], t, periodic, sc, "qpos")
    if fixed_base:
        qpos = qpos[7:]

    return {
        "qpos": qpos,
        "joint_pos": qpos[q_start_idx + mj_joint_indices],
        "body_pos": _interp_field(
            interpolator, motion_ref["body_pos"], t, periodic, sc, "body_pos",
        ),
        "body_quat": _interp_field(
            interpolator, motion_ref["body_quat"], t, periodic, sc, "body_quat",
        ),
        "body_lin_vel": _interp_field(
            interpolator, motion_ref["body_lin_vel"], t, periodic, sc, "body_lin_vel",
        ),
        "body_ang_vel": _interp_field(
            interpolator, motion_ref["body_ang_vel"], t, periodic, sc, "body_ang_vel",
        ),
        "site_pos": _interp_field(
            interpolator, motion_ref["site_pos"], t, periodic, sc, "site_pos",
        ),
        "site_quat": _interp_field(
            interpolator, motion_ref["site_quat"], t, periodic, sc, "site_quat",
        ),
    }


def _create_patched_ref(ref_class, interpolation_method, robot, dt, fixed_base=False):
    """Create an interpolated reference with the given method.

    Use with functools.partial to bind ref_class and interpolation_method,
    producing a callable compatible with the (robot, dt, fixed_base) signature.
    """
    return ref_class(
        robot, dt, fixed_base=fixed_base,
        interpolation_method=interpolation_method,
    )


def apply_monkey_patches(interp_method: str) -> tuple:
    """Patch env modules to use interpolated references.

    Args:
        interp_method: Interpolation method name.

    Returns:
        Tuple of (crawl_module, orig_crawl, cartwheel_module, orig_cartwheel)
        for use with restore_monkey_patches.
    """
    CrawlRefClass = create_interpolated_crawl_ref_class()
    CartwheelRefClass = create_interpolated_cartwheel_ref_class()

    import toddlerbot.locomotion.cartwheel_env as cartwheel_env_module
    import toddlerbot.locomotion.crawl_env as crawl_env_module

    orig_crawl = crawl_env_module.CrawlReference
    orig_cartwheel = cartwheel_env_module.CartwheelReference

    crawl_env_module.CrawlReference = functools.partial(
        _create_patched_ref, CrawlRefClass, interp_method,
    )
    cartwheel_env_module.CartwheelReference = functools.partial(
        _create_patched_ref, CartwheelRefClass, interp_method,
    )

    return crawl_env_module, orig_crawl, cartwheel_env_module, orig_cartwheel


def restore_monkey_patches(patch_state: tuple) -> None:
    """Restore original reference classes from apply_monkey_patches output."""
    crawl_mod, orig_crawl, cartwheel_mod, orig_cartwheel = patch_state
    crawl_mod.CrawlReference = orig_crawl
    cartwheel_mod.CartwheelReference = orig_cartwheel


# ---------------------------------------------------------------------------
# Factory functions: create subclasses with deferred base-class imports.
# The inner classes MUST be defined here because they inherit from toddlerbot
# base classes that require MuJoCo at import time.
# ---------------------------------------------------------------------------


def create_interpolated_crawl_ref_class():
    """Lazily import and create InterpolatedCrawlReference class.

    Returns the class (not an instance). This avoids importing toddlerbot
    at module load time, which requires MuJoCo and other heavy deps.
    """
    from toddlerbot.reference.crawl_ref import CrawlReference
    from toddlerbot.utils.array_utils import array_lib as tb_np

    class InterpolatedCrawlReference(CrawlReference):
        """CrawlReference with configurable keyframe interpolation.

        Overrides get_state_ref to use float indexing + interpolation
        instead of integer indexing for all trajectory fields.
        """

        def __init__(self, robot, dt, fixed_base=False, interpolation_method="linear"):
            self._interpolator = make_interpolator(interpolation_method)
            self._spline_coeffs = None
            super().__init__(robot, dt, fixed_base)
            if interpolation_method == "cubic_spline":
                self._spline_coeffs = _precompute_spline_coefficients(
                    self.motion_ref, TRAJECTORY_KEYS, periodic=True,
                )
            print(f"  - Interpolation: {interpolation_method}")

        def get_state_ref(self, time_curr, command, last_state, init_idx=0):
            t = (init_idx + time_curr / self.dt) % self.n_frames
            fields = _interpolate_common_fields(
                self._interpolator, self.motion_ref, t, True,
                self._spline_coeffs, self.fixed_base,
                self.q_start_idx, self.mj_joint_indices,
            )
            motor_pos = _interp_field(
                self._interpolator, self.motion_ref["action"],
                t, True, self._spline_coeffs, "action",
            )
            path_state = self.integrate_path_state(command, last_state)
            return {
                **path_state,
                **fields,
                "motor_pos": motor_pos,
                "stance_mask": tb_np.ones(2, dtype=tb_np.float32),
            }

    return InterpolatedCrawlReference


def create_interpolated_cartwheel_ref_class():
    """Lazily import and create InterpolatedCartwheelReference class."""
    from toddlerbot.reference.cartwheel_ref import CartwheelReference
    from toddlerbot.utils.array_utils import array_lib as tb_np

    class InterpolatedCartwheelReference(CartwheelReference):
        """CartwheelReference with configurable keyframe interpolation.

        Non-periodic (clamped) interpolation for one-shot motions.
        """

        def __init__(self, robot, dt, fixed_base=False, interpolation_method="linear"):
            self._interpolator = make_interpolator(interpolation_method)
            self._spline_coeffs = None
            super().__init__(robot, dt, fixed_base)
            if interpolation_method == "cubic_spline":
                self._spline_coeffs = _precompute_spline_coefficients(
                    self.motion_ref, TRAJECTORY_KEYS, periodic=False,
                )
            print(f"  - Interpolation: {interpolation_method}")

        def get_state_ref(self, time_curr, command, last_state, init_idx=0):
            t = jnp.clip(init_idx + time_curr / self.dt, 0.0, self.n_frames - 1.0)
            fields = _interpolate_common_fields(
                self._interpolator, self.motion_ref, t, False,
                self._spline_coeffs, self.fixed_base,
                self.q_start_idx, self.mj_joint_indices,
            )
            motor_pos = fields["qpos"][self.q_start_idx + self.mj_motor_indices]
            path_state = self.integrate_path_state(command, last_state)
            return {
                **path_state,
                **fields,
                "motor_pos": motor_pos,
                "stance_mask": tb_np.ones(2, dtype=tb_np.float32),
            }

    return InterpolatedCartwheelReference
