"""Minimum-jerk keyframe interpolation.

Implements piecewise minimum-jerk blending between adjacent keyframes.
The minimum-jerk profile minimizes the integral of squared jerk (third
derivative of position), producing trajectories that match human motor
control patterns (Flash & Hogan, 1985).

Between two keyframes, the blending weight follows:
    alpha(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5

This 5th-order polynomial satisfies:
    alpha(0) = 0,  alpha(1) = 1
    alpha'(0) = 0, alpha'(1) = 0
    alpha''(0) = 0, alpha''(1) = 0

Ensuring zero velocity and acceleration at keyframe boundaries.
"""

import jax.numpy as jnp

from interp_analysis.interpolation.base import KeyframeInterpolator, resolve_index


def min_jerk_profile(tau: jnp.ndarray) -> jnp.ndarray:
    """Compute the minimum-jerk blending weight for parameter tau in [0, 1].

    Args:
        tau: Local parameter in [0, 1].

    Returns:
        Blending weight in [0, 1] following the minimum-jerk profile.
    """
    tau2 = tau * tau
    tau3 = tau2 * tau
    return tau3 * (10.0 - 15.0 * tau + 6.0 * tau2)


class MinJerkInterpolator(KeyframeInterpolator):
    """Interpolate between adjacent keyframes using minimum-jerk profile.

    Compared to linear interpolation, this produces:
    - Zero velocity at keyframe boundaries (smooth starts/stops)
    - Zero acceleration at keyframe boundaries (no sudden force changes)
    - Biologically plausible motion profiles
    """

    def interpolate(
        self,
        frames: jnp.ndarray,
        t: jnp.ndarray,
        periodic: bool = False,
    ) -> jnp.ndarray:
        n_frames = frames.shape[0]
        idx, next_idx, tau = resolve_index(t, n_frames, periodic)

        alpha = min_jerk_profile(tau)

        f0 = frames[idx]
        f1 = frames[next_idx]

        return f0 + alpha * (f1 - f0)
