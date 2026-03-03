"""Via-point minimum-jerk keyframe interpolation.

Implements a 5th-order polynomial per segment with estimated non-zero
velocities at keyframes, unlike the rest-to-rest ``min_jerk`` formulation
that forces zero velocity at every boundary.

Velocity estimation uses Catmull-Rom central differences:
    Interior:     v[i] = (frames[i+1] - frames[i-1]) / 2
    Periodic:     wraps around via jnp.roll
    Non-periodic: forward/backward differences at boundaries

Per-segment polynomial (tau in [0, 1]):
    x(tau) = x0 + v0*tau + a3*tau^3 + a4*tau^4 + a5*tau^5

where:
    dx = x1 - x0
    a3 = 10*dx - 6*v0 - 4*v1
    a4 = -15*dx + 8*v0 + 7*v1
    a5 = 6*dx - 3*v0 - 3*v1

This satisfies position continuity (x(0)=x0, x(1)=x1), velocity
continuity (x'(0)=v0, x'(1)=v1), and zero acceleration at segment
endpoints (x''(0)=0, x''(1)=0).
"""

import jax.numpy as jnp

from interp_analysis.interpolation.base import KeyframeInterpolator, resolve_index


def _estimate_velocities(
    frames: jnp.ndarray,
    periodic: bool,
) -> jnp.ndarray:
    """Estimate keyframe velocities using Catmull-Rom central differences.

    Args:
        frames: Keyframe array of shape (N, ...).
        periodic: If True, uses periodic wrapping for boundary velocities.

    Returns:
        Velocity array of shape (N, ...), same shape as frames.
    """
    n = frames.shape[0]

    if periodic:
        prev = jnp.roll(frames, 1, axis=0)
        nxt = jnp.roll(frames, -1, axis=0)
        return (nxt - prev) / 2.0

    # Non-periodic: central differences for interior, one-sided at boundaries
    prev = jnp.concatenate([frames[:1], frames[:-1]], axis=0)
    nxt = jnp.concatenate([frames[1:], frames[-1:]], axis=0)
    velocities = (nxt - prev) / 2.0

    # Boundary corrections: forward difference at start, backward at end
    v_start = frames[1] - frames[0]
    v_end = frames[-1] - frames[-2]
    velocities = velocities.at[0].set(v_start)
    velocities = velocities.at[-1].set(v_end)

    return velocities


class MinJerkViapointInterpolator(KeyframeInterpolator):
    """Via-point minimum-jerk interpolation between keyframes.

    Unlike the rest-to-rest MinJerkInterpolator, this formulation estimates
    non-zero velocities at keyframes using Catmull-Rom central differences,
    producing smoother trajectories for cyclic motions.

    No precomputation is needed --- velocities are computed inline from the
    frames array at O(N) cost, where N is the number of keyframes.
    """

    def interpolate(
        self,
        frames: jnp.ndarray,
        t: jnp.ndarray,
        periodic: bool = False,
    ) -> jnp.ndarray:
        n_frames = frames.shape[0]
        idx, next_idx, tau = resolve_index(t, n_frames, periodic)

        velocities = _estimate_velocities(frames, periodic)

        x0 = frames[idx]
        x1 = frames[next_idx]
        v0 = velocities[idx]
        v1 = velocities[next_idx]

        dx = x1 - x0

        # 5th-order polynomial coefficients (a0=x0, a1=v0, a2=0)
        a3 = 10.0 * dx - 6.0 * v0 - 4.0 * v1
        a4 = -15.0 * dx + 8.0 * v0 + 7.0 * v1
        a5 = 6.0 * dx - 3.0 * v0 - 3.0 * v1

        tau2 = tau * tau
        tau3 = tau2 * tau
        tau4 = tau3 * tau
        tau5 = tau4 * tau

        return x0 + v0 * tau + a3 * tau3 + a4 * tau4 + a5 * tau5
