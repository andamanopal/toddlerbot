"""Linear (baseline) keyframe interpolation.

Implements standard linear blending between adjacent keyframes.
This serves as the controlled baseline for comparison against
minimum-jerk and cubic spline methods.
"""

import jax.numpy as jnp

from interp_analysis.interpolation.base import KeyframeInterpolator, resolve_index


class LinearInterpolator(KeyframeInterpolator):
    """Linearly interpolate between adjacent keyframes.

    Given frames[idx] and frames[next_idx], computes:
        result = (1 - tau) * frames[idx] + tau * frames[next_idx]

    This is equivalent to numpy.interp generalized to N-d arrays.
    """

    def interpolate(
        self,
        frames: jnp.ndarray,
        t: jnp.ndarray,
        periodic: bool = False,
    ) -> jnp.ndarray:
        n_frames = frames.shape[0]
        idx, next_idx, tau = resolve_index(t, n_frames, periodic)

        f0 = frames[idx]
        f1 = frames[next_idx]

        return f0 + tau * (f1 - f0)
