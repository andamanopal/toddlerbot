"""Abstract base class and utilities for keyframe interpolation.

All interpolators share a common index-resolution step that converts a
continuous time index into a segment index, next-segment index, and local
parameter tau in [0, 1].
"""

from abc import ABC, abstractmethod

import jax.numpy as jnp


def resolve_index(
    t: jnp.ndarray,
    n_frames: int,
    periodic: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert a continuous index to segment indices and local parameter.

    Requires at least 2 keyframes.

    Args:
        t: Continuous frame index (scalar or batch). For example 3.7 means
           70% of the way from frame 3 to frame 4.
        n_frames: Total number of keyframes. Must be >= 2.
        periodic: If True, indices wrap around (for cyclic motions like crawl).
                  If False, indices are clamped (for one-shot motions like cartwheel).

    Returns:
        Tuple of (idx, next_idx, tau) where:
            idx: Integer index of the segment start, shape same as t.
            next_idx: Integer index of the segment end, shape same as t.
            tau: Local parameter in [0, 1]. Can be exactly 1.0 when t
                 equals n_frames - 1 in non-periodic mode. All interpolators
                 must handle tau = 1.0 correctly (returns frames[next_idx]).

    Raises:
        ValueError: If n_frames < 2.
    """
    if n_frames < 2:
        raise ValueError(f"Need at least 2 keyframes, got {n_frames}")

    if periodic:
        t = t % n_frames
        idx = jnp.floor(t).astype(jnp.int32)
        next_idx = (idx + 1) % n_frames
        tau = t - idx
    else:
        t = jnp.clip(t, 0.0, n_frames - 1.0)
        idx = jnp.floor(t).astype(jnp.int32)
        idx = jnp.minimum(idx, n_frames - 2)
        next_idx = idx + 1
        tau = t - idx

    return idx, next_idx, tau


class KeyframeInterpolator(ABC):
    """Abstract base for keyframe interpolation strategies.

    Subclasses implement `interpolate` which takes a frame array and a
    continuous index and returns the interpolated value. All implementations
    must be JAX-jittable.
    """

    @abstractmethod
    def interpolate(
        self,
        frames: jnp.ndarray,
        t: jnp.ndarray,
        periodic: bool = False,
    ) -> jnp.ndarray:
        """Interpolate keyframes at a continuous index.

        Args:
            frames: Keyframe array of shape (N, ...) where N is the number
                    of frames and ... is arbitrary trailing dimensions.
            t: Continuous frame index (scalar). E.g. 3.7 means 70% between
               frame 3 and frame 4.
            periodic: Whether the motion is cyclic.

        Returns:
            Interpolated value of shape (...), matching frames' trailing dims.
        """

    def __call__(
        self,
        frames: jnp.ndarray,
        t: jnp.ndarray,
        periodic: bool = False,
    ) -> jnp.ndarray:
        """Convenience alias for interpolate."""
        return self.interpolate(frames, t, periodic)
