"""Cubic spline keyframe interpolation.

Implements natural cubic spline interpolation through keyframes.
Coefficients are precomputed once (using numpy) and stored as JAX arrays
for fast evaluation during RL training.

The spline is C2-continuous: position, velocity, and acceleration are all
continuous at keyframe boundaries, providing the smoothest polynomial
interpolation of degree 3.

For a segment between frame i and frame i+1, the polynomial is:
    S_i(tau) = a_i + b_i*tau + c_i*tau^2 + d_i*tau^3
where tau in [0, 1] is the local parameter within the segment.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from interp_analysis.interpolation.base import KeyframeInterpolator, resolve_index


class SplineCoeffs(NamedTuple):
    """Precomputed cubic spline coefficients.

    Each array has shape (n_segments, ...) where ... matches the
    trailing dimensions of the input keyframe array.

    Note: ``n_frames`` and ``periodic`` are Python scalars (not JAX arrays).
    They act as static values during JIT tracing. Changing them between
    calls triggers recompilation.
    """

    a: jnp.ndarray
    b: jnp.ndarray
    c: jnp.ndarray
    d: jnp.ndarray
    n_frames: int
    periodic: bool


def _solve_tridiagonal(lower: np.ndarray, diag: np.ndarray,
                       upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal system using the Thomas algorithm.

    Operates along the first axis of rhs, supporting arbitrary trailing dims.

    Precondition: The system must be diagonally dominant or otherwise
    guarantee non-zero pivots. No pivoting is performed. For the natural
    cubic spline system (diagonal=4, off-diagonals=1) this is always safe.

    Args:
        lower: Sub-diagonal, shape (n-1, ...).
        diag: Main diagonal, shape (n, ...).
        upper: Super-diagonal, shape (n-1, ...).
        rhs: Right-hand side, shape (n, ...).

    Returns:
        Solution array, shape (n, ...).
    """
    n = diag.shape[0]
    # Copy to avoid mutating caller's arrays. In-place mutation within
    # this function is intentional for the Thomas algorithm's O(n) space.
    diag = diag.copy()
    rhs = rhs.copy()

    for i in range(1, n):
        factor = lower[i - 1] / diag[i - 1]
        diag[i] = diag[i] - factor * upper[i - 1]
        rhs[i] = rhs[i] - factor * rhs[i - 1]

    x = np.zeros_like(rhs)
    x[n - 1] = rhs[n - 1] / diag[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (rhs[i] - upper[i] * x[i + 1]) / diag[i]

    return x


def compute_natural_spline_coeffs(frames: np.ndarray) -> tuple:
    """Compute natural cubic spline coefficients for non-periodic data.

    Natural boundary conditions: S''(0) = 0 and S''(n-1) = 0.

    Args:
        frames: Keyframe array of shape (N, ...) with N >= 2.

    Returns:
        Tuple of (a, b, c, d) coefficient arrays, each shape (N-1, ...).
    """
    n = frames.shape[0]
    n_seg = n - 1
    trailing_shape = frames.shape[1:]

    a = frames[:-1].copy()

    c_full = np.zeros_like(frames)

    if n > 2:
        n_interior = n - 2
        ones = np.ones((n_interior,) + trailing_shape, dtype=frames.dtype)

        diag = 4.0 * ones
        lower = ones[:-1].copy()
        upper = ones[:-1].copy()

        rhs = np.zeros((n_interior,) + trailing_shape, dtype=frames.dtype)
        for i in range(n_interior):
            j = i + 1
            rhs[i] = 3.0 * (frames[j + 1] - 2.0 * frames[j] + frames[j - 1])

        c_interior = _solve_tridiagonal(lower, diag, upper, rhs)
        c_full[1:n - 1] = c_interior

    b = np.zeros((n_seg,) + trailing_shape, dtype=frames.dtype)
    d = np.zeros((n_seg,) + trailing_shape, dtype=frames.dtype)

    for i in range(n_seg):
        d[i] = (c_full[i + 1] - c_full[i]) / 3.0
        b[i] = (frames[i + 1] - frames[i]) - (2.0 * c_full[i] + c_full[i + 1]) / 3.0

    c = c_full[:-1].copy()

    return a, b, c, d


def compute_periodic_spline_coeffs(frames: np.ndarray) -> tuple:
    """Compute periodic cubic spline coefficients for cyclic data.

    The spline wraps: S(0) = S(N), S'(0) = S'(N), S''(0) = S''(N).
    The last frame is treated as wrapping back to the first frame.

    Uses a dense solve for the cyclic tridiagonal system. This is O(n^3)
    but sufficient for typical keyframe counts (< 100 frames). For larger
    sequences, a Sherman-Morrison reduction to two Thomas solves would
    give O(n) complexity.

    Args:
        frames: Keyframe array of shape (N, ...). Frame N-1 connects back to frame 0.

    Returns:
        Tuple of (a, b, c, d) coefficient arrays, each shape (N, ...).
        Segment i goes from frame i to frame (i+1) % N.
    """
    n = frames.shape[0]
    trailing_shape = frames.shape[1:]

    a = frames.copy()

    rhs = np.zeros((n,) + trailing_shape, dtype=frames.dtype)
    for i in range(n):
        i_prev = (i - 1) % n
        i_next = (i + 1) % n
        rhs[i] = 3.0 * (frames[i_next] - 2.0 * frames[i] + frames[i_prev])

    matrix = np.zeros((n, n), dtype=frames.dtype)
    for i in range(n):
        matrix[i, i] = 4.0
        matrix[i, (i - 1) % n] = 1.0
        matrix[i, (i + 1) % n] = 1.0

    flat_rhs = rhs.reshape(n, -1)
    flat_c = np.linalg.solve(matrix, flat_rhs)
    c = flat_c.reshape((n,) + trailing_shape)

    b = np.zeros((n,) + trailing_shape, dtype=frames.dtype)
    d = np.zeros((n,) + trailing_shape, dtype=frames.dtype)

    for i in range(n):
        i_next = (i + 1) % n
        d[i] = (c[i_next] - c[i]) / 3.0
        b[i] = (frames[i_next] - frames[i]) - (2.0 * c[i] + c[i_next]) / 3.0

    return a, b, c, d


def precompute_coeffs(frames: np.ndarray, periodic: bool = False) -> SplineCoeffs:
    """Precompute cubic spline coefficients from keyframe data.

    Call this once per motion field during initialization. The returned
    coefficients are converted to JAX arrays for use during training.

    Coefficients are computed in float64 for numerical stability then
    stored as float32 for JAX compatibility.

    Args:
        frames: Keyframe array of shape (N, ...) as a numpy array. N >= 2.
        periodic: Whether the motion is cyclic.

    Returns:
        SplineCoeffs namedtuple with JAX arrays ready for evaluation.
    """
    frames_np = np.asarray(frames, dtype=np.float64)

    if periodic:
        a, b, c, d = compute_periodic_spline_coeffs(frames_np)
    else:
        a, b, c, d = compute_natural_spline_coeffs(frames_np)

    return SplineCoeffs(
        a=jnp.array(a, dtype=jnp.float32),
        b=jnp.array(b, dtype=jnp.float32),
        c=jnp.array(c, dtype=jnp.float32),
        d=jnp.array(d, dtype=jnp.float32),
        n_frames=frames.shape[0],
        periodic=periodic,
    )


def evaluate_spline(coeffs: SplineCoeffs, t: jnp.ndarray) -> jnp.ndarray:
    """Evaluate a precomputed cubic spline at continuous index t.

    This function is JAX-jittable and differentiable.

    Args:
        coeffs: Precomputed SplineCoeffs from precompute_coeffs.
        t: Continuous frame index (scalar).

    Returns:
        Interpolated value of shape (...) matching the keyframe trailing dims.
    """
    idx, _, tau = resolve_index(t, coeffs.n_frames, coeffs.periodic)

    a = coeffs.a[idx]
    b = coeffs.b[idx]
    c = coeffs.c[idx]
    d = coeffs.d[idx]

    # Horner's method: a + tau * (b + tau * (c + tau * d))
    return a + tau * (b + tau * (c + tau * d))


class CubicSplineInterpolator(KeyframeInterpolator):
    """Cubic spline interpolation through keyframes.

    Unlike linear and min-jerk interpolators, this requires a precomputation
    step. Use ``precompute`` to generate coefficients for each motion field,
    then use ``interpolate_with_coeffs`` during training for best performance.

    The ``interpolate`` method computes coefficients on the fly (slower, and
    NOT jit-compatible due to numpy precomputation). It exists only to
    satisfy the base class interface for testing.

    Recommended usage for RL training::

        interp = CubicSplineInterpolator()
        coeffs_qpos = interp.precompute(motion_ref["qpos"], periodic=True)
        # Inside jitted training step:
        qpos = interp.interpolate_with_coeffs(coeffs_qpos, t_float)
    """

    def precompute(self, frames: np.ndarray, periodic: bool = False) -> SplineCoeffs:
        """Precompute spline coefficients for a keyframe array.

        Args:
            frames: Keyframe array of shape (N, ...) as numpy array.
            periodic: Whether the motion is cyclic.

        Returns:
            SplineCoeffs for use with interpolate_with_coeffs.
        """
        return precompute_coeffs(frames, periodic)

    def interpolate_with_coeffs(
        self,
        coeffs: SplineCoeffs,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate precomputed spline at continuous index t.

        This is the fast path for use during RL training. JAX-jittable.

        Args:
            coeffs: Precomputed SplineCoeffs.
            t: Continuous frame index (scalar).

        Returns:
            Interpolated value.
        """
        return evaluate_spline(coeffs, t)

    def interpolate(
        self,
        frames: jnp.ndarray,
        t: jnp.ndarray,
        periodic: bool = False,
    ) -> jnp.ndarray:
        """Interpolate by computing coefficients on the fly.

        WARNING: NOT jit-compatible. Precomputation uses numpy internally.
        For JIT-compatible evaluation, use:
            coeffs = interp.precompute(frames_np, periodic)
            result = interp.interpolate_with_coeffs(coeffs, t)

        Args:
            frames: Keyframe array of shape (N, ...).
            t: Continuous frame index.
            periodic: Whether the motion is cyclic.

        Returns:
            Interpolated value.
        """
        frames_np = np.asarray(frames)
        coeffs = precompute_coeffs(frames_np, periodic)
        return evaluate_spline(coeffs, t)
