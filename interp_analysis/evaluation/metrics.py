"""Evaluation metrics for comparing interpolation methods.

Computes smoothness, tracking error, and reference trajectory quality
metrics. All functions work with numpy arrays for offline analysis.
"""

import numpy as np


def compute_jerk(trajectory: np.ndarray, dt: float) -> np.ndarray:
    """Compute per-timestep jerk magnitude of a trajectory.

    Jerk is the third derivative of position. Lower jerk = smoother motion.

    Args:
        trajectory: Array of shape (T, ...) where T is the number of timesteps.
        dt: Time step between frames.

    Returns:
        Jerk magnitudes of shape (T-3,). Each element is the L2 norm
        of the jerk vector at that timestep.
    """
    vel = np.diff(trajectory, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt

    # Flatten trailing dims and compute L2 norm per timestep
    jerk_flat = jerk.reshape(jerk.shape[0], -1)
    return np.linalg.norm(jerk_flat, axis=1)


def compute_mean_jerk(trajectory: np.ndarray, dt: float) -> float:
    """Compute mean jerk magnitude over a trajectory.

    Args:
        trajectory: Array of shape (T, ...).
        dt: Time step.

    Returns:
        Scalar mean jerk.
    """
    jerk_vals = compute_jerk(trajectory, dt)
    if len(jerk_vals) == 0:
        return 0.0
    return float(np.mean(jerk_vals))


def compute_tracking_rmse(
    actual: np.ndarray, reference: np.ndarray
) -> float:
    """Compute root-mean-squared tracking error between actual and reference.

    Args:
        actual: Actual joint positions, shape (T, n_joints).
        reference: Reference joint positions, shape (T, n_joints).

    Returns:
        Scalar RMSE averaged over all joints and timesteps.
    """
    diff = actual - reference
    return float(np.sqrt(np.mean(diff ** 2)))


def compute_per_joint_rmse(
    actual: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Compute per-joint RMSE between actual and reference.

    Args:
        actual: shape (T, n_joints).
        reference: shape (T, n_joints).

    Returns:
        Array of shape (n_joints,) with RMSE per joint.
    """
    diff = actual - reference
    return np.sqrt(np.mean(diff ** 2, axis=0))


def compute_spectral_energy(trajectory: np.ndarray, dt: float) -> dict:
    """Compute spectral energy distribution of a trajectory.

    Useful for comparing frequency content: smoother interpolation should
    have less high-frequency energy.

    Args:
        trajectory: shape (T, n_dims).
        dt: Time step.

    Returns:
        Dict with 'freqs' (Hz), 'power' (per-dim), 'high_freq_ratio'
        (fraction of energy above 10 Hz).
    """
    if trajectory.ndim == 1:
        trajectory = trajectory[:, np.newaxis]

    n = trajectory.shape[0]
    freqs = np.fft.rfftfreq(n, d=dt)

    # FFT per dimension, average power spectrum
    fft_vals = np.fft.rfft(trajectory, axis=0)
    power = np.mean(np.abs(fft_vals) ** 2, axis=1)

    total_energy = np.sum(power)
    high_freq_mask = freqs > 10.0
    high_freq_energy = np.sum(power[high_freq_mask]) if np.any(high_freq_mask) else 0.0

    high_freq_ratio = float(high_freq_energy / total_energy) if total_energy > 0 else 0.0

    return {
        "freqs": freqs,
        "power": power,
        "high_freq_ratio": high_freq_ratio,
    }


def compute_reference_smoothness(
    frames: np.ndarray,
    dt: float,
    method: str,
    n_samples: int = 1000,
) -> dict:
    """Evaluate smoothness of an interpolated reference trajectory.

    Generates a dense trajectory by sampling the reference at n_samples
    evenly spaced points and computing smoothness metrics.

    This is an offline metric that doesn't require GPU/training.

    Args:
        frames: Keyframe array of shape (N, ...).
        dt: Time step between original keyframes.
        method: Interpolation method name.
        n_samples: Number of samples in the dense trajectory.

    Returns:
        Dict with 'mean_jerk', 'max_jerk', 'high_freq_ratio'.
    """
    import jax.numpy as jnp

    from interp_analysis.interpolated_refs import make_interpolator

    interp = make_interpolator(method)

    n_frames = frames.shape[0]
    jax_frames = jnp.array(frames)

    t_values = np.linspace(0, n_frames - 1, n_samples, endpoint=True)
    dense_dt = (n_frames - 1) * dt / n_samples

    samples = []
    for t_val in t_values:
        val = np.array(interp(jax_frames, jnp.float32(t_val), periodic=False))
        samples.append(val)
    trajectory = np.stack(samples, axis=0)

    jerk_vals = compute_jerk(trajectory, dense_dt)
    spectral = compute_spectral_energy(trajectory, dense_dt)

    return {
        "mean_jerk": float(np.mean(jerk_vals)) if len(jerk_vals) > 0 else 0.0,
        "max_jerk": float(np.max(jerk_vals)) if len(jerk_vals) > 0 else 0.0,
        "high_freq_ratio": spectral["high_freq_ratio"],
    }


def parse_reward_log(log_path: str) -> tuple:
    """Parse reward values from a training output log.

    Extracts "Mean reward:" values and "Learning steps X/Y" step counts
    from the text log produced by train_mjx.

    Args:
        log_path: Path to output.log from train_mjx.

    Returns:
        Tuple of (steps, rewards) as lists, aligned to equal length.
    """
    steps = []
    rewards = []
    try:
        with open(log_path) as f:
            for line in f:
                if "Mean reward:" in line:
                    parts = line.strip().split()
                    try:
                        rewards.append(float(parts[-1]))
                    except (ValueError, IndexError):
                        pass
                elif "Learning steps" in line:
                    try:
                        parts = line.strip().split()
                        for part in parts:
                            if "/" in part:
                                steps.append(int(part.split("/")[0]))
                                break
                    except (ValueError, IndexError):
                        pass
    except OSError:
        pass

    min_len = min(len(steps), len(rewards))
    return steps[:min_len], rewards[:min_len]


