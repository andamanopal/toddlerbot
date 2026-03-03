"""Visualization utilities for interpolation comparison.

Generates publication-quality figures comparing interpolation methods.
All functions accept numpy data and produce matplotlib figures.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# Consistent color and label mappings across all figures
METHOD_COLORS = {
    "linear": "#1f77b4",
    "min_jerk": "#ff7f0e",
    "min_jerk_viapoint": "#d62728",
    "cubic_spline": "#2ca02c",
}
METHOD_LABELS = {
    "linear": "Linear",
    "min_jerk": "MJ (rest-to-rest)",
    "min_jerk_viapoint": "MJ (via-point)",
    "cubic_spline": "Cubic Spline",
}

# Brief descriptions for the method footnote
METHOD_DESCRIPTIONS = {
    "Linear": "Piecewise linear blend between adjacent keyframes",
    "MJ (rest-to-rest)": "5th-order min-jerk profile; zero velocity at every keyframe",
    "MJ (via-point)": "5th-order polynomial with Catmull\u2013Rom velocity estimates",
    "Cubic Spline": "C\u00b2-continuous natural cubic spline through all keyframes",
}


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Apply rolling-mean smoothing to a 1-D array."""
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def _format_number(val: float) -> str:
    """Format a number for the summary table: commas, sensible rounding."""
    if abs(val) >= 1000:
        return f"{val:,.0f}"
    if abs(val) >= 1:
        return f"{val:,.2f}"
    return f"{val:.4f}"


def _format_ratio(val: float) -> str:
    """Format a ratio value (e.g., 0.19x or 11.5x)."""
    if not np.isfinite(val):
        return "N/A"
    if val >= 100:
        return f"{val:,.0f}x"
    if val >= 10:
        return f"{val:.1f}x"
    return f"{val:.2f}x"


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Compute ratio with zero-denominator guard."""
    if denominator == 0.0:
        return float("nan")
    return numerator / denominator


def plot_reward_curves(
    results: dict,
    title: str = "Training Reward Curves",
    output_path: Optional[str] = None,
    smooth_window: int = 50,
) -> plt.Figure:
    """Plot reward curves for multiple interpolation methods.

    Args:
        results: Dict mapping method name to dict with 'steps' and 'rewards' arrays.
        title: Figure title.
        output_path: If provided, save figure to this path.
        smooth_window: Rolling-average window for smoothing curves.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for method, data in results.items():
        steps = np.array(data["steps"])
        rewards = np.array(data["rewards"])
        color = METHOD_COLORS.get(method, None)
        label = METHOD_LABELS.get(method, method)

        if rewards.ndim == 2:
            mean = np.mean(rewards, axis=0)
            std = np.std(rewards, axis=0)
            mean_s = _smooth(mean, smooth_window)
            std_s = _smooth(std, smooth_window)
            steps_s = steps[:len(mean_s)]
            ax.plot(steps_s, mean_s, label=label, color=color, linewidth=1.5)
            ax.fill_between(
                steps_s, mean_s - std_s, mean_s + std_s, alpha=0.15, color=color,
            )
        else:
            rewards_s = _smooth(rewards, smooth_window)
            steps_s = steps[:len(rewards_s)]
            ax.plot(steps_s, rewards_s, label=label, color=color, linewidth=1.5)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title(title, fontsize=12, pad=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_jerk_comparison(
    jerk_data: dict,
    title: str = "Reference Trajectory Smoothness",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing jerk across methods, normalized to linear baseline.

    Shows two subplots: mean jerk and peak jerk, both as ratios relative
    to the linear interpolation baseline (linear = 1.0x).

    Args:
        jerk_data: Dict mapping method name to dict with 'mean_jerk' and 'max_jerk'.
        title: Figure title.
        output_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    methods = list(jerk_data.keys())
    display_names = [METHOD_LABELS.get(m, m) for m in methods]
    bar_colors = [METHOD_COLORS.get(m, "#999999") for m in methods]

    # Normalize to linear baseline
    linear_mean = jerk_data.get("linear", {}).get("mean_jerk", 0.0)
    linear_max = jerk_data.get("linear", {}).get("max_jerk", 0.0)

    mean_ratios = [_safe_ratio(jerk_data[m]["mean_jerk"], linear_mean) for m in methods]
    max_ratios = [_safe_ratio(jerk_data[m]["max_jerk"], linear_max) for m in methods]

    # Shared y-axis limit
    y_max = max(max(mean_ratios), max(max_ratios))
    label_offset = y_max * 0.03

    # Mean jerk subplot
    bars1 = ax1.bar(
        display_names, mean_ratios, color=bar_colors,
        edgecolor="black", linewidth=0.5,
    )
    ax1.set_ylabel("Jerk (relative to Linear)")
    ax1.set_title("Mean Jerk", fontsize=11)
    ax1.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax1.tick_params(axis="x", rotation=20)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    for bar, ratio in zip(bars1, mean_ratios):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            _format_ratio(ratio),
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # Peak jerk subplot
    bars2 = ax2.bar(
        display_names, max_ratios, color=bar_colors,
        edgecolor="black", linewidth=0.5,
    )
    ax2.set_title("Peak Jerk", fontsize=11)
    ax2.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax2.tick_params(axis="x", rotation=20)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for bar, ratio in zip(bars2, max_ratios):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            _format_ratio(ratio),
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_tracking_error(
    tracking_data: dict,
    joint_names: Optional[list] = None,
    title: str = "Per-Joint Tracking RMSE",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Grouped bar chart of per-joint tracking RMSE.

    Args:
        tracking_data: Dict mapping method name to array of per-joint RMSE.
        joint_names: Optional list of joint names for x-axis labels.
        title: Figure title.
        output_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    methods = list(tracking_data.keys())
    n_joints = len(next(iter(tracking_data.values())))

    if joint_names is None:
        joint_names = [f"Joint {i}" for i in range(n_joints)]

    fig, ax = plt.subplots(figsize=(max(8, n_joints * 0.5), 5))

    x = np.arange(n_joints)
    width = 0.8 / len(methods)

    for i, method in enumerate(methods):
        offset = (i - len(methods) / 2 + 0.5) * width
        vals = np.array(tracking_data[method])
        ax.bar(
            x + offset,
            vals,
            width,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, None),
            edgecolor="black",
            linewidth=0.3,
        )

    ax.set_xlabel("Joint")
    ax.set_ylabel("RMSE (rad)")
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(joint_names, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_summary_table(
    summary_data: dict,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Create a summary comparison table as a figure.

    Displays absolute jerk values with commas and relative ratios
    normalized to the linear baseline. Includes method descriptions.

    Args:
        summary_data: Dict mapping method name to dict of metrics.
        output_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    methods = list(summary_data.keys())
    col_labels = [METHOD_LABELS.get(m, m) for m in methods]

    # Compute normalized ratios relative to linear
    linear_data = summary_data.get("linear", {})
    linear_mean = linear_data.get("mean_jerk", 0.0)
    linear_max = linear_data.get("max_jerk", 0.0)

    row_specs = [
        ("Mean Jerk", lambda m: _format_number(summary_data[m]["mean_jerk"])),
        ("Peak Jerk", lambda m: _format_number(summary_data[m]["max_jerk"])),
        (
            "Mean Jerk (vs. Linear)",
            lambda m: _format_ratio(_safe_ratio(summary_data[m]["mean_jerk"], linear_mean)),
        ),
        (
            "Peak Jerk (vs. Linear)",
            lambda m: _format_ratio(_safe_ratio(summary_data[m]["max_jerk"], linear_max)),
        ),
    ]

    row_labels = [spec[0] for spec in row_specs]
    cell_text = [[spec[1](m) for m in methods] for spec in row_specs]

    n_cols = len(methods)
    fig_width = max(8, 2.2 * n_cols)
    active_descs = [
        f"{label}: {desc}"
        for label, desc in METHOD_DESCRIPTIONS.items()
        if label in col_labels
    ]

    fig, ax = plt.subplots(figsize=(fig_width, 3.5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        bbox=[0.0, 0.35, 1.0, 0.65],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Footnotes directly below the table
    footnote = "\n".join(active_descs)
    ax.text(
        0.0, 0.30, footnote,
        fontsize=7.5, va="top",
        style="italic", color="#444444",
        transform=ax.transAxes,
    )

    ax.set_title("Interpolation Method Comparison", fontsize=12, pad=12)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_policy_vs_reference_jerk(
    policy_jerk_data: dict,
    title: str = "Policy vs Reference Jerk by Interpolation Method",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Grouped bar chart comparing policy-output jerk to reference jerk.

    For each method, shows reference jerk (hatched) alongside policy jerk (solid).
    This is the key figure for answering Haochen's question.

    Args:
        policy_jerk_data: Dict mapping method name to dict with
            'policy_mean_jerk', 'policy_max_jerk',
            'ref_mean_jerk', 'ref_max_jerk'.
        title: Figure title.
        output_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    methods = list(policy_jerk_data.keys())
    display_names = [METHOD_LABELS.get(m, m) for m in methods]
    bar_colors = [METHOD_COLORS.get(m, "#999999") for m in methods]
    x = np.arange(len(methods))
    bar_width = 0.35

    # Mean jerk subplot
    ref_means = [policy_jerk_data[m].get("ref_mean_jerk", 0) for m in methods]
    policy_means = [policy_jerk_data[m]["policy_mean_jerk"] for m in methods]

    bars_ref = ax1.bar(
        x - bar_width / 2, ref_means, bar_width,
        label="Reference", color=bar_colors,
        edgecolor="black", linewidth=0.5, hatch="//", alpha=0.5,
    )
    bars_pol = ax1.bar(
        x + bar_width / 2, policy_means, bar_width,
        label="Policy", color=bar_colors,
        edgecolor="black", linewidth=0.5,
    )

    ax1.set_ylabel("Mean Jerk")
    ax1.set_title("Mean Jerk", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_names, rotation=20, ha="right", fontsize=9)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    y_max = max(max(ref_means + policy_means, default=1), 1e-9)
    label_offset = y_max * 0.02
    for bar in bars_ref:
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + label_offset,
            _format_number(bar.get_height()),
            ha="center", va="bottom", fontsize=7, color="#666666",
        )
    for bar in bars_pol:
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + label_offset,
            _format_number(bar.get_height()),
            ha="center", va="bottom", fontsize=7, fontweight="bold",
        )

    # Peak jerk subplot
    ref_maxes = [policy_jerk_data[m].get("ref_max_jerk", 0) for m in methods]
    policy_maxes = [policy_jerk_data[m]["policy_max_jerk"] for m in methods]

    ax2.bar(
        x - bar_width / 2, ref_maxes, bar_width,
        label="Reference", color=bar_colors,
        edgecolor="black", linewidth=0.5, hatch="//", alpha=0.5,
    )
    ax2.bar(
        x + bar_width / 2, policy_maxes, bar_width,
        label="Policy", color=bar_colors,
        edgecolor="black", linewidth=0.5,
    )

    ax2.set_ylabel("Peak Jerk")
    ax2.set_title("Peak Jerk", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(display_names, rotation=20, ha="right", fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_jerk_transfer_ratio(
    policy_jerk_data: dict,
    title: str = "Jerk Transfer Ratio (Policy / Reference)",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of jerk transfer ratio per method.

    Transfer ratio = policy_mean_jerk / ref_mean_jerk.
    If close to 1.0, reference smoothness fully transfers to the policy.
    If much higher, RL adds its own jerk on top.

    Args:
        policy_jerk_data: Dict mapping method name to dict with
            'jerk_transfer_ratio' or 'policy_mean_jerk' and 'ref_mean_jerk'.
        title: Figure title.
        output_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    methods = list(policy_jerk_data.keys())
    display_names = [METHOD_LABELS.get(m, m) for m in methods]
    bar_colors = [METHOD_COLORS.get(m, "#999999") for m in methods]

    ratios = []
    for m in methods:
        d = policy_jerk_data[m]
        if "jerk_transfer_ratio" in d:
            ratios.append(d["jerk_transfer_ratio"])
        elif d.get("ref_mean_jerk", 0) > 0:
            ratios.append(d["policy_mean_jerk"] / d["ref_mean_jerk"])
        else:
            ratios.append(0.0)

    bars = ax.bar(
        display_names, ratios, color=bar_colors,
        edgecolor="black", linewidth=0.5,
    )

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Perfect transfer (1.0x)")
    ax.set_ylabel("Jerk Transfer Ratio")
    ax.set_title(title, fontsize=12, pad=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.tick_params(axis="x", rotation=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_max = max(ratios) if ratios else 1
    label_offset = y_max * 0.03
    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            _format_ratio(ratio),
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_summary_table_policy(
    policy_jerk_data: dict,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Extended summary table with both reference and policy-level metrics.

    Args:
        policy_jerk_data: Dict mapping method name to dict with both
            reference and policy jerk metrics.
        output_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    methods = list(policy_jerk_data.keys())
    col_labels = [METHOD_LABELS.get(m, m) for m in methods]

    linear_data = policy_jerk_data.get("linear", {})
    linear_pol_mean = linear_data.get("policy_mean_jerk", 0.0)

    default_rows = [
        ("Ref Mean Jerk", lambda m: _format_number(
            policy_jerk_data[m].get("ref_mean_jerk", 0)
        )),
        ("Policy Mean Jerk", lambda m: _format_number(
            policy_jerk_data[m]["policy_mean_jerk"]
        )),
        ("Policy Peak Jerk", lambda m: _format_number(
            policy_jerk_data[m]["policy_max_jerk"]
        )),
        ("Jerk Transfer Ratio", lambda m: _format_ratio(
            _safe_ratio(
                policy_jerk_data[m]["policy_mean_jerk"],
                policy_jerk_data[m].get("ref_mean_jerk", 0.0),
            )
        )),
        ("Tracking RMSE", lambda m: f"{policy_jerk_data[m].get('tracking_rmse', 0):.4f}"
         if "tracking_rmse" in policy_jerk_data[m] else "N/A"),
        ("Policy Jerk (vs. Linear)", lambda m: _format_ratio(
            _safe_ratio(policy_jerk_data[m]["policy_mean_jerk"], linear_pol_mean)
        )),
    ]

    row_labels = [r[0] for r in default_rows]
    cell_text = [[r[1](m) for m in methods] for r in default_rows]

    n_cols = len(methods)
    fig_width = max(8, 2.2 * n_cols)
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        bbox=[0.0, 0.25, 1.0, 0.75],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    active_descs = [
        f"{label}: {desc}"
        for label, desc in METHOD_DESCRIPTIONS.items()
        if label in col_labels
    ]
    footnotes = [
        "Note: action_rate penalty = 0.0 in crawl.gin; "
        "active regularization is motor_torque (0.5) and energy (0.05) only.",
    ] + active_descs

    ax.text(
        0.0, 0.20, "\n".join(footnotes),
        fontsize=7, va="top",
        style="italic", color="#444444",
        transform=ax.transAxes,
    )

    ax.set_title(
        "Policy vs Reference: Interpolation Method Comparison",
        fontsize=12, pad=12,
    )

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
