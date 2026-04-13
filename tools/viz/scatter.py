"""Figure 7: ERD-to-resting-state correlation scatterplot (PAPER.md §3).

X = mean ERD magnitude during training (sessions 1,3,5 averaged)
Y = resting-state power change (session 6 minus session 1, eyes-closed)
One point per active subject (n~23), color-coded by group.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS, GROUP_LABELS, apply_style, save_fig


def plot_erd_resting_scatter(
    erd_mean: Dict[str, np.ndarray],
    resting_delta: Dict[str, np.ndarray],
    xlabel: str = "Mean ERD (dB, sessions 1, 3, 5, 6)",
    ylabel: str = "Resting power change (S6 - S1, dB)",
    title: str = "ERD Predicts Follow-Up Change (Active Subjects)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scatterplot with regression line and stats.

    Parameters
    ----------
    erd_mean : dict[group_key] → 1-D array of per-subject mean ERD
    resting_delta : dict[group_key] → 1-D array of per-subject delta power
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(5, 4.5))

    all_x = []
    all_y = []

    for gkey in ("c3_smr", "c3_beta", "c4_smr"):
        if gkey not in erd_mean or gkey not in resting_delta:
            continue
        x = erd_mean[gkey]
        y = resting_delta[gkey]
        ax.scatter(x, y, color=COLORS[gkey], label=GROUP_LABELS[gkey],
                   s=40, edgecolors="white", linewidths=0.5, zorder=3)
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())

    # Regression line across all active subjects
    if len(all_x) > 2:
        from scipy.stats import pearsonr, linregress
        x_arr = np.array(all_x)
        y_arr = np.array(all_y)
        slope, intercept, r, p, se = linregress(x_arr, y_arr)
        x_fit = np.linspace(x_arr.min(), x_arr.max(), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, "k-", lw=1.0, zorder=2)

        # 95% CI band
        n = len(x_arr)
        x_mean = x_arr.mean()
        s_res = np.sqrt(np.sum((y_arr - (slope * x_arr + intercept)) ** 2) / (n - 2))
        ci_band = 1.96 * s_res * np.sqrt(1 / n + (x_fit - x_mean) ** 2 / np.sum((x_arr - x_mean) ** 2))
        ax.fill_between(x_fit, y_fit - ci_band, y_fit + ci_band,
                        color="0.8", alpha=0.4, zorder=1)

        ax.text(0.05, 0.95,
                f"r = {r:.3f}, R² = {r**2:.3f}\np = {p:.4f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    if save_path:
        save_fig(fig, save_path)
    return fig
