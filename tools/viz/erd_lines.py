"""Figure 3: ERD magnitude by group and session (PAPER.md §3).

Line/bar plot — X=Session (1,3,5,6), Y=ERD magnitude (dB), separate
lines per group, with individual data points and forest-plot inset.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..config import BFB_SESSIONS
from .style import (
    COLORS, GROUP_LABELS, GROUP_ORDER, apply_style,
    save_fig,
)


def plot_erd_by_session(
    erd_data: Dict[str, Dict[int, np.ndarray]],
    sessions: tuple = BFB_SESSIONS,
    ylabel: str = "ERD (dB, reward band 200-800 ms)",
    title: str = "Reward-Band ERD by Group and Session",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Line plot of ERD magnitude across sessions.

    Parameters
    ----------
    erd_data : dict[group_key][session] → 1-D array of per-subject ERD values
    sessions : sequence of int
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))

    x_positions = np.arange(len(sessions))
    jitter_scale = 0.06

    for gkey in GROUP_ORDER:
        if gkey not in erd_data:
            continue
        means = []
        ci_lo = []
        ci_hi = []
        for sess in sessions:
            vals = erd_data[gkey].get(sess, np.array([]))
            m = vals.mean() if len(vals) > 0 else np.nan
            se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            means.append(m)
            ci_lo.append(m - 1.96 * se)
            ci_hi.append(m + 1.96 * se)

        means = np.array(means)
        ci_lo = np.array(ci_lo)
        ci_hi = np.array(ci_hi)

        ax.plot(x_positions, means, "o-",
                color=COLORS[gkey], label=GROUP_LABELS[gkey], zorder=3)
        ax.fill_between(x_positions, ci_lo, ci_hi,
                        color=COLORS[gkey], alpha=0.15, zorder=1)

        # Individual data points (jittered)
        for xi, sess in enumerate(sessions):
            vals = erd_data[gkey].get(sess, np.array([]))
            if len(vals) > 0:
                jx = xi + np.random.default_rng(42).uniform(
                    -jitter_scale, jitter_scale, size=len(vals))
                ax.scatter(jx, vals, color=COLORS[gkey], s=12, alpha=0.4,
                           edgecolors="none", zorder=2)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(s) for s in sessions])
    ax.set_xlabel("Session")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    ax.axhline(0, color="0.5", ls=":", lw=0.6)
    fig.tight_layout()

    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_forest(
    contrasts,
    title: str = "Effect Sizes: Active vs Sham",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Forest plot of Cohen's d with 95% CI from ContrastResult list."""
    apply_style()
    fig, ax = plt.subplots(figsize=(5, 0.6 * len(contrasts) + 1))

    y_pos = np.arange(len(contrasts))
    for i, c in enumerate(contrasts):
        ax.plot(c.cohens_d, i, "ko", markersize=6)
        ax.hlines(i, c.ci_low, c.ci_high, colors="k", linewidths=1.2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([c.name for c in contrasts])
    ax.axvline(0, color="0.5", ls="--", lw=0.8)
    ax.set_xlabel("Cohen's d (95% CI)")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()

    if save_path:
        save_fig(fig, save_path)
    return fig
