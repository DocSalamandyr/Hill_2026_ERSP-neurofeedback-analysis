"""Within-session learning curves: ERD over sliding trial windows (PAPER.md §3).

Plots ERD magnitude (from ``epochs.compute_sliding_metric``) across trial
windows within each session, one panel per group, sessions as separate lines.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..config import BFB_SESSIONS
from .style import COLORS, GROUP_LABELS, GROUP_ORDER, apply_style, save_fig


def plot_learning_curves(
    curves: Dict[str, Dict[int, List[Dict]]],
    metric_key: str = "primary_erd_C3",
    sessions: tuple = BFB_SESSIONS,
    title: str = "Within-Session Learning Curves (ERD)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot within-session ERD trajectory across sliding trial windows.

    Parameters
    ----------
    curves : dict[group_key][session] → list of dicts from
        ``epochs.compute_sliding_metric()``.  Each dict has
        ``window_start``, ``window_end``, and metric keys.
    metric_key : str
        Which scalar metric to plot from each window dict.
    sessions : tuple of int
    """
    apply_style()

    groups = [g for g in GROUP_ORDER if g in curves]
    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(4 * n_groups, 4), sharey=True)
    if n_groups == 1:
        axes = [axes]

    session_colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(sessions)))

    for ax, grp in zip(axes, groups):
        for si, sess in enumerate(sessions):
            windows = curves.get(grp, {}).get(sess, [])
            if not windows:
                continue
            x = [(w["window_start"] + w["window_end"]) / 2.0 for w in windows]
            y = [w.get(metric_key, float("nan")) for w in windows]
            ax.plot(x, y, marker="o", ms=3, color=session_colors[si],
                    label=f"S{sess}", linewidth=1.5)

        ax.set_title(GROUP_LABELS.get(grp, grp), fontsize=10)
        ax.set_xlabel("Trial window centre")
        if ax is axes[0]:
            ax.set_ylabel("ERD (dB)")
        ax.legend(fontsize=7, frameon=False)

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()

    if save_path:
        save_fig(fig, save_path)

    return fig
