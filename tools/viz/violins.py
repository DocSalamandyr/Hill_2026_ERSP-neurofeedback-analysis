"""Figure 9: Violin plots for ERD band power by group (PAPER.md §3).

Individual-subject ERD magnitudes (session-averaged) for each group,
separate panels for C3 and C4.  Violin + strip overlay shows both the
distribution shape and individual data points.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import COLORS, GROUP_LABELS, GROUP_ORDER, apply_style, save_fig


def plot_erd_violins(
    scalars: pd.DataFrame,
    channels: Sequence[str] = ("C3", "C4"),
    metric_prefix: str = "primary_erd_",
    title: str = "ERD Magnitude by Group",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Violin + strip plot of per-subject ERD values.

    Parameters
    ----------
    scalars : DataFrame
        Long-format with columns: subject, group, session, channel, metric, value.
        Typically from :func:`group.assemble_ersp_scalars`.
    channels : sequence of str
        One panel per channel.
    metric_prefix : str
        Filter scalars to metrics starting with this prefix + channel name.
    """
    apply_style()
    n_ch = len(channels)
    fig, axes = plt.subplots(1, n_ch, figsize=(4 * n_ch, 5), sharey=True)
    if n_ch == 1:
        axes = [axes]

    for ax, ch in zip(axes, channels):
        mask = scalars["metric"].str.startswith(f"{metric_prefix}{ch}")
        ch_data = scalars[mask].copy()
        subj_means = ch_data.groupby(["subject", "group"])["value"].mean().reset_index()

        positions: list[int] = []
        group_vals: list[np.ndarray] = []
        colors: list[str] = []
        labels: list[str] = []

        for i, gkey in enumerate(GROUP_ORDER):
            vals = subj_means[subj_means["group"] == gkey]["value"].values
            if len(vals) == 0:
                continue
            positions.append(i)
            group_vals.append(vals)
            colors.append(COLORS[gkey])
            labels.append(GROUP_LABELS[gkey])

        if not group_vals:
            continue

        vp = ax.violinplot(group_vals, positions=positions,
                           showmedians=True, showextrema=False)
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(colors[i])
            body.set_alpha(0.3)
        vp["cmedians"].set_color("k")

        rng = np.random.default_rng(42)
        for i, (pos, vals) in enumerate(zip(positions, group_vals)):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(pos + jitter, vals, color=colors[i], s=22, alpha=0.7,
                       edgecolors="white", linewidths=0.3, zorder=3)

        ax.axhline(0, color="0.5", ls=":", lw=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(f"ERD at {ch}")
        if ax is axes[0]:
            ax.set_ylabel("ERD (dB)")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    if save_path:
        save_fig(fig, save_path)
    return fig
