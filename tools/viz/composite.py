"""Figure 10: Composite multi-panel summary figure (PAPER.md §3).

Combines ERSP heatmaps (Active / Sham), difference map, frequency crossover,
and violin plot in a single overview for the manuscript.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import BANDS, ERSP as ERSP_CFG
from .style import (
    COLORS, GROUP_LABELS, GROUP_ORDER, apply_style,
    ersp_cmap, add_reward_onset_line, save_fig,
)


def plot_composite_summary(
    ersp_active: np.ndarray,
    ersp_sham: np.ndarray,
    diff_data: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    freq_profiles: Dict[str, np.ndarray],
    freq_profiles_se: Dict[str, np.ndarray],
    scalars: pd.DataFrame,
    cluster_mask: Optional[np.ndarray] = None,
    channel: str = "C3",
    title: str = "ERSP Mechanism Summary",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Multi-panel summary combining key result types.

    Layout (2 x 3 grid)::

        A. Active ERSP heatmap   B. Sham ERSP heatmap   C. Active-Sham diff
        D. Frequency crossover   E-F. Violin plot (spanning 2 columns)

    Parameters
    ----------
    ersp_active, ersp_sham : ndarray (n_freqs, n_times)
    diff_data : ndarray (n_freqs, n_times)
    freq_profiles, freq_profiles_se : dict[group_key] -> 1-D (n_freqs,)
    scalars : DataFrame from assemble_ersp_scalars
    cluster_mask : optional bool array for difference significance contours
    """
    apply_style()

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    disp_mask = (times >= ERSP_CFG.display_tmin) & (times <= ERSP_CFG.display_tmax)
    all_heatmap = np.concatenate([
        ersp_active[:, disp_mask].ravel(),
        ersp_sham[:, disp_mask].ravel(),
    ])
    clim = max(abs(np.percentile(all_heatmap, 2)),
               abs(np.percentile(all_heatmap, 98)))
    clim = round(clim, 2) if clim > 0.1 else 0.5
    cmap, _, _ = ersp_cmap(-clim, clim)

    diff_cropped = diff_data[:, disp_mask]
    diff_clim = max(abs(np.percentile(diff_cropped, 2)),
                    abs(np.percentile(diff_cropped, 98)))
    diff_clim = round(diff_clim, 2) if diff_clim > 0.1 else 0.5

    dtmin, dtmax = ERSP_CFG.display_tmin, ERSP_CFG.display_tmax

    # A: Active ERSP
    ax_a = fig.add_subplot(gs[0, 0])
    im = ax_a.pcolormesh(times, freqs, ersp_active,
                         cmap=cmap, vmin=-clim, vmax=clim, shading="auto")
    add_reward_onset_line(ax_a)
    ax_a.set_xlim(dtmin, dtmax)
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel("Frequency (Hz)")
    ax_a.set_title("A. Active (pooled)", fontsize=10)
    fig.colorbar(im, ax=ax_a, shrink=0.8, label="dB")

    # B: Sham ERSP
    ax_b = fig.add_subplot(gs[0, 1])
    im2 = ax_b.pcolormesh(times, freqs, ersp_sham,
                          cmap=cmap, vmin=-clim, vmax=clim, shading="auto")
    add_reward_onset_line(ax_b)
    ax_b.set_xlim(dtmin, dtmax)
    ax_b.set_xlabel("Time (s)")
    ax_b.set_ylabel("Frequency (Hz)")
    ax_b.set_title("B. Sham", fontsize=10)
    fig.colorbar(im2, ax=ax_b, shrink=0.8, label="dB")

    # C: Difference map
    ax_c = fig.add_subplot(gs[0, 2])
    cmap_d, _, _ = ersp_cmap(-diff_clim, diff_clim)
    im3 = ax_c.pcolormesh(times, freqs, diff_data,
                          cmap=cmap_d, vmin=-diff_clim, vmax=diff_clim,
                          shading="auto")
    add_reward_onset_line(ax_c)
    if cluster_mask is not None and cluster_mask.any():
        ax_c.contour(times, freqs, cluster_mask.astype(float),
                     levels=[0.5], colors="k", linewidths=1.0)
    ax_c.set_xlim(dtmin, dtmax)
    ax_c.set_xlabel("Time (s)")
    ax_c.set_ylabel("Frequency (Hz)")
    ax_c.set_title("C. Active \u2212 Sham", fontsize=10)
    fig.colorbar(im3, ax=ax_c, shrink=0.8, label="\u0394 dB")

    # D: Frequency crossover
    ax_d = fig.add_subplot(gs[1, 0])
    freq_range = (8.0, 25.0)
    fmask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    f = freqs[fmask]
    for gkey in ("c3_smr", "c3_beta", "sham"):
        if gkey not in freq_profiles:
            continue
        y = freq_profiles[gkey][fmask]
        se = freq_profiles_se[gkey][fmask]
        ax_d.plot(f, y, color=COLORS[gkey], label=GROUP_LABELS[gkey])
        ax_d.fill_between(f, y - 1.96 * se, y + 1.96 * se,
                          color=COLORS[gkey], alpha=0.15)

    smr = BANDS["smr"]
    beta = BANDS["beta"]
    ax_d.axvspan(smr[0], smr[1], color=COLORS["c3_smr"], alpha=0.08)
    ax_d.axvspan(beta[0], beta[1], color=COLORS["c3_beta"], alpha=0.08)
    ax_d.axhline(0, color="0.5", ls=":", lw=0.6)
    ax_d.set_xlabel("Frequency (Hz)")
    ax_d.set_ylabel("ERD (dB)")
    ax_d.set_title("D. Frequency Crossover", fontsize=10)
    ax_d.legend(fontsize=7, loc="best")

    # E: Violin plot (spans 2 columns)
    ax_e = fig.add_subplot(gs[1, 1:])
    ch_mask = scalars["metric"].str.startswith(f"primary_erd_{channel}")
    ch_data = scalars[ch_mask].copy()
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

    if group_vals:
        vp = ax_e.violinplot(group_vals, positions=positions,
                             showmedians=True, showextrema=False)
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(colors[i])
            body.set_alpha(0.3)
        vp["cmedians"].set_color("k")

        rng = np.random.default_rng(42)
        for i, (pos, vals) in enumerate(zip(positions, group_vals)):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax_e.scatter(pos + jitter, vals, color=colors[i], s=22, alpha=0.7,
                         edgecolors="white", linewidths=0.3, zorder=3)

    ax_e.axhline(0, color="0.5", ls=":", lw=0.6)
    ax_e.set_xticks(positions)
    ax_e.set_xticklabels(labels)
    ax_e.set_ylabel("ERD (dB)")
    ax_e.set_title(f"E. Individual ERD at {channel}", fontsize=10)

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()

    if save_path:
        save_fig(fig, save_path)
    return fig
