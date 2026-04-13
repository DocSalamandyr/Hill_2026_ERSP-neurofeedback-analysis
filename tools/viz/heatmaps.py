"""Figures 1-2: ERSP time-frequency heatmaps at C3 and C4 (PAPER.md §3).

Layout: N rows (groups) x M columns (segments/sessions).
Color: dB (blue=ERD, red=ERS). Contour overlay for significant clusters.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..config import ErspConfig, ERSP as ERSP_CFG
from .style import (
    GROUP_LABELS, GROUP_ORDER, apply_style, ersp_cmap,
    add_reward_onset_line, save_fig,
)


def _display_crop_mask(times: np.ndarray) -> np.ndarray:
    """Boolean mask restricting to the display window (avoids edge contamination)."""
    return (times >= ERSP_CFG.display_tmin) & (times <= ERSP_CFG.display_tmax)


def plot_ersp_heatmaps(
    ersp_data: Dict[str, Dict[str, np.ndarray]],
    times: np.ndarray,
    freqs: np.ndarray,
    channel: str = "C3",
    cluster_masks: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    segments: Optional[List[str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Generate an N-group x M-segment ERSP heatmap grid.

    Parameters
    ----------
    ersp_data : dict[group_key][segment_label] -> ndarray (n_freqs, n_times)
        Data should already be baseline-recentered (baseline mean ≈ 0).
    times, freqs : ndarray
    channel : str
        Electrode label (for title).
    cluster_masks : optional dict with same structure, bool arrays for
        significant clusters.
    segments : list of str or None
        Column labels.  Defaults to early/late splits.
    vmin, vmax : colorbar range (dB).  If None, auto-scale from data
        percentiles within the display window (symmetric around 0).
    """
    apply_style()
    if segments is None:
        segments = ["S1 Early", "S1 Late", "S5 Early", "S5 Late"]

    disp_mask = _display_crop_mask(times)

    if vmin is None or vmax is None:
        all_vals = np.concatenate([
            arr[:, disp_mask].ravel()
            for gd in ersp_data.values()
            for arr in gd.values()
            if arr is not None
        ])
        clim = max(abs(np.percentile(all_vals, 2)),
                   abs(np.percentile(all_vals, 98)))
        clim = round(clim, 2) if clim > 0.1 else 0.5
        if vmin is None:
            vmin = -clim
        if vmax is None:
            vmax = clim

    cmap, _, _ = ersp_cmap(vmin, vmax)

    n_rows = len(GROUP_ORDER)
    n_cols = len(segments)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3 * n_cols, 2.5 * n_rows),
        sharex=True, sharey=True,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    im = None
    for ri, gkey in enumerate(GROUP_ORDER):
        for ci, seg in enumerate(segments):
            ax = axes[ri, ci]
            data = ersp_data.get(gkey, {}).get(seg)
            if data is None:
                ax.set_visible(False)
                continue

            im = ax.pcolormesh(
                times, freqs, data,
                cmap=cmap, vmin=vmin, vmax=vmax,
                shading="auto",
            )
            add_reward_onset_line(ax)

            if cluster_masks and gkey in cluster_masks and seg in cluster_masks[gkey]:
                mask = cluster_masks[gkey][seg]
                ax.contour(
                    times, freqs, mask.astype(float),
                    levels=[0.5], colors="k", linewidths=0.6,
                )

            if ri == 0:
                ax.set_title(seg, fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"{GROUP_LABELS[gkey]}\nFreq (Hz)", fontsize=8)
            if ri == n_rows - 1:
                ax.set_xlabel("Time (s)", fontsize=8)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlim(ERSP_CFG.display_tmin, ERSP_CFG.display_tmax)

    fig.suptitle(title or f"ERSP at {channel}", fontsize=11, y=1.01)
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label="Power (dB)")
    fig.tight_layout()

    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_ersp_difference_map(
    diff_data: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    channel: str = "C3",
    cluster_mask: Optional[np.ndarray] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Active-minus-Sham difference heatmap with cluster contours.

    Parameters
    ----------
    diff_data : ndarray (n_freqs, n_times)
        Mean difference (Active pooled - Sham).
    cluster_mask : optional bool array, same shape, marking significant clusters.
    vmin, vmax : if None, auto-scale symmetrically from percentiles within
        the display window.
    """
    apply_style()
    disp_mask = _display_crop_mask(times)

    if vmin is None or vmax is None:
        cropped = diff_data[:, disp_mask]
        clim = max(abs(np.percentile(cropped, 2)),
                   abs(np.percentile(cropped, 98)))
        clim = round(clim, 2) if clim > 0.1 else 0.5
        if vmin is None:
            vmin = -clim
        if vmax is None:
            vmax = clim

    cmap, _, _ = ersp_cmap(vmin, vmax)
    fig, ax = plt.subplots(figsize=(8, 4))

    im = ax.pcolormesh(
        times, freqs, diff_data,
        cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
    )
    add_reward_onset_line(ax)

    if cluster_mask is not None and cluster_mask.any():
        ax.contour(
            times, freqs, cluster_mask.astype(float),
            levels=[0.5], colors="k", linewidths=1.0,
        )

    ax.set_xlim(ERSP_CFG.display_tmin, ERSP_CFG.display_tmax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title or f"Active \u2212 Sham at {channel}")
    fig.colorbar(im, ax=ax, label="\u0394 Power (dB)")
    fig.tight_layout()

    if save_path:
        save_fig(fig, save_path)
    return fig
