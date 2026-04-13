"""Figure 6: Reward-evoked ERP waveforms (PAPER.md §3).

Grand-average ERPs at C3, C4, Pz for all groups, Sessions 1 and 5.
Shaded component windows: P50 (40-80 ms), N1 (80-140 ms), P2 (140-260 ms).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..config import ERP as ERP_CFG
from .style import COLORS, GROUP_LABELS, GROUP_ORDER, apply_style, save_fig


def plot_erp_waveforms(
    erp_data: Dict[str, Dict[str, np.ndarray]],
    times: np.ndarray,
    channels: List[str] = None,
    sessions_shown: List[str] = None,
    title: str = "Reward-Evoked ERPs",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot grand-average ERP waveforms with component windows.

    Parameters
    ----------
    erp_data : dict[group_key][session_label] → ndarray (n_channels, n_times)
        Grand-average ERP in µV.
    times : ndarray (seconds)
    channels : list of str
        Channel names (rows of the array and subplot panels).
    sessions_shown : list of str
        Session labels (e.g. ["Session 1", "Session 5"]).
    """
    apply_style()
    if channels is None:
        channels = ["C3", "C4", "Pz"]
    if sessions_shown is None:
        sessions_shown = ["Session 1", "Session 5"]

    n_channels = len(channels)
    n_sessions = len(sessions_shown)

    fig, axes = plt.subplots(
        n_channels, n_sessions, figsize=(5 * n_sessions, 3 * n_channels),
        sharex=True, sharey="row",
    )
    if n_channels == 1:
        axes = axes[np.newaxis, :]
    if n_sessions == 1:
        axes = axes[:, np.newaxis]

    # Component windows
    comp_windows = [
        ("P50", ERP_CFG.p50_window, "#fee08b"),
        ("N1",  ERP_CFG.n1_window,  "#d9ef8b"),
        ("P2",  ERP_CFG.p2_window,  "#abd9e9"),
    ]

    for ci, ch in enumerate(channels):
        for si, slabel in enumerate(sessions_shown):
            ax = axes[ci, si]

            # Shade component windows (convert s → ms)
            for comp_name, (t0, t1), color in comp_windows:
                ax.axvspan(t0 * 1000, t1 * 1000, color=color, alpha=0.2, zorder=0)

            for gkey in GROUP_ORDER:
                data = erp_data.get(gkey, {}).get(slabel)
                if data is None:
                    continue
                ax.plot(times * 1000, data[ci], color=COLORS[gkey],
                        label=GROUP_LABELS[gkey] if ci == 0 and si == 0 else "_nolegend_")

            ax.axvline(0, color="k", ls="--", lw=0.6)
            ax.axhline(0, color="0.5", ls=":", lw=0.5)

            if ci == 0:
                ax.set_title(slabel, fontsize=9)
            if si == 0:
                ax.set_ylabel(f"{ch}\nAmplitude (µV)", fontsize=8)
            if ci == n_channels - 1:
                ax.set_xlabel("Time (ms)", fontsize=8)

    # Place component labels in top-left panel after y-limits are set
    ax0 = axes[0, 0]
    ymax = ax0.get_ylim()[1]
    for comp_name, (t0, t1), color in comp_windows:
        ax0.annotate(comp_name, xy=((t0 + t1) / 2 * 1000, ymax * 0.85),
                     ha="center", va="top", fontsize=7, color="0.3")

    axes[0, 0].legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    fig.suptitle(title, fontsize=11)

    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_erp_p2_focus(
    erp_data: Dict[str, Dict[str, np.ndarray]],
    times: np.ndarray,
    channels: List[str],
    c3_index: int = 0,
    title: str = "Reward-Evoked ERPs at C3 (Session-Averaged)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Single-panel session-averaged ERP at C3, zoomed to the P2 region.

    Parameters
    ----------
    erp_data : dict[group_key][session_label] -> ndarray (n_channels, n_times)
    times : ndarray (seconds)
    channels : list of str
    c3_index : index of C3 in the channels list
    """
    apply_style()

    if "C3" in channels:
        c3_index = channels.index("C3")

    fig, ax = plt.subplots(figsize=(6, 3.5))

    comp_windows = [
        ("P50", ERP_CFG.p50_window, "#fee08b", 0.15),
        ("N1",  ERP_CFG.n1_window,  "#d9ef8b", 0.15),
        ("P2",  ERP_CFG.p2_window,  "#abd9e9", 0.35),
    ]
    for comp_name, (t0, t1), color, alpha in comp_windows:
        ax.axvspan(t0 * 1000, t1 * 1000, color=color, alpha=alpha, zorder=0)

    t_ms = times * 1000

    for gkey in GROUP_ORDER:
        sess_data = erp_data.get(gkey, {})
        arrs = [v[c3_index] for v in sess_data.values() if v is not None]
        if not arrs:
            continue
        avg = np.mean(arrs, axis=0)
        ax.plot(t_ms, avg, color=COLORS[gkey], lw=1.8,
                label=GROUP_LABELS[gkey])

    ax.axvline(0, color="k", ls="--", lw=0.6)
    ax.axhline(0, color="0.5", ls=":", lw=0.5)
    ax.set_xlim(-50, 300)

    p2_mid = (ERP_CFG.p2_window[0] + ERP_CFG.p2_window[1]) / 2 * 1000
    ylims = ax.get_ylim()
    ax.text(p2_mid, ylims[1] * 0.92, "P2", ha="center", fontsize=9,
            fontweight="bold", color="#2166ac")

    ax.set_xlabel("Time (ms)", fontsize=9)
    ax.set_ylabel("Amplitude (µV)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    if save_path:
        save_fig(fig, save_path)
    return fig
