"""Figure 5: Topographic ERSP maps (PAPER.md §3).

Scalp topographies of reward-band ERD magnitude (200-800 ms, dB)
for each group at Sessions 1 and 5 (Late).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .style import GROUP_LABELS, GROUP_ORDER, apply_style, ersp_cmap, save_fig


def plot_topo_ersp(
    topo_data: Dict[str, Dict[str, np.ndarray]],
    info,
    sessions_labels: List[str] = None,
    vmin: float = -2.0,
    vmax: float = 2.0,
    title: str = "Topographic ERD Maps (Reward Band, 200-800 ms)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot scalp topographies of ERD for each group × session.

    Parameters
    ----------
    topo_data : dict[group_key][session_label] → 1-D array (n_channels,)
        ERD magnitude per channel.
    info : mne.Info
        Channel info with montage for topomap layout.
    sessions_labels : list of str
        Column labels (e.g. ["S1 Late", "S5 Late"]).
    """
    import mne

    apply_style()
    if sessions_labels is None:
        sessions_labels = ["S1 Late", "S5 Late"]

    cmap, _, _ = ersp_cmap(vmin, vmax)
    n_rows = len(GROUP_ORDER)
    n_cols = len(sessions_labels)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for ri, gkey in enumerate(GROUP_ORDER):
        for ci, slabel in enumerate(sessions_labels):
            ax = axes[ri, ci]
            data = topo_data.get(gkey, {}).get(slabel)
            if data is None:
                ax.set_visible(False)
                continue

            mne.viz.plot_topomap(
                data, info, axes=ax, cmap=cmap, vlim=(vmin, vmax),
                show=False,
            )
            if ri == 0:
                ax.set_title(slabel, fontsize=9)
            if ci == 0:
                ax.set_ylabel(GROUP_LABELS[gkey], fontsize=9, rotation=0,
                              labelpad=50, va="center")

    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()

    if save_path:
        save_fig(fig, save_path)
    return fig
