"""Figure 4: Frequency crossover plot (PAPER.md §3).

X=frequency (8-25 Hz), Y=ERD magnitude (dB, 200-800 ms) at C3.
Separate lines for C3 SMR, C3 Beta, Sham with 95% CI bands.
Vertical shaded regions mark SMR (12-15 Hz) and Beta (15-18 Hz) bands.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..config import BANDS
from .style import COLORS, GROUP_LABELS, apply_style, save_fig


def plot_frequency_crossover(
    freq_profiles: Dict[str, np.ndarray],
    freq_profiles_se: Dict[str, np.ndarray],
    freqs: np.ndarray,
    freq_range: tuple = (8.0, 25.0),
    title: str = "Frequency Crossover at C3",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot ERD magnitude as a function of frequency for key groups.

    Parameters
    ----------
    freq_profiles : dict[group_key] → 1-D array (n_freqs,)
        Mean ERD (dB) at each frequency, averaged across the 200-800 ms
        window and across subjects.
    freq_profiles_se : dict[group_key] → 1-D array (n_freqs,)
        Standard error.
    freqs : ndarray
    freq_range : (low, high) Hz to display.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    f = freqs[mask]

    for gkey in freq_profiles:
        y = freq_profiles[gkey][mask]
        se = freq_profiles_se[gkey][mask]
        ax.plot(f, y, color=COLORS.get(gkey, "0.3"),
                label=GROUP_LABELS.get(gkey, gkey))
        ax.fill_between(f, y - 1.96 * se, y + 1.96 * se,
                        color=COLORS.get(gkey, "0.3"), alpha=0.15)

    # Shaded band regions
    smr = BANDS["smr"]
    beta = BANDS["beta"]
    ax.axvspan(smr[0], smr[1], color=COLORS["c3_smr"], alpha=0.08, label="_nolegend_")
    ax.axvspan(beta[0], beta[1], color=COLORS["c3_beta"], alpha=0.08, label="_nolegend_")
    ax.text(np.mean(smr), ax.get_ylim()[0] * 0.95, "SMR", ha="center",
            fontsize=7, color=COLORS["c3_smr"])
    ax.text(np.mean(beta), ax.get_ylim()[0] * 0.95, "Beta", ha="center",
            fontsize=7, color=COLORS["c3_beta"])

    ax.axhline(0, color="0.5", ls=":", lw=0.6)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("ERD (dB, 200-800 ms)")
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()

    if save_path:
        save_fig(fig, save_path)
    return fig
