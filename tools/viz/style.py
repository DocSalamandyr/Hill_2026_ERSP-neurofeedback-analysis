"""Shared publication style for all ERSP mechanism paper figures.

Defines colormaps, group colors, font sizes, and matplotlib rcParams
for consistent journal-quality output (300+ DPI).
"""

from __future__ import annotations

from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ..config import GROUP_COLORS

# ---------------------------------------------------------------------------
# Group aesthetics
# ---------------------------------------------------------------------------

COLORS: Dict[str, str] = dict(GROUP_COLORS)

GROUP_LABELS: Dict[str, str] = {
    "c3_smr":  "C3 SMR",
    "c3_beta": "C3 Beta",
    "c4_smr":  "C4 SMR",
    "sham":    "Sham",
}

GROUP_ORDER = ("c3_smr", "c3_beta", "c4_smr", "sham")

# ---------------------------------------------------------------------------
# Colormap for ERSP heatmaps (blue=ERD, red=ERS)
# ---------------------------------------------------------------------------


def ersp_cmap(vmin: float = -3.0, vmax: float = 3.0):
    """Return a diverging colormap centred on zero for ERSP dB plots."""
    return plt.cm.RdBu_r, vmin, vmax


# ---------------------------------------------------------------------------
# Matplotlib rc overrides for publication
# ---------------------------------------------------------------------------


def apply_style() -> None:
    """Set matplotlib rcParams for publication-quality figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.figsize": (7.0, 5.0),
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,        # editable text in PDF
        "ps.fonttype": 42,
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def add_reward_onset_line(ax, color: str = "k", ls: str = "--", lw: float = 0.8):
    """Draw a vertical line at time=0 (reward onset)."""
    ax.axvline(0, color=color, ls=ls, lw=lw, zorder=5)


def add_baseline_shade(ax, tmin: float = -0.1, tmax: float = 0.0,
                       color: str = "0.9"):
    """Shade the baseline period."""
    ax.axvspan(tmin, tmax, color=color, alpha=0.4, zorder=0)


def save_fig(fig, path, formats=("pdf", "png")):
    """Save a figure in multiple formats."""
    from pathlib import Path as P
    p = P(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(p.with_suffix(f".{fmt}"))
