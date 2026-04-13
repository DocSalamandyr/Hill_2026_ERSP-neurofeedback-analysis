"""Bayes factor visualization: BF01 half-violin / evidence categorization.

Plots Bayes factors from planned contrasts to visually summarise the
strength of evidence for H0 (absence of effect) or H1.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .style import apply_style, save_fig

BF_THRESHOLDS = {
    "Strong H1": (0, 1 / 10),
    "Moderate H1": (1 / 10, 1 / 3),
    "Anecdotal": (1 / 3, 3),
    "Moderate H0": (3, 10),
    "Strong H0": (10, float("inf")),
}

BF_COLORS = {
    "Strong H1": "#d62728",
    "Moderate H1": "#ff7f0e",
    "Anecdotal": "#7f7f7f",
    "Moderate H0": "#2ca02c",
    "Strong H0": "#1f77b4",
}


def _classify_bf(bf01: float) -> str:
    for label, (lo, hi) in BF_THRESHOLDS.items():
        if lo <= bf01 < hi:
            return label
    return "Strong H0"


def plot_bf01_summary(
    names: Sequence[str],
    bf01_values: Sequence[float],
    title: str = "Bayes Factor Evidence Summary (BF₀₁)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Horizontal bar chart of BF01 values with evidence categorization.

    Parameters
    ----------
    names : sequence of str
        Contrast labels.
    bf01_values : sequence of float
        BF01 values (> 1 favours H0).
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(names))))

    log_bf = np.log10(np.array(bf01_values, dtype=float))
    colors = [BF_COLORS[_classify_bf(bf)] for bf in bf01_values]
    categories = [_classify_bf(bf) for bf in bf01_values]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, log_bf, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("log₁₀(BF₀₁)")
    ax.axvline(0, color="k", linewidth=0.5)

    for threshold, label in [(np.log10(1 / 3), "1/3"), (np.log10(3), "3"),
                              (np.log10(10), "10")]:
        ax.axvline(threshold, color="gray", linewidth=0.5, linestyle=":")
        ax.text(threshold, len(names) - 0.3, label, fontsize=7,
                ha="center", va="bottom", color="gray")

    for i, (bf, cat) in enumerate(zip(bf01_values, categories)):
        ax.text(
            log_bf[i] + 0.05 * np.sign(log_bf[i]),
            i, f"{bf:.2f} ({cat})",
            va="center", fontsize=7,
        )

    ax.set_title(title, fontsize=11)
    fig.tight_layout()

    if save_path:
        save_fig(fig, save_path)

    return fig
