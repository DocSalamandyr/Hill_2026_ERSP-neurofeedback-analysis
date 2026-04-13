"""Central analysis configuration for the ERSP mechanism paper.

Every tuneable parameter referenced in PIPELINE.md and PAPER.md is declared
here so that the entire pipeline can be driven from a single import.  Sensitivity
analysis variants are exposed as factory functions that return modified copies.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Sequence, Tuple

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path layout
#
#   /path/to/your/ERSP_data/
#     Data/           raw BDFs + hfinf2 sidecars (36 analyzable subjects)
#     Analyses/       intermediate working files (preprocessed .fif, epochs, …)
#     Derivatives/    final pipeline output (ERSP h5, figures, stats, …)
#     Related/        curated reference materials (bad channels, PACDEL CSVs, …)
#     .venv/          Python virtual environment
#
# The original recovered data at Dissertation/BDF Correct Name-PACDEL-Recovered/
# and Dissertation/Supplementary/ are treated as read-only archival.
# Override via environment variables for CI, tests, or a different machine.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent          # ERSP/

_DEVDISK = Path("/path/to/your/ERSP_data")

DATA_ROOT = Path(os.environ.get(
    "ERSP_DATA_ROOT",
    str(_DEVDISK / "Data"),
))
ANALYSES_ROOT = Path(os.environ.get(
    "ERSP_ANALYSES_ROOT",
    str(_DEVDISK / "Analyses"),
))
DERIVATIVES_ROOT = Path(os.environ.get(
    "ERSP_DERIVATIVES_ROOT",
    str(_DEVDISK / "Derivatives"),
))
RELATED_ROOT = Path(os.environ.get(
    "ERSP_RELATED_ROOT",
    str(_DEVDISK / "Related"),
))

if not DATA_ROOT.exists() and "ERSP_DATA_ROOT" not in os.environ:
    _logger.warning(
        "DATA_ROOT %s does not exist (data directory not found?). "
        "Set ERSP_DATA_ROOT env var to override.", DATA_ROOT,
    )

# ---------------------------------------------------------------------------
# Frequency bands (Hz)
# ---------------------------------------------------------------------------

BANDS: Dict[str, Tuple[float, float]] = {
    "theta":     (4.0, 7.0),
    "alpha":     (8.0, 12.0),
    "smr":       (12.0, 15.0),
    "beta":      (15.0, 18.0),
    "high_beta": (18.0, 25.0),
}

# ---------------------------------------------------------------------------
# Group definitions  (subject IDs will be filled from study metadata)
# ---------------------------------------------------------------------------

GROUPS: Dict[str, str] = {
    "c3_smr":  "C3 SMR 12-15 Hz",
    "c3_beta": "C3 Beta 15-18 Hz",
    "c4_smr":  "C4 SMR 12-15 Hz",
    "sham":    "Active-placebo sham",
}

GROUP_REWARD_BAND: Dict[str, str] = {
    "c3_smr":  "smr",
    "c3_beta": "beta",
    "c4_smr":  "smr",
    "sham":    "smr",      # nominal; sham has no true reward band
}

GROUP_TRAINING_CHANNEL: Dict[str, str] = {
    "c3_smr":  "C3",
    "c3_beta": "C3",
    "c4_smr":  "C4",
    "sham":    "C3",       # varies by nominal arm; default for pooled analyses
}

GROUP_COLORS: Dict[str, str] = {
    "c3_smr":  "#1f77b4",  # blue
    "c3_beta": "#d62728",  # red
    "c4_smr":  "#2ca02c",  # green
    "sham":    "#7f7f7f",  # gray
}

# ---------------------------------------------------------------------------
# Recording-type aliases and known data issues
# ---------------------------------------------------------------------------

REC_TYPE_ALIASES: Dict[str, str] = {
    "EOPORE": "EOPRE",   # subject 129 misspelling
}

CORRUPT_BFBS: FrozenSet[Tuple[str, int]] = frozenset({
    ("108", 3),   # 19K header-only — server-verified unrecoverable
    ("110", 5),   # 19K header-only — server-verified unrecoverable
    ("131", 3),   # 19K header-only — server-verified unrecoverable
})

TRUNCATED_BFBS: Dict[str, Tuple[int, ...]] = {
    "125": (5,),  # 62M, ~1/3 normal — check trial count at runtime
}

EDF_ONLY_RESTING: Dict[str, Tuple[int, ...]] = {
    "104": (5,),  # session 5 resting: 16-bit EDF, harmonized with BDF pipeline
    "138": (6,),  # session 6 resting: 16-bit EDF, harmonized with BDF pipeline
}

# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

BFB_SESSIONS: Tuple[int, ...] = (1, 3, 5, 6)
RESTING_SESSIONS: Tuple[int, ...] = (1, 3, 5, 6)
RESTING_TYPES: Tuple[str, ...] = ("EOPRE", "ECPRE", "EOPOST", "ECPOST")

# ---------------------------------------------------------------------------
# Channels of interest
# ---------------------------------------------------------------------------

PRIMARY_CHANNELS: Tuple[str, ...] = ("C3", "C4")
SECONDARY_CHANNELS: Tuple[str, ...] = ("Cz", "Fz", "Pz")
ERP_CHANNELS: Tuple[str, ...] = ("C3", "C4", "Pz")
EOG_CHANNEL_INDICES: Tuple[int, int] = (64, 65)   # 0-indexed: VEOU, VEOL

# ---------------------------------------------------------------------------
# BioSemi event codes
# ---------------------------------------------------------------------------

REWARD_EVENT_CODE: int = 0x0100   # 256
EXPECTED_REWARD_COUNT: Tuple[int, int] = (500, 800)  # flag outside range

# ---------------------------------------------------------------------------
# Preprocessing dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessConfig:
    """Parameters for the preprocessing pipeline (PIPELINE.md §3)."""

    highpass_hz: float = 0.1
    notch_hz: float = 60.0
    montage_name: str = "biosemi64"
    reref: str = "average"
    exclude_eog: bool = True

    # Bad channel detection
    bad_channel_method: str = "ransac"

    # ICA
    ica_method: str = "infomax"
    ica_n_components: float = 0.99          # explain 99% variance
    iclabel_threshold: float = 0.80         # reject IC if non-brain prob > this
    iclabel_reject_classes: Tuple[str, ...] = (
        "eye blink", "eye movement", "muscle artifact",
        "heart beat", "line noise", "channel noise",
    )

    # Sensitivity branch (Attack 5)
    sensitivity_highpass_hz: float = 0.16
    sensitivity_skip_ica: bool = True

    def __post_init__(self) -> None:
        if self.ica_method not in ("infomax", "fastica", "picard", "none"):
            raise ValueError(f"Invalid ica_method: {self.ica_method!r}")
        if not (0.0 < self.iclabel_threshold <= 1.0):
            raise ValueError(
                f"iclabel_threshold must be in (0, 1], got {self.iclabel_threshold}"
            )
        if self.highpass_hz <= 0:
            raise ValueError(f"highpass_hz must be > 0, got {self.highpass_hz}")


# ---------------------------------------------------------------------------
# Epoch / ERSP dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErspConfig:
    """Parameters for epoching + ERSP/ITC computation (PIPELINE.md §4-5)."""

    # Epoch window (seconds relative to reward onset)
    tmin: float = -0.5
    tmax: float = 1.5

    # Baseline (seconds)
    baseline_tmin: float = -0.1
    baseline_tmax: float = 0.0

    # Artifact rejection
    epoch_reject_method: str = "peak_to_peak"  # "peak_to_peak" or "autorej_statistical"
    reject_peak_to_peak_uv: float = 200.0      # µV; for peak_to_peak mode, or step-1 threshold for autorej
    min_clean_trials: int = 100

    # pop_autorej-equivalent parameters (only used when epoch_reject_method="autorej_statistical")
    autorej_startprob: float = 5.0             # starting z-score for joint probability
    autorej_maxrej_pct: float = 5.0            # max % of epochs rejected per iteration
    autorej_kurtosis_thresh: float = 6.0       # z-score threshold for kurtosis rejection

    # Early / late split
    early_late_minutes: float = 10.0        # first & last N minutes
    sliding_window_trials: int = 50

    # Time-frequency decomposition
    freqs_hz: Tuple[float, ...] = (3.0, 3.5, 4.0, 4.5, 5.0) + tuple(float(f) for f in range(6, 41))  # 3-40 Hz
    n_cycles_low: float = 3.0              # cycles at lowest freq
    n_cycles_high: float = 12.0            # cycles at highest freq
    method: str = "morlet"                  # or "stft" for cross-check

    # Analysis time windows (seconds)
    erd_window: Tuple[float, float] = (0.2, 0.8)
    ers_window: Tuple[float, float] = (0.1, 0.5)

    # Display crop for heatmaps (avoids adjacent-reward contamination at edges)
    display_tmin: float = -0.2
    display_tmax: float = 1.0

    def __post_init__(self) -> None:
        if not (self.tmin <= self.baseline_tmin < self.baseline_tmax <= 0):
            raise ValueError(
                f"Baseline [{self.baseline_tmin}, {self.baseline_tmax}] "
                f"must be within [tmin={self.tmin}, 0]"
            )
        freqs = self.freqs_hz
        if len(freqs) > 1 and not all(
            freqs[i] < freqs[i + 1] for i in range(len(freqs) - 1)
        ):
            raise ValueError("freqs_hz must be monotonically increasing")

    @property
    def n_cycles(self) -> List[float]:
        """Linearly spaced cycle counts matching *freqs_hz*."""
        n = len(self.freqs_hz)
        if n == 1:
            return [self.n_cycles_low]
        return [
            self.n_cycles_low + (self.n_cycles_high - self.n_cycles_low) * i / (n - 1)
            for i in range(n)
        ]

    @property
    def freqs_array(self):
        """Return freqs as a numpy array (deferred import)."""
        import numpy as np
        return np.array(self.freqs_hz)


# ---------------------------------------------------------------------------
# ERP dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErpConfig:
    """Parameters for ERP component extraction (PIPELINE.md §6)."""

    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 30.0
    channels: Tuple[str, ...] = ERP_CHANNELS

    # Component windows (seconds)
    p50_window: Tuple[float, float] = (0.040, 0.080)
    n1_window: Tuple[float, float] = (0.080, 0.140)
    p2_window: Tuple[float, float] = (0.140, 0.260)


# ---------------------------------------------------------------------------
# Resting-state dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RestingConfig:
    """Parameters for resting-state spectral analysis (PIPELINE.md §7)."""

    welch_window_sec: float = 2.0
    welch_overlap: float = 0.5              # fraction
    channels: Tuple[str, ...] = PRIMARY_CHANNELS
    bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: dict(BANDS))


# ---------------------------------------------------------------------------
# Statistics dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatsConfig:
    """Parameters for the statistical battery (PAPER.md §2b)."""

    cluster_n_permutations: int = 5000
    cluster_alpha: float = 0.05

    bonferroni_n_tests: int = 3             # 3 active-vs-sham comparisons
    bonferroni_alpha: float = 0.05 / 3      # ~0.017

    fdr_method: str = "fdr_bh"              # Benjamini-Hochberg

    bayes_prior: float = 0.707              # pingouin default Cauchy r

    def __post_init__(self) -> None:
        if not (0.0 < self.cluster_alpha < 1.0):
            raise ValueError(f"cluster_alpha must be in (0, 1), got {self.cluster_alpha}")
        if self.cluster_n_permutations < 1:
            raise ValueError(f"cluster_n_permutations must be > 0, got {self.cluster_n_permutations}")


# ---------------------------------------------------------------------------
# Sensitivity analysis factories
# ---------------------------------------------------------------------------


def sensitivity_baseline_short() -> ErspConfig:
    """Baseline -100 to 0 ms (matches dissertation)."""
    return ErspConfig(baseline_tmin=-0.1, baseline_tmax=0.0)


def sensitivity_baseline_long() -> ErspConfig:
    """Baseline -500 to -100 ms (pre-reward, longer window)."""
    return ErspConfig(baseline_tmin=-0.5, baseline_tmax=-0.1)


def sensitivity_minimal_preprocess() -> PreprocessConfig:
    """Dissertation-era processing: 0.16 Hz highpass, no ICA, no RANSAC."""
    return PreprocessConfig(
        highpass_hz=0.16,
        bad_channel_method="none",
        ica_method="none",
    )


def sensitivity_minimal_ersp() -> ErspConfig:
    """Dissertation-era epoch rejection: pop_autorej statistical method.

    Uses 1000 µV absolute threshold (step 1), iterative joint probability
    at 5 SD (step 2), and kurtosis at 6 SD (step 3) — matching EEGLAB
    pop_autorej defaults from the original analysis.
    """
    return ErspConfig(
        epoch_reject_method="autorej_statistical",
        reject_peak_to_peak_uv=1000.0,
    )


# ---------------------------------------------------------------------------
# Convenience: default configs
# ---------------------------------------------------------------------------

PREPROCESS = PreprocessConfig()
ERSP = ErspConfig()
ERP = ErpConfig()
RESTING = RestingConfig()
STATS = StatsConfig()
