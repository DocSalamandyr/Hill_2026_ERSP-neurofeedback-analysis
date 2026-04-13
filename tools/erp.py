"""Reward-evoked ERP component extraction (PIPELINE.md §6, PAPER.md §2c).

Uses a separate 0.5-30 Hz bandpass (different from ERSP filter path) to
extract P50, N1, and P2 components at C3, C4, Pz from reward-locked epochs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import ERP as ERP_CFG, ErpConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ErpComponentMeasure:
    """Peak / mean amplitude measurement for one component × channel."""
    component: str                 # "P50", "N1", "P2"
    channel: str
    peak_amplitude_uv: float
    peak_latency_sec: float
    mean_amplitude_uv: float


@dataclass
class ErpResult:
    """Full ERP result for one subject × session."""
    subject: str
    session: int
    times: np.ndarray              # (n_times,)
    evoked: np.ndarray             # (n_channels, n_times) µV
    channel_names: List[str]
    n_trials: int
    components: List[ErpComponentMeasure]


# ---------------------------------------------------------------------------
# Bandpass for ERP
# ---------------------------------------------------------------------------


def bandpass_for_erp(raw, cfg: ErpConfig = ERP_CFG):
    """Apply ERP-specific bandpass (0.5-30 Hz) to a *copy* of raw data.

    This is separate from the ERSP filter path to avoid low-frequency
    distortion that would affect time-frequency analysis.
    """
    raw_erp = raw.copy()
    raw_erp.filter(
        l_freq=cfg.bandpass_low_hz,
        h_freq=cfg.bandpass_high_hz,
        method="fir",
        phase="zero-double",
        verbose="WARNING",
    )
    return raw_erp


# ---------------------------------------------------------------------------
# Component detection
# ---------------------------------------------------------------------------


def _find_peak(
    signal: np.ndarray,
    times: np.ndarray,
    window: Tuple[float, float],
    polarity: str = "positive",
) -> Tuple[float, float]:
    """Find peak amplitude and latency within a time window.

    Parameters
    ----------
    signal : 1-D array (time points)
    times : 1-D array (seconds)
    window : (tmin, tmax) in seconds
    polarity : "positive" or "negative"

    Returns
    -------
    (amplitude, latency_sec)
    """
    mask = (times >= window[0]) & (times <= window[1])
    seg = signal[mask]
    seg_times = times[mask]

    if len(seg) == 0:
        return 0.0, 0.0

    if polarity == "positive":
        idx = np.argmax(seg)
    else:
        idx = np.argmin(seg)

    return float(seg[idx]), float(seg_times[idx])


def measure_components(
    evoked_data: np.ndarray,
    times: np.ndarray,
    channel_names: List[str],
    cfg: ErpConfig = ERP_CFG,
) -> List[ErpComponentMeasure]:
    """Extract P50, N1, P2 from an evoked waveform array.

    Parameters
    ----------
    evoked_data : ndarray, shape (n_channels, n_times)
        Already in µV.
    times : ndarray (seconds)
    channel_names : list of str
    cfg : ErpConfig

    Returns
    -------
    list of ErpComponentMeasure
    """
    measures: List[ErpComponentMeasure] = []
    components = [
        ("P50", cfg.p50_window, "positive"),
        ("N1",  cfg.n1_window,  "negative"),
        ("P2",  cfg.p2_window,  "positive"),
    ]

    for ci, ch in enumerate(channel_names):
        sig = evoked_data[ci]
        for comp_name, window, polarity in components:
            peak_amp, peak_lat = _find_peak(sig, times, window, polarity)

            # Mean amplitude in window
            mask = (times >= window[0]) & (times <= window[1])
            mean_amp = float(sig[mask].mean()) if mask.any() else 0.0

            measures.append(ErpComponentMeasure(
                component=comp_name,
                channel=ch,
                peak_amplitude_uv=peak_amp,
                peak_latency_sec=peak_lat,
                mean_amplitude_uv=mean_amp,
            ))

    return measures


# ---------------------------------------------------------------------------
# Full ERP pipeline for one set of epochs
# ---------------------------------------------------------------------------


def compute_erp(
    epochs,
    cfg: ErpConfig = ERP_CFG,
    subject: str = "",
    session: int = 0,
) -> ErpResult:
    """Compute evoked ERP and extract components from reward-locked epochs.

    The epochs should already have been created from ERP-bandpassed data
    (or the raw data should be bandpassed before epoching).

    Parameters
    ----------
    epochs : mne.Epochs
        Reward-locked epochs.
    cfg : ErpConfig
    subject, session : for labelling.

    Returns
    -------
    ErpResult
    """
    epochs_pick = epochs.copy().pick(list(cfg.channels))
    ch_names = list(epochs_pick.ch_names)

    # Apply baseline correction for ERP (use -200 to 0 ms)
    epochs_pick.apply_baseline((-0.2, 0.0))

    evoked = epochs_pick.average()
    data_uv = evoked.data * 1e6  # V → µV
    times = evoked.times

    components = measure_components(data_uv, times, ch_names, cfg)

    logger.info(
        "ERP computed: %s session %d, %d trials, %d channels",
        subject, session, len(epochs_pick), len(ch_names),
    )
    for m in components:
        logger.debug(
            "  %s @ %s: peak=%.2f µV @ %.0f ms, mean=%.2f µV",
            m.component, m.channel,
            m.peak_amplitude_uv, m.peak_latency_sec * 1000,
            m.mean_amplitude_uv,
        )

    return ErpResult(
        subject=subject,
        session=session,
        times=times,
        evoked=data_uv,
        channel_names=ch_names,
        n_trials=len(epochs_pick),
        components=components,
    )


# ---------------------------------------------------------------------------
# HDF5 persistence
# ---------------------------------------------------------------------------


def save_erp(result: ErpResult, path: Path) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["subject"] = result.subject
        f.attrs["session"] = result.session
        f.attrs["n_trials"] = result.n_trials
        f.attrs["channel_names"] = result.channel_names
        f.create_dataset("times", data=result.times)
        f.create_dataset("evoked", data=result.evoked)

        for i, m in enumerate(result.components):
            grp = f.create_group(f"component/{i}")
            grp.attrs["component"] = m.component
            grp.attrs["channel"] = m.channel
            grp.attrs["peak_amplitude_uv"] = m.peak_amplitude_uv
            grp.attrs["peak_latency_sec"] = m.peak_latency_sec
            grp.attrs["mean_amplitude_uv"] = m.mean_amplitude_uv


def load_erp(path: Path) -> ErpResult:
    import h5py

    with h5py.File(path, "r") as f:
        components = []
        i = 0
        while f"component/{i}" in f:
            grp = f[f"component/{i}"]
            components.append(ErpComponentMeasure(
                component=str(grp.attrs["component"]),
                channel=str(grp.attrs["channel"]),
                peak_amplitude_uv=float(grp.attrs["peak_amplitude_uv"]),
                peak_latency_sec=float(grp.attrs["peak_latency_sec"]),
                mean_amplitude_uv=float(grp.attrs["mean_amplitude_uv"]),
            ))
            i += 1

        return ErpResult(
            subject=str(f.attrs["subject"]),
            session=int(f.attrs["session"]),
            times=f["times"][:],
            evoked=f["evoked"][:],
            channel_names=list(f.attrs["channel_names"]),
            n_trials=int(f.attrs["n_trials"]),
            components=components,
        )
