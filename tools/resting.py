"""Resting-state spectral analysis (PIPELINE.md §7).

Computes Welch PSD for eyes-open and eyes-closed resting-state recordings
(EOPRE, ECPRE, EOPOST, ECPOST) across sessions 1, 3, 5, 6.

Key analyses:
- Absolute and relative band power (theta, alpha, SMR, beta, high-beta)
- Change scores (session 6 minus session 1) for persistence
- IAF estimation from eyes-closed baseline (optional covariate)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import BANDS, RESTING as RESTING_CFG, RestingConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PsdResult:
    """PSD result for one resting-state recording."""
    subject: str
    resting_type: str              # EOPRE, ECPRE, EOPOST, ECPOST
    session: int
    freqs: np.ndarray              # (n_freqs,)
    psd: np.ndarray                # (n_channels, n_freqs) µV²/Hz
    channel_names: List[str]
    band_power: Dict[str, Dict[str, float]]   # band → channel → abs power
    band_power_rel: Dict[str, Dict[str, float]]  # relative power
    n_clean_segments: int = 0
    n_total_segments: int = 0


# ---------------------------------------------------------------------------
# Welch PSD
# ---------------------------------------------------------------------------


def compute_psd(
    raw,
    cfg: RestingConfig = RESTING_CFG,
    channel_picks: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute Welch PSD on a continuous raw recording.

    Parameters
    ----------
    raw : mne.io.Raw
        Resting-state recording (preprocessed).
    cfg : RestingConfig
    channel_picks : list of str, optional
        Restrict to these channels.

    Returns
    -------
    freqs : ndarray (n_freqs,)
    psd : ndarray (n_channels, n_freqs) in V²/Hz
    ch_names : list of str
    """
    import mne

    picks = channel_picks or list(cfg.channels)
    raw_pick = raw.copy().pick(picks)
    ch_names = list(raw_pick.ch_names)

    n_fft = int(cfg.welch_window_sec * raw_pick.info["sfreq"])
    n_overlap = int(n_fft * cfg.welch_overlap)

    spectrum = raw_pick.compute_psd(
        method="welch",
        fmin=1.0,
        fmax=45.0,
        n_fft=n_fft,
        n_overlap=n_overlap,
        verbose="WARNING",
    )
    psd = spectrum.get_data()   # (n_channels, n_freqs) V²/Hz
    freqs = spectrum.freqs

    return freqs, psd, ch_names


# ---------------------------------------------------------------------------
# Band power extraction
# ---------------------------------------------------------------------------


def extract_band_powers(
    freqs: np.ndarray,
    psd: np.ndarray,
    channel_names: List[str],
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Compute absolute and relative band power.

    Returns
    -------
    abs_power : dict[band][channel] → float (µV²)
    rel_power : dict[band][channel] → float (fraction of total 1-45 Hz)
    """
    if bands is None:
        bands = dict(BANDS)

    freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    total_power = psd.sum(axis=1) * freq_res  # per channel

    abs_power: Dict[str, Dict[str, float]] = {}
    rel_power: Dict[str, Dict[str, float]] = {}

    for band_name, (flo, fhi) in bands.items():
        mask = (freqs >= flo) & (freqs < fhi)
        if not mask.any():
            continue
        bp = psd[:, mask].sum(axis=1) * freq_res  # per channel

        abs_power[band_name] = {}
        rel_power[band_name] = {}
        for ci, ch in enumerate(channel_names):
            abs_power[band_name][ch] = float(bp[ci]) * 1e12  # V² → µV²
            rel_power[band_name][ch] = (
                float(bp[ci] / total_power[ci]) if total_power[ci] > 0 else 0.0
            )

    return abs_power, rel_power


# ---------------------------------------------------------------------------
# IAF estimation
# ---------------------------------------------------------------------------


def estimate_iaf(
    freqs: np.ndarray,
    psd: np.ndarray,
    channel_names: List[str],
    alpha_range: Tuple[float, float] = (7.0, 14.0),
    target_channels: Tuple[str, ...] = ("Pz", "Oz", "O1", "O2"),
) -> Optional[float]:
    """Estimate individual alpha frequency (IAF) as peak in alpha range.

    Uses the posterior channels with the clearest alpha peak.
    Returns None if no clear peak is found.
    """
    from scipy.signal import find_peaks

    mask = (freqs >= alpha_range[0]) & (freqs <= alpha_range[1])
    if not mask.any():
        return None

    alpha_freqs = freqs[mask]

    # Average PSD across available target channels
    ch_idx = [i for i, ch in enumerate(channel_names) if ch in target_channels]
    if not ch_idx:
        ch_idx = list(range(len(channel_names)))

    avg_psd = psd[ch_idx][:, mask].mean(axis=0)

    peaks, properties = find_peaks(avg_psd, prominence=avg_psd.std() * 0.5)
    if len(peaks) == 0:
        logger.warning("No alpha peak found in %.1f-%.1f Hz", *alpha_range)
        return None

    # Take the most prominent peak
    best = peaks[np.argmax(properties["prominences"])]
    iaf = float(alpha_freqs[best])
    logger.info("IAF estimated: %.1f Hz", iaf)
    return iaf


# ---------------------------------------------------------------------------
# Full resting pipeline for one recording
# ---------------------------------------------------------------------------


def analyze_resting(
    raw,
    resting_type: str,
    session: int,
    cfg: RestingConfig = RESTING_CFG,
    subject: str = "",
    channel_picks: Optional[List[str]] = None,
) -> PsdResult:
    """Compute PSD and band powers for one resting-state recording.

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed resting-state recording.
    resting_type : str
        One of EOPRE, ECPRE, EOPOST, ECPOST.
    session : int
    cfg : RestingConfig
    subject : str
    channel_picks : list of str, optional

    Returns
    -------
    PsdResult
    """
    freqs, psd, ch_names = compute_psd(raw, cfg, channel_picks)
    abs_bp, rel_bp = extract_band_powers(freqs, psd, ch_names, cfg.bands)

    logger.info(
        "Resting PSD: %s %s session %d, %d channels",
        subject, resting_type, session, len(ch_names),
    )

    return PsdResult(
        subject=subject,
        resting_type=resting_type,
        session=session,
        freqs=freqs,
        psd=psd,
        channel_names=ch_names,
        band_power=abs_bp,
        band_power_rel=rel_bp,
    )


def compute_psd_with_rejection(
    raw,
    cfg: RestingConfig = RESTING_CFG,
    channel_picks: Optional[List[str]] = None,
    segment_sec: float = 2.0,
    reject_uv: float = 200.0,
) -> Tuple[np.ndarray, np.ndarray, List[str], int, int]:
    """Compute Welch PSD on clean segments after amplitude-based rejection.

    Splits the continuous recording into fixed-length epochs, rejects those
    exceeding *reject_uv* peak-to-peak, then computes PSD on surviving
    segments.

    Returns
    -------
    freqs, psd, ch_names, n_clean, n_total
    """
    import mne

    picks = channel_picks or list(cfg.channels)

    events = mne.make_fixed_length_events(raw, duration=segment_sec)
    epochs = mne.Epochs(
        raw, events, tmin=0, tmax=segment_sec,
        picks=picks, baseline=None, preload=True,
        reject={"eeg": reject_uv * 1e-6},
        verbose="WARNING",
    )
    n_total = len(events)
    n_clean = len(epochs)

    logger.info(
        "Segment rejection: %d/%d clean (%.1f%% rejected)",
        n_clean, n_total, (1 - n_clean / max(n_total, 1)) * 100,
    )

    if n_clean == 0:
        logger.error("All segments rejected — returning empty PSD")
        return np.array([]), np.empty((0, 0)), list(picks), 0, n_total

    spectrum = epochs.compute_psd(
        method="welch",
        fmin=1.0,
        fmax=45.0,
        n_fft=int(cfg.welch_window_sec * raw.info["sfreq"]),
        n_overlap=int(cfg.welch_window_sec * raw.info["sfreq"] * cfg.welch_overlap),
        verbose="WARNING",
    )
    psd = spectrum.get_data().mean(axis=0)  # average across epochs → (n_ch, n_freqs)
    freqs = spectrum.freqs

    return freqs, psd, list(epochs.ch_names), n_clean, n_total


def analyze_resting_clean(
    raw,
    resting_type: str,
    session: int,
    cfg: RestingConfig = RESTING_CFG,
    subject: str = "",
    channel_picks: Optional[List[str]] = None,
    reject_uv: float = 200.0,
) -> PsdResult:
    """PSD with segment-based artifact rejection (for fully preprocessed data).

    Use this instead of :func:`analyze_resting` when the raw has been through
    the full ``preprocess()`` pipeline (bad channels, ICA, re-reference).
    """
    freqs, psd, ch_names, n_clean, n_total = compute_psd_with_rejection(
        raw, cfg, channel_picks, segment_sec=cfg.welch_window_sec,
        reject_uv=reject_uv,
    )

    if psd.size == 0:
        return PsdResult(
            subject=subject, resting_type=resting_type, session=session,
            freqs=freqs, psd=psd, channel_names=ch_names,
            band_power={}, band_power_rel={},
            n_clean_segments=n_clean, n_total_segments=n_total,
        )

    abs_bp, rel_bp = extract_band_powers(freqs, psd, ch_names, cfg.bands)

    logger.info(
        "Resting PSD (clean): %s %s session %d, %d channels, %d/%d segments",
        subject, resting_type, session, len(ch_names), n_clean, n_total,
    )

    return PsdResult(
        subject=subject,
        resting_type=resting_type,
        session=session,
        freqs=freqs,
        psd=psd,
        channel_names=ch_names,
        band_power=abs_bp,
        band_power_rel=rel_bp,
        n_clean_segments=n_clean,
        n_total_segments=n_total,
    )


# ---------------------------------------------------------------------------
# Change scores
# ---------------------------------------------------------------------------


def compute_change_scores(
    pre: PsdResult,
    post: PsdResult,
) -> Dict[str, Dict[str, float]]:
    """Compute band-power change (post minus pre) for each channel.

    Typically: session 6 vs session 1, or POST vs PRE within session.

    Returns
    -------
    dict[band][channel] → delta (µV²)
    """
    deltas: Dict[str, Dict[str, float]] = {}
    for band in pre.band_power:
        deltas[band] = {}
        for ch in pre.band_power[band]:
            if ch in post.band_power.get(band, {}):
                deltas[band][ch] = (
                    post.band_power[band][ch] - pre.band_power[band][ch]
                )
    return deltas


# ---------------------------------------------------------------------------
# Alpha reactivity
# ---------------------------------------------------------------------------


def compute_alpha_reactivity(
    eo_result: PsdResult,
    ec_result: PsdResult,
    alpha_band: Tuple[float, float] = (8.0, 12.0),
) -> Dict[str, float]:
    """Compute alpha reactivity (EO minus EC alpha power) per channel.

    Positive values indicate EC > EO (normal alpha blocking / reactivity).
    Returned as EC - EO so that positive = reactive.
    """
    reactivity: Dict[str, float] = {}
    for ch in eo_result.channel_names:
        eo_alpha = eo_result.band_power.get("alpha", {}).get(ch)
        ec_alpha = ec_result.band_power.get("alpha", {}).get(ch)
        if eo_alpha is not None and ec_alpha is not None:
            reactivity[ch] = ec_alpha - eo_alpha
    return reactivity


# ---------------------------------------------------------------------------
# HDF5 persistence
# ---------------------------------------------------------------------------


def save_psd(result: PsdResult, path: Path) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["subject"] = result.subject
        f.attrs["resting_type"] = result.resting_type
        f.attrs["session"] = result.session
        f.attrs["channel_names"] = result.channel_names
        f.create_dataset("freqs", data=result.freqs)
        f.create_dataset("psd", data=result.psd)

        for band, ch_vals in result.band_power.items():
            for ch, val in ch_vals.items():
                f.attrs[f"abs/{band}/{ch}"] = val
        for band, ch_vals in result.band_power_rel.items():
            for ch, val in ch_vals.items():
                f.attrs[f"rel/{band}/{ch}"] = val


def load_psd(path: Path) -> PsdResult:
    import h5py

    with h5py.File(path, "r") as f:
        abs_bp: Dict[str, Dict[str, float]] = {}
        rel_bp: Dict[str, Dict[str, float]] = {}
        for key in f.attrs:
            if key.startswith("abs/"):
                _, band, ch = key.split("/")
                abs_bp.setdefault(band, {})[ch] = float(f.attrs[key])
            elif key.startswith("rel/"):
                _, band, ch = key.split("/")
                rel_bp.setdefault(band, {})[ch] = float(f.attrs[key])

        return PsdResult(
            subject=str(f.attrs["subject"]),
            resting_type=str(f.attrs["resting_type"]),
            session=int(f.attrs["session"]),
            freqs=f["freqs"][:],
            psd=f["psd"][:],
            channel_names=list(f.attrs["channel_names"]),
            band_power=abs_bp,
            band_power_rel=rel_bp,
        )
