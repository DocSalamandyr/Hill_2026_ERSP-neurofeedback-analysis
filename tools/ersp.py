"""ERSP and ITC computation with single-trial normalization (PIPELINE.md §5).

Implements the Grandchamp & Delorme (2011) procedure:
  1. Compute Morlet TFR for each single trial  → power (freq × time)
  2. Convert each trial to dB relative to its *own* baseline period
  3. Average the dB-normalised spectrograms across trials  → ERSP
  4. ITC is computed from the complex TFR coefficients (phase consistency)

Also extracts scalar ERD/ERS metrics for the statistical battery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import BANDS, ERSP as ERSP_CFG, ErspConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ErspResult:
    """Arrays and scalar metrics from one subject × session × channel set."""
    subject: str
    session: int
    freqs: np.ndarray              # (n_freqs,)
    times: np.ndarray              # (n_times,)
    ersp: np.ndarray               # (n_channels, n_freqs, n_times) dB
    itc: np.ndarray                # (n_channels, n_freqs, n_times) 0-1
    channel_names: List[str]
    n_trials: int
    scalars: Dict[str, float]      # band-window scalar summaries


# ---------------------------------------------------------------------------
# Single-trial TFR
# ---------------------------------------------------------------------------


def _morlet_tfr_single_trials(epochs, cfg: ErspConfig = ERSP_CFG):
    """Compute per-trial Morlet power and complex coefficients.

    Returns
    -------
    power : ndarray, shape (n_trials, n_channels, n_freqs, n_times)
        Squared magnitude of the wavelet transform.
    complex_tfr : ndarray, same shape
        Complex coefficients (for ITC).
    times : ndarray
    freqs : ndarray
    """
    import mne
    from mne.time_frequency import tfr_array_morlet

    data = epochs.get_data(copy=True)  # (n_trials, n_channels, n_times_raw)
    sfreq = epochs.info["sfreq"]
    freqs = cfg.freqs_array
    n_cycles = np.array(cfg.n_cycles)

    # tfr_array_morlet returns complex coefficients
    # shape: (n_trials, n_channels, n_freqs, n_times_tfr)
    tfr_complex = tfr_array_morlet(
        data, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles,
        output="complex", verbose="WARNING",
    )

    power = np.abs(tfr_complex) ** 2
    times = epochs.times  # MNE aligns TFR times to epoch times

    return power, tfr_complex, times, freqs


def _stft_tfr_single_trials(epochs, cfg: ErspConfig = ERSP_CFG):
    """STFT cross-check method (Hanning windows, fixed length).

    Provides an alternative to Morlet for convergence testing against
    EEGLAB ``newtimef`` or Nogn nf-dsp.
    """
    from mne.time_frequency import tfr_array_multitaper

    data = epochs.get_data(copy=True)
    sfreq = epochs.info["sfreq"]
    freqs = cfg.freqs_array
    n_cycles = np.array(cfg.n_cycles)

    tfr_complex = tfr_array_multitaper(
        data, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles,
        output="complex", verbose="WARNING",
    )
    power = np.abs(tfr_complex) ** 2
    times = epochs.times
    return power, tfr_complex, times, freqs


# ---------------------------------------------------------------------------
# Grandchamp & Delorme single-trial normalization
# ---------------------------------------------------------------------------


def _baseline_indices(times: np.ndarray, cfg: ErspConfig) -> np.ndarray:
    """Return boolean mask for the baseline period in *times*."""
    return (times >= cfg.baseline_tmin) & (times <= cfg.baseline_tmax)


def single_trial_normalize(
    power: np.ndarray,
    times: np.ndarray,
    cfg: ErspConfig = ERSP_CFG,
) -> np.ndarray:
    """Grandchamp & Delorme (2011) single-trial dB normalization.

    For each trial independently:
        1. Compute mean baseline power at each frequency
        2. Convert the full spectrogram to dB: 10 * log10(power / baseline)

    Then average across trials.

    Parameters
    ----------
    power : ndarray, shape (n_trials, n_channels, n_freqs, n_times)
    times : ndarray, shape (n_times,)
    cfg : ErspConfig

    Returns
    -------
    ersp : ndarray, shape (n_channels, n_freqs, n_times)
        Trial-averaged dB-normalised ERSP.
    """
    bl_mask = _baseline_indices(times, cfg)
    if not bl_mask.any():
        raise ValueError(
            f"No time points in baseline [{cfg.baseline_tmin}, {cfg.baseline_tmax}]"
        )

    # Mean power in baseline per trial, per channel, per freq
    # shape: (n_trials, n_channels, n_freqs, 1)
    bl_mean = power[:, :, :, bl_mask].mean(axis=-1, keepdims=True)

    # Avoid log(0)
    bl_mean = np.maximum(bl_mean, np.finfo(float).eps)
    power_safe = np.maximum(power, np.finfo(float).eps)

    # dB conversion per trial
    ersp_trials = 10.0 * np.log10(power_safe / bl_mean)

    # Average across trials
    ersp = ersp_trials.mean(axis=0)  # (n_channels, n_freqs, n_times)
    return ersp


# ---------------------------------------------------------------------------
# ITC
# ---------------------------------------------------------------------------


def compute_itc(
    complex_tfr: np.ndarray,
) -> np.ndarray:
    """Inter-trial coherence from complex TFR coefficients.

    ITC = | mean_over_trials( tfr / |tfr| ) |

    Parameters
    ----------
    complex_tfr : ndarray, shape (n_trials, n_channels, n_freqs, n_times)

    Returns
    -------
    itc : ndarray, shape (n_channels, n_freqs, n_times)  in [0, 1]
    """
    magnitude = np.abs(complex_tfr)
    magnitude = np.maximum(magnitude, np.finfo(float).eps)
    normalised = complex_tfr / magnitude
    itc = np.abs(normalised.mean(axis=0))
    return itc


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------


def _time_mask(times: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
    return (times >= window[0]) & (times <= window[1])


def _freq_mask(freqs: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    return (freqs >= band[0]) & (freqs < band[1])


def extract_scalar_metrics(
    ersp: np.ndarray,
    itc: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    channel_names: List[str],
    reward_band: str,
    cfg: ErspConfig = ERSP_CFG,
) -> Dict[str, float]:
    """Extract per-band, per-window scalar summaries from ERSP/ITC.

    Returns a flat dict with keys like ``"erd_smr_C3"`` and ``"ers_theta_C3"``.
    """
    metrics: Dict[str, float] = {}
    erd_t = _time_mask(times, cfg.erd_window)
    ers_t = _time_mask(times, cfg.ers_window)

    for band_name, (flo, fhi) in BANDS.items():
        f_mask = _freq_mask(freqs, (flo, fhi))
        if not f_mask.any():
            continue

        for ci, ch in enumerate(channel_names):
            # ERSP mean dB in the band × window
            erd_val = float(ersp[ci, f_mask][:, erd_t].mean())
            ers_val = float(ersp[ci, f_mask][:, ers_t].mean())
            metrics[f"erd_{band_name}_{ch}"] = erd_val
            metrics[f"ers_{band_name}_{ch}"] = ers_val

            # ITC in the same windows
            itc_erd_val = float(itc[ci, f_mask][:, erd_t].mean())
            metrics[f"itc_{band_name}_{ch}"] = itc_erd_val

    # Primary scalar: ERD in the subject's reward band at C3/C4
    rb = BANDS.get(reward_band)
    if rb is not None:
        f_mask = _freq_mask(freqs, rb)
        for ci, ch in enumerate(channel_names):
            metrics[f"primary_erd_{ch}"] = float(
                ersp[ci, f_mask][:, erd_t].mean()
            )

    return metrics


# ---------------------------------------------------------------------------
# Full ERSP pipeline for one set of epochs
# ---------------------------------------------------------------------------


def compute_ersp(
    epochs,
    reward_band: str = "smr",
    cfg: ErspConfig = ERSP_CFG,
    subject: str = "",
    session: int = 0,
    channel_picks: Optional[List[str]] = None,
) -> ErspResult:
    """Compute ERSP and ITC for a set of reward-locked epochs.

    Parameters
    ----------
    epochs : mne.Epochs
        Reward-locked epochs (preprocessed, artifact-rejected).
    reward_band : str
        Key into :data:`config.BANDS` for this subject's trained frequency.
    cfg : ErspConfig
        Time-frequency and normalization parameters.
    subject, session : str, int
        For labelling.
    channel_picks : list of str, optional
        Restrict computation to these channels (default: all EEG).

    Returns
    -------
    ErspResult
    """
    if channel_picks is not None:
        epochs = epochs.copy().pick(channel_picks)

    ch_names = epochs.ch_names

    logger.info(
        "Computing ERSP: %s session %d, %d trials, %d channels, method=%s",
        subject, session, len(epochs), len(ch_names), cfg.method,
    )

    if cfg.method == "morlet":
        power, cplx, times, freqs = _morlet_tfr_single_trials(epochs, cfg)
    elif cfg.method == "stft":
        power, cplx, times, freqs = _stft_tfr_single_trials(epochs, cfg)
    else:
        raise ValueError(f"Unknown TFR method: {cfg.method}")

    ersp = single_trial_normalize(power, times, cfg)
    itc = compute_itc(cplx)

    scalars = extract_scalar_metrics(
        ersp, itc, times, freqs, list(ch_names), reward_band, cfg,
    )

    return ErspResult(
        subject=subject,
        session=session,
        freqs=freqs,
        times=times,
        ersp=ersp,
        itc=itc,
        channel_names=list(ch_names),
        n_trials=len(epochs),
        scalars=scalars,
    )


# ---------------------------------------------------------------------------
# HDF5 persistence
# ---------------------------------------------------------------------------


def save_ersp(result: ErspResult, path: Path) -> None:
    """Save an ErspResult to HDF5."""
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["subject"] = result.subject
        f.attrs["session"] = result.session
        f.attrs["n_trials"] = result.n_trials
        f.create_dataset("freqs", data=result.freqs)
        f.create_dataset("times", data=result.times)
        f.create_dataset("ersp", data=result.ersp)
        f.create_dataset("itc", data=result.itc)
        f.attrs["channel_names"] = result.channel_names
        for k, v in result.scalars.items():
            f.attrs[f"scalar/{k}"] = v
    logger.info("Saved ERSP: %s", path)


def load_ersp(path: Path) -> ErspResult:
    """Load an ErspResult from HDF5."""
    import h5py

    with h5py.File(path, "r") as f:
        scalars = {
            k.replace("scalar/", ""): float(f.attrs[k])
            for k in f.attrs if k.startswith("scalar/")
        }
        return ErspResult(
            subject=str(f.attrs["subject"]),
            session=int(f.attrs["session"]),
            freqs=f["freqs"][:],
            times=f["times"][:],
            ersp=f["ersp"][:],
            itc=f["itc"][:],
            channel_names=list(f.attrs["channel_names"]),
            n_trials=int(f.attrs["n_trials"]),
            scalars=scalars,
        )
