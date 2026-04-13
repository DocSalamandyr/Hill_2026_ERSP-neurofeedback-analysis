"""Reward-locked epoch extraction with early/late splits (PIPELINE.md §4).

Provides:
- Epoch creation from preprocessed Raw + reward events
- Peak-to-peak artifact rejection (200 µV for ICA mode)
- Statistical artifact rejection matching EEGLAB pop_autorej (minimal mode)
- Early / late session splits (first/last ~10 min)
- Sliding-window trial bins for learning-curve analysis
- Per-subject trial-count and rejection-rate logging
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sp_stats

from .config import ERSP, ErspConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class EpochResult:
    """Metadata from a single epoch-extraction run."""
    subject: str
    session: int
    total_events: int
    n_good: int
    n_rejected: int
    rejection_rate: float
    duration_sec: float          # recording duration used


# ---------------------------------------------------------------------------
# Statistical epoch rejection (EEGLAB pop_autorej equivalent)
# ---------------------------------------------------------------------------


def _joint_probability(data: np.ndarray) -> np.ndarray:
    """Compute per-epoch joint log-probability across channels.

    Parameters
    ----------
    data : ndarray, shape (n_epochs, n_channels, n_times)

    Returns
    -------
    jp : ndarray, shape (n_epochs,)
        Sum of per-channel log-probabilities for each epoch.
    """
    n_epochs, n_channels, _ = data.shape
    jp = np.zeros(n_epochs)
    for ch in range(n_channels):
        ch_data = data[:, ch, :]
        ch_mean = ch_data.mean(axis=1)
        mu = ch_mean.mean()
        sigma = ch_mean.std(ddof=1)
        if sigma < 1e-12:
            continue
        jp += np.abs(ch_mean - mu) / sigma
    return jp


def autorej_epochs(
    epochs,
    threshold_uv: float = 1000.0,
    startprob: float = 5.0,
    maxrej_pct: float = 5.0,
    kurtosis_thresh: float = 6.0,
) -> np.ndarray:
    """Statistical epoch rejection replicating EEGLAB pop_autorej defaults.

    Three-stage algorithm:
    1. Reject epochs exceeding ``threshold_uv`` absolute amplitude.
    2. Iterative joint-probability rejection starting at ``startprob`` SD,
       rejecting at most ``maxrej_pct`` percent per pass, with adaptive
       threshold refinement (up to 8 iterations).
    3. Kurtosis-based rejection at ``kurtosis_thresh`` SD.

    Parameters
    ----------
    epochs : mne.Epochs
        Loaded epochs (preload=True).
    threshold_uv : float
        Absolute amplitude threshold in microvolts for step 1.
    startprob : float
        Starting z-score threshold (in SD) for joint probability.
    maxrej_pct : float
        Maximum percentage of remaining epochs to reject per iteration.
    kurtosis_thresh : float
        Z-score threshold for per-epoch kurtosis rejection.

    Returns
    -------
    bad_indices : ndarray of int
        Indices (into the original epochs) of epochs to drop.
    """
    data = epochs.get_data(copy=True) * 1e6  # V -> µV
    n_epochs = data.shape[0]
    all_indices = np.arange(n_epochs)
    bad = np.zeros(n_epochs, dtype=bool)

    # --- Step 1: absolute amplitude threshold ---
    peak_max = data.max(axis=(1, 2))
    peak_min = data.min(axis=(1, 2))
    amp_bad = (peak_max > threshold_uv) | (peak_min < -threshold_uv)
    n_amp = amp_bad.sum()
    if n_amp > 0:
        logger.info("autorej step 1: %d epochs exceed ±%.0f µV", n_amp, threshold_uv)
    bad |= amp_bad

    # --- Step 2: iterative joint probability ---
    prob_thresh = startprob
    remaining = ~bad
    max_outer_iter = 8
    outer_count = 0

    while outer_count < max_outer_iter:
        if remaining.sum() < 3:
            break
        sub_data = data[remaining]
        jp = _joint_probability(sub_data)
        jp_mean = jp.mean()
        jp_std = jp.std(ddof=1)
        if jp_std < 1e-12:
            break

        z_jp = (jp - jp_mean) / jp_std
        marked = z_jp > prob_thresh
        n_marked = marked.sum()

        if n_marked == 0:
            if prob_thresh > startprob:
                prob_thresh -= 0.5
                outer_count += 1
                continue
            else:
                break

        if (n_marked / remaining.sum()) > (maxrej_pct / 100.0):
            prob_thresh += 0.5
            outer_count += 1
            continue

        remaining_idx = all_indices[remaining]
        bad[remaining_idx[marked]] = True
        remaining = ~bad
        outer_count += 1

    n_jp = bad.sum() - n_amp
    if n_jp > 0:
        logger.info("autorej step 2: %d epochs rejected by joint probability (thresh=%.1f SD)", n_jp, prob_thresh)

    # --- Step 3: kurtosis ---
    remaining = ~bad
    if remaining.sum() >= 3:
        sub_data = data[remaining]
        n_sub = sub_data.shape[0]
        epoch_kurt = np.zeros(n_sub)
        for i in range(n_sub):
            epoch_kurt[i] = sp_stats.kurtosis(sub_data[i].ravel(), fisher=True)

        kurt_mean = epoch_kurt.mean()
        kurt_std = epoch_kurt.std(ddof=1)
        if kurt_std > 1e-12:
            z_kurt = (epoch_kurt - kurt_mean) / kurt_std
            kurt_bad = z_kurt > kurtosis_thresh
            if kurt_bad.any():
                remaining_idx = all_indices[remaining]
                bad[remaining_idx[kurt_bad]] = True
                logger.info(
                    "autorej step 3: %d epochs rejected by kurtosis (>%.1f SD)",
                    kurt_bad.sum(), kurtosis_thresh,
                )

    total_rej = bad.sum()
    logger.info(
        "autorej total: %d / %d epochs rejected (%.1f%%)",
        total_rej, n_epochs, 100.0 * total_rej / n_epochs if n_epochs else 0,
    )
    return all_indices[bad]


# ---------------------------------------------------------------------------
# Core epoch creation
# ---------------------------------------------------------------------------


def create_reward_epochs(
    raw,
    events: np.ndarray,
    cfg: ErspConfig = ERSP,
    event_id: int = 1,
    picks: str = "eeg",
):
    """Create MNE Epochs locked to reward events.

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed continuous data.
    events : ndarray, shape (n, 3)
        MNE-format events array (from ``io.events_to_mne``).
    cfg : ErspConfig
        Epoch window and rejection parameters.
    event_id : int
        Event code in the *events* array.
    picks : str
        Channel type to include.

    Returns
    -------
    mne.Epochs
    """
    import mne

    method = getattr(cfg, "epoch_reject_method", "peak_to_peak")

    if method == "peak_to_peak":
        reject = {"eeg": cfg.reject_peak_to_peak_uv * 1e-6}  # µV → V
    else:
        reject = None  # statistical rejection applied after loading

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=cfg.tmin,
        tmax=cfg.tmax,
        baseline=None,           # baseline applied separately per analysis
        reject=reject,
        preload=True,
        picks=picks,
        verbose="WARNING",
    )

    if method == "autorej_statistical" and len(epochs) > 0:
        bad_idx = autorej_epochs(
            epochs,
            threshold_uv=cfg.reject_peak_to_peak_uv,
            startprob=cfg.autorej_startprob,
            maxrej_pct=cfg.autorej_maxrej_pct,
            kurtosis_thresh=cfg.autorej_kurtosis_thresh,
        )
        if len(bad_idx) > 0:
            epochs.drop(bad_idx, reason="autorej_statistical", verbose="WARNING")

    return epochs


def epoch_metadata(
    epochs, subject: str = "", session: int = 0,
) -> EpochResult:
    """Extract summary statistics from an MNE Epochs object."""
    drop_log = epochs.drop_log
    n_total = len(drop_log)
    n_good = len(epochs)
    n_rejected = n_total - n_good
    rate = n_rejected / n_total if n_total > 0 else 0.0
    dur = epochs.times[-1] - epochs.times[0]

    logger.info(
        "%s session %d: %d/%d epochs retained (%.1f%% rejected)",
        subject, session, n_good, n_total, rate * 100,
    )
    return EpochResult(
        subject=subject,
        session=session,
        total_events=n_total,
        n_good=n_good,
        n_rejected=n_rejected,
        rejection_rate=rate,
        duration_sec=dur,
    )


# ---------------------------------------------------------------------------
# Early / late splitting
# ---------------------------------------------------------------------------


def split_early_late(
    epochs,
    early_minutes: float = ERSP.early_late_minutes,
) -> Tuple:
    """Split epochs into *early* (first N minutes) and *late* (last N minutes).

    Uses the event onset times (relative to recording start) to determine
    which trials fall in the first and last time blocks.

    Returns
    -------
    early_epochs, late_epochs : mne.Epochs
    """
    if len(epochs) == 0:
        return epochs[:0], epochs[:0]
    event_times = epochs.events[:, 0] / epochs.info["sfreq"]
    t_start = event_times[0]
    t_end = event_times[-1]
    cutoff_sec = early_minutes * 60.0

    early_mask = (event_times - t_start) < cutoff_sec
    late_mask = (t_end - event_times) < cutoff_sec

    # Overlapping trials belong to whichever block they started in
    overlap = early_mask & late_mask
    if overlap.any():
        late_mask[overlap] = False

    early_idx = np.where(early_mask)[0]
    late_idx = np.where(late_mask)[0]

    early_epochs = epochs[early_idx]
    late_epochs = epochs[late_idx]

    logger.info(
        "Early/late split: %d / %d trials (cutoff %.0f min)",
        len(early_epochs), len(late_epochs), early_minutes,
    )
    return early_epochs, late_epochs


# ---------------------------------------------------------------------------
# Thirds splitting
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------


def sliding_window_indices(
    n_epochs: int,
    window_size: int = ERSP.sliding_window_trials,
    step: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Return ``(start, stop)`` index pairs for overlapping trial windows.

    Default step is half the window size.
    """
    if step is None:
        step = max(1, window_size // 2)
    windows = []
    start = 0
    while start + window_size <= n_epochs:
        windows.append((start, start + window_size))
        start += step
    if not windows:
        windows.append((0, n_epochs))
    return windows


def compute_sliding_metric(
    epochs,
    metric_fn,
    window_size: int = ERSP.sliding_window_trials,
    step: Optional[int] = None,
) -> List[Dict]:
    """Apply *metric_fn* to sliding windows of epochs.

    Parameters
    ----------
    epochs : mne.Epochs
        Full set of reward-locked epochs.
    metric_fn : callable
        ``metric_fn(sub_epochs) -> dict`` returning scalar metrics.
    window_size : int
        Number of trials per window.
    step : int, optional
        Step size (default: half window).

    Returns
    -------
    list of dict
        One entry per window with ``window_start``, ``window_end``, and
        whatever keys *metric_fn* returns.
    """
    windows = sliding_window_indices(len(epochs), window_size, step)
    results = []
    for start, stop in windows:
        sub = epochs[start:stop]
        metrics = metric_fn(sub)
        metrics["window_start"] = start
        metrics["window_end"] = stop
        results.append(metrics)
    return results


# ---------------------------------------------------------------------------
# Convenience: full extraction pipeline for one recording
# ---------------------------------------------------------------------------


def extract_and_split(
    raw,
    events: np.ndarray,
    cfg: ErspConfig = ERSP,
    subject: str = "",
    session: int = 0,
    save_path: Optional[Path] = None,
) -> Tuple:
    """Create epochs, log metadata, split early/late, optionally save.

    Returns
    -------
    epochs, early_epochs, late_epochs, meta : Epochs, Epochs, Epochs, EpochResult
    """
    epochs = create_reward_epochs(raw, events, cfg=cfg)
    meta = epoch_metadata(epochs, subject=subject, session=session)

    if meta.n_good < cfg.min_clean_trials:
        logger.warning(
            "%s session %d: only %d clean trials (minimum %d)",
            subject, session, meta.n_good, cfg.min_clean_trials,
        )

    early, late = split_early_late(epochs, cfg.early_late_minutes)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        epochs.save(save_path, overwrite=True, verbose="WARNING")
        logger.info("Saved epochs: %s", save_path)

    return epochs, early, late, meta
