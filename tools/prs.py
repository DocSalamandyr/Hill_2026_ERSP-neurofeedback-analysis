"""Post-reinforcement synchronization (PRS) analysis.

Computes per-subject / per-session parietal alpha (8–12 Hz) event-related
synchronization in a post-ERD window (800–1500 ms post-reward), following
the classical Sterman signature. Pre-specified in
``audit/PRS_ANALYSIS.md``; motivated by Arns pre-submission feedback.

The Morlet TFR and Grandchamp & Delorme (2011) single-trial dB normalization
are reused from :mod:`tools.ersp` so the PRS analysis is consistent with
the primary ERSP pipeline (same baseline, same normalization, same epochs).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import ERSP as ERSP_CFG, ErspConfig
from .ersp import _morlet_tfr_single_trials, single_trial_normalize

logger = logging.getLogger(__name__)


PRS_PRIMARY_CHANNELS: Tuple[str, ...] = ("Pz", "P3", "P4")
PRS_SECONDARY_CHANNELS: Tuple[str, ...] = ("POz", "PO3", "PO4")
PRS_CONTROL_CHANNELS: Tuple[str, ...] = ("Fz",)

PRS_CHANNELS: Tuple[str, ...] = (
    PRS_PRIMARY_CHANNELS + PRS_SECONDARY_CHANNELS + PRS_CONTROL_CHANNELS
)

PRS_ALPHA_BAND: Tuple[float, float] = (8.0, 12.0)
PRS_REWARD_BAND_SMR: Tuple[float, float] = (12.0, 15.0)
PRS_REWARD_BAND_BETA: Tuple[float, float] = (15.0, 18.0)

PRS_WINDOW_POST: Tuple[float, float] = (0.800, 1.500)   # primary
PRS_WINDOW_LATE_ERD: Tuple[float, float] = (0.500, 0.800)


@dataclass
class PrsResult:
    """Per subject × session PRS result.

    We persist a compact representation: scalars per (channel × band × window),
    plus the full time-frequency map restricted to the PRS channels (needed
    for the Pz cluster permutation at the group level) and the frequency/time
    axes. This is a few MB per subject-session rather than tens of MB.
    """

    subject: str
    session: int
    freqs: np.ndarray                 # (n_freqs,)
    times: np.ndarray                 # (n_times,)
    ersp: np.ndarray                  # (n_channels, n_freqs, n_times) dB
    channel_names: List[str]
    n_trials: int
    scalars: Dict[str, float]


def _time_mask(times: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
    return (times >= window[0]) & (times <= window[1])


def _freq_mask(freqs: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    return (freqs >= band[0]) & (freqs < band[1])


def _scalar(ersp: np.ndarray, times: np.ndarray, freqs: np.ndarray,
            ch_idx: int, band: Tuple[float, float],
            window: Tuple[float, float]) -> float:
    t_mask = _time_mask(times, window)
    f_mask = _freq_mask(freqs, band)
    if not t_mask.any() or not f_mask.any():
        return float("nan")
    return float(ersp[ch_idx, f_mask][:, t_mask].mean())


def extract_prs_scalars(
    ersp: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    channel_names: List[str],
) -> Dict[str, float]:
    """Extract the pre-specified PRS scalars (band × window × channel).

    Keys:
      prs_alpha_<ch>        : 8-12 Hz, 800-1500 ms, primary PRS scalar
      prs_alpha_late_<ch>   : 8-12 Hz, 500-800 ms, late-ERD window
      prs_smr_<ch>          : 12-15 Hz, 800-1500 ms, reward-band secondary
      prs_beta_<ch>         : 15-18 Hz, 800-1500 ms, reward-band secondary
      prs_alpha_primary_roi : mean over Pz + P3 + P4 at 8-12 Hz, 800-1500 ms
      prs_alpha_secondary_roi : mean over POz + PO3 + PO4 at 8-12 Hz, 800-1500 ms
    """
    metrics: Dict[str, float] = {}
    for ci, ch in enumerate(channel_names):
        metrics[f"prs_alpha_{ch}"] = _scalar(
            ersp, times, freqs, ci, PRS_ALPHA_BAND, PRS_WINDOW_POST,
        )
        metrics[f"prs_alpha_late_{ch}"] = _scalar(
            ersp, times, freqs, ci, PRS_ALPHA_BAND, PRS_WINDOW_LATE_ERD,
        )
        metrics[f"prs_smr_{ch}"] = _scalar(
            ersp, times, freqs, ci, PRS_REWARD_BAND_SMR, PRS_WINDOW_POST,
        )
        metrics[f"prs_beta_{ch}"] = _scalar(
            ersp, times, freqs, ci, PRS_REWARD_BAND_BETA, PRS_WINDOW_POST,
        )

    def _roi_mean(roi_channels: Tuple[str, ...]) -> float:
        idxs = [i for i, ch in enumerate(channel_names) if ch in roi_channels]
        if not idxs:
            return float("nan")
        vals = [metrics[f"prs_alpha_{channel_names[i]}"] for i in idxs]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    metrics["prs_alpha_primary_roi"] = _roi_mean(PRS_PRIMARY_CHANNELS)
    metrics["prs_alpha_secondary_roi"] = _roi_mean(PRS_SECONDARY_CHANNELS)

    return metrics


def compute_prs(
    epochs,
    cfg: ErspConfig = ERSP_CFG,
    subject: str = "",
    session: int = 0,
    channel_picks: Optional[List[str]] = None,
) -> PrsResult:
    """Compute PRS for a set of reward-locked epochs.

    Parameters
    ----------
    epochs : mne.Epochs
        Reward-locked epochs (already preprocessed + artifact-rejected,
        from the primary pipeline's ``stage_epochs``).
    cfg : ErspConfig
        Time-frequency and baseline parameters; same as the primary pipeline.
    subject, session : str, int
        For labelling.
    channel_picks : list of str, optional
        Default is the parietal + parieto-occipital + Fz control set
        defined in :data:`PRS_CHANNELS`. Any missing channels in the
        epochs are silently dropped.
    """
    picks = list(channel_picks) if channel_picks is not None else list(PRS_CHANNELS)
    present = [ch for ch in picks if ch in epochs.ch_names]
    missing = [ch for ch in picks if ch not in epochs.ch_names]
    if missing:
        logger.warning(
            "PRS %s s%d: %d channels missing from epochs and dropped: %s",
            subject, session, len(missing), missing,
        )
    if not present:
        raise ValueError(
            f"No PRS channels present in epochs for {subject} session {session}"
        )

    ep = epochs.copy().pick(present)

    logger.info(
        "Computing PRS: %s session %d, %d trials, channels=%s, method=%s",
        subject, session, len(ep), present, cfg.method,
    )

    power, _cplx, times, freqs = _morlet_tfr_single_trials(ep, cfg)
    ersp = single_trial_normalize(power, times, cfg)

    scalars = extract_prs_scalars(ersp, times, freqs, present)

    return PrsResult(
        subject=subject,
        session=session,
        freqs=freqs,
        times=times,
        ersp=ersp,
        channel_names=present,
        n_trials=len(ep),
        scalars=scalars,
    )


def save_prs(result: PrsResult, path: Path) -> None:
    """Save a PrsResult to HDF5."""
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["subject"] = result.subject
        f.attrs["session"] = result.session
        f.attrs["n_trials"] = result.n_trials
        f.create_dataset("freqs", data=result.freqs)
        f.create_dataset("times", data=result.times)
        f.create_dataset("ersp", data=result.ersp)
        f.attrs["channel_names"] = result.channel_names
        for k, v in result.scalars.items():
            f.attrs[f"scalar/{k}"] = v
    logger.info("Saved PRS: %s", path)


def load_prs(path: Path) -> PrsResult:
    """Load a PrsResult from HDF5."""
    import h5py

    with h5py.File(path, "r") as f:
        scalars = {
            k.replace("scalar/", ""): float(f.attrs[k])
            for k in f.attrs if k.startswith("scalar/")
        }
        return PrsResult(
            subject=str(f.attrs["subject"]),
            session=int(f.attrs["session"]),
            freqs=f["freqs"][:],
            times=f["times"][:],
            ersp=f["ersp"][:],
            channel_names=list(f.attrs["channel_names"]),
            n_trials=int(f.attrs["n_trials"]),
            scalars=scalars,
        )
