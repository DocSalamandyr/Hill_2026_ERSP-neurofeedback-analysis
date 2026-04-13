"""Grand-average assembly: per-subject HDF5 → group-level DataFrames/arrays.

Bridges the gap between per-subject pipeline outputs and the group-level
statistics / visualisation modules.  All functions accept a :class:`Study`
and return structures ready for ``stats.py`` or ``tools/viz/*.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import BFB_SESSIONS, BANDS, RESTING_SESSIONS
from .erp import ErpResult, load_erp
from .ersp import ErspResult, load_ersp
from .resting import PsdResult, load_psd
from .study import Study, Subject

logger = logging.getLogger(__name__)


def recenter_baseline(
    ersp: np.ndarray,
    times: np.ndarray,
    baseline: Tuple[float, float] = (-0.1, 0.0),
) -> np.ndarray:
    """Subtract per-frequency baseline mean to re-center ERSP at 0 dB.

    Single-trial dB normalization (Grandchamp & Delorme, 2011) shifts the
    trial-averaged baseline negative due to Jensen's inequality:
    E[log(X)] < log(E[X]).  This correction removes that offset for
    visualization while preserving the single-trial normalization for
    statistics.

    Works on 2-D (n_freqs, n_times) or 3-D (n_subjects, n_freqs, n_times).
    """
    bl_mask = (times >= baseline[0]) & (times <= baseline[1])
    if not bl_mask.any():
        return ersp

    if ersp.ndim == 2:
        offset = ersp[:, bl_mask].mean(axis=-1, keepdims=True)
        return ersp - offset
    elif ersp.ndim == 3:
        offset = ersp[:, :, bl_mask].mean(axis=-1, keepdims=True)
        return ersp - offset
    return ersp


def auto_ersp_clim(ersp: np.ndarray, percentile: float = 98.0) -> float:
    """Symmetric color limit from data percentiles (after re-centering)."""
    vabs = max(abs(np.percentile(ersp, 100 - percentile)),
               abs(np.percentile(ersp, percentile)))
    return round(vabs, 2) if vabs > 0.1 else 0.5


def assemble_ersp_scalars(
    study: Study,
    sessions: Sequence[int] = BFB_SESSIONS,
    scalar_prefix: str = "primary_erd_",
    channels: Sequence[str] = ("C3", "C4"),
) -> pd.DataFrame:
    """Collect scalar ERD/ERS metrics across subjects into long-format DataFrame.

    Returns columns: subject, group, session, channel, metric, value.
    """
    rows: list[dict] = []
    for subj in study.included():
        for sess in sessions:
            h5 = study.ersp_h5(subj.subject_id, sess)
            if not h5.is_file():
                logger.debug("Missing ERSP: %s session %d", subj.subject_id, sess)
                continue
            result = load_ersp(h5)
            for ch in channels:
                for key, val in result.scalars.items():
                    if ch in key:
                        rows.append({
                            "subject": subj.subject_id,
                            "group": subj.group,
                            "session": sess,
                            "channel": ch,
                            "metric": key,
                            "value": val,
                        })

    df = pd.DataFrame(rows)
    logger.info("Assembled ERSP scalars: %d rows from %d subjects",
                len(df), df["subject"].nunique() if len(df) else 0)
    return df


def assemble_ersp_tfr(
    study: Study,
    sessions: Sequence[int],
    channel: str,
    group: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Stack per-subject ERSP TFR arrays for one channel and group.

    Returns
    -------
    data : ndarray, shape (n_subjects, n_freqs, n_times)
    freqs : ndarray
    times : ndarray
    subject_ids : list of str
    """
    subjects = study.by_group(group) if group else study.included()
    arrays: list[np.ndarray] = []
    ids: list[str] = []
    ref_freqs: Optional[np.ndarray] = None
    ref_times: Optional[np.ndarray] = None

    for subj in subjects:
        sess_arrays: list[np.ndarray] = []
        for sess in sessions:
            h5 = study.ersp_h5(subj.subject_id, sess)
            if not h5.is_file():
                continue
            result = load_ersp(h5)
            if channel not in result.channel_names:
                continue
            ci = result.channel_names.index(channel)
            sess_arrays.append(result.ersp[ci])
            if ref_freqs is None:
                ref_freqs = result.freqs
                ref_times = result.times

        if sess_arrays:
            arrays.append(np.mean(sess_arrays, axis=0))
            ids.append(subj.subject_id)

    if not arrays:
        logger.warning("No ERSP TFR data assembled for channel=%s group=%s",
                        channel, group)
        return np.empty((0, 0, 0)), np.array([]), np.array([]), []

    data = np.stack(arrays, axis=0)
    logger.info("Assembled ERSP TFR: %s, shape %s", channel, data.shape)
    return data, ref_freqs, ref_times, ids


def assemble_erp_grand_average(
    study: Study,
    sessions: Sequence[int] = BFB_SESSIONS,
    channels: Sequence[str] = ("C3", "C4", "Pz"),
    group: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Average ERP waveforms across subjects within a group.

    Returns
    -------
    grand_avg : ndarray, shape (n_channels, n_times)
    times : ndarray
    channel_names : list of str
    subject_ids : list of str
    """
    subjects = study.by_group(group) if group else study.included()
    all_evoked: list[np.ndarray] = []
    ids: list[str] = []
    ref_times: Optional[np.ndarray] = None
    ref_ch: Optional[list[str]] = None

    for subj in subjects:
        sess_evoked: list[np.ndarray] = []
        for sess in sessions:
            h5 = study.erp_h5(subj.subject_id, sess)
            if not h5.is_file():
                continue
            result = load_erp(h5)
            ch_idx = [result.channel_names.index(c)
                       for c in channels if c in result.channel_names]
            if not ch_idx:
                continue
            sess_evoked.append(result.evoked[ch_idx])
            if ref_times is None:
                ref_times = result.times
                ref_ch = [result.channel_names[i] for i in ch_idx]

        if sess_evoked:
            all_evoked.append(np.mean(sess_evoked, axis=0))
            ids.append(subj.subject_id)

    if not all_evoked:
        logger.warning("No ERP data assembled for group=%s", group)
        return np.empty((0, 0)), np.array([]), [], []

    grand_avg = np.mean(all_evoked, axis=0)
    logger.info("Assembled ERP grand average: %d subjects, shape %s",
                len(ids), grand_avg.shape)
    return grand_avg, ref_times, ref_ch or [], ids


def assemble_erp_by_session(
    study: Study,
    sessions: Sequence[int] = (1, 5),
    channels: Sequence[str] = ("C3", "C4", "Pz"),
    group: Optional[str] = None,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, List[str], List[str]]:
    """Grand-average ERP per session for one group.

    Unlike :func:`assemble_erp_grand_average` which collapses across sessions,
    this returns a separate grand-average waveform for each session.

    Returns
    -------
    by_session : dict[int, ndarray], shape (n_channels, n_times) per session
    times : ndarray
    channel_names : list of str
    subject_ids : list of str  (subjects contributing to at least one session)
    """
    subjects = study.by_group(group) if group else study.included()
    per_session: Dict[int, list] = {s: [] for s in sessions}
    all_ids: set = set()
    ref_times: Optional[np.ndarray] = None
    ref_ch: Optional[list[str]] = None

    for subj in subjects:
        for sess in sessions:
            h5 = study.erp_h5(subj.subject_id, sess)
            if not h5.is_file():
                continue
            result = load_erp(h5)
            ch_idx = [result.channel_names.index(c)
                      for c in channels if c in result.channel_names]
            if not ch_idx:
                continue
            per_session[sess].append(result.evoked[ch_idx])
            all_ids.add(subj.subject_id)
            if ref_times is None:
                ref_times = result.times
                ref_ch = [result.channel_names[i] for i in ch_idx]

    by_session: Dict[int, np.ndarray] = {}
    for sess, arrays in per_session.items():
        if arrays:
            by_session[sess] = np.mean(arrays, axis=0)
            logger.info("ERP session %d: %d subjects, shape %s",
                        sess, len(arrays), by_session[sess].shape)

    if not by_session:
        logger.warning("No ERP data assembled by session for group=%s", group)
        return {}, np.array([]), [], []

    return by_session, ref_times, ref_ch or [], sorted(all_ids)


def assemble_resting_change(
    study: Study,
    pre_session: int = 1,
    post_session: int = 6,
    resting_type: str = "ECPRE",
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    channels: Sequence[str] = ("C3", "C4"),
) -> pd.DataFrame:
    """Compute session-to-session PSD change per subject.

    Returns long-format DataFrame with: subject, group, channel, band,
    pre_power, post_power, delta.
    """
    if bands is None:
        bands = dict(BANDS)

    rows: list[dict] = []
    for subj in study.included():
        pre_path = study.resting_h5(subj.subject_id, resting_type, pre_session)
        post_path = study.resting_h5(subj.subject_id, resting_type, post_session)
        if not pre_path.is_file() or not post_path.is_file():
            logger.debug(
                "Missing resting for %s: pre=%s post=%s",
                subj.subject_id, pre_path.exists(), post_path.exists(),
            )
            continue

        pre = load_psd(pre_path)
        post = load_psd(post_path)

        for band_name in bands:
            for ch in channels:
                pre_val = pre.band_power.get(band_name, {}).get(ch)
                post_val = post.band_power.get(band_name, {}).get(ch)
                if pre_val is None or post_val is None:
                    continue
                rows.append({
                    "subject": subj.subject_id,
                    "group": subj.group,
                    "channel": ch,
                    "band": band_name,
                    "pre_power": pre_val,
                    "post_power": post_val,
                    "delta": post_val - pre_val,
                })

    df = pd.DataFrame(rows)
    logger.info("Assembled resting change: %d rows from %d subjects",
                len(df), df["subject"].nunique() if len(df) else 0)
    return df


def assemble_resting_trajectory(
    study: Study,
    sessions: Sequence[int] = BFB_SESSIONS,
    conditions: Sequence[str] = ("ECPRE", "ECPOST", "EOPRE", "EOPOST"),
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    channels: Sequence[str] = ("C3", "C4"),
) -> pd.DataFrame:
    """Assemble resting-state band power across all sessions and conditions.

    Returns long-format DataFrame with: subject, group, session, condition,
    channel, band, power.  Enables both within-session (POST-PRE) and
    across-session (PRE trajectory) analyses.
    """
    from .resting import load_psd

    if bands is None:
        bands = dict(BANDS)

    rows: list[dict] = []
    for subj in study.included():
        for sess in sessions:
            for cond in conditions:
                path = study.resting_h5(subj.subject_id, cond, sess)
                if not path.is_file():
                    continue
                psd = load_psd(path)
                for band_name in bands:
                    for ch in channels:
                        val = psd.band_power.get(band_name, {}).get(ch)
                        if val is not None:
                            rows.append({
                                "subject": subj.subject_id,
                                "group": subj.group,
                                "session": sess,
                                "condition": cond,
                                "channel": ch,
                                "band": band_name,
                                "power": val,
                            })

    df = pd.DataFrame(rows)
    logger.info("Assembled resting trajectory: %d rows from %d subjects",
                len(df), df["subject"].nunique() if len(df) else 0)
    return df


def assemble_frequency_profile(
    study: Study,
    sessions: Sequence[int] = BFB_SESSIONS,
    channel: str = "C3",
    group: Optional[str] = None,
    erd_window: Tuple[float, float] = (0.2, 0.8),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average ERSP across the ERD time window for each frequency bin.

    For each subject, loads ERSP h5 files, extracts the specified channel,
    and averages across the ERD time window to produce a 1-D frequency profile.
    Session-averaged profiles are then aggregated for group mean +/- SE.

    Returns
    -------
    mean_profile : ndarray (n_freqs,)
    se_profile : ndarray (n_freqs,)
    freqs : ndarray (n_freqs,)
    """
    subjects = study.by_group(group) if group else study.included()
    profiles: list[np.ndarray] = []
    ref_freqs: Optional[np.ndarray] = None

    for subj in subjects:
        sess_profiles: list[np.ndarray] = []
        for sess in sessions:
            h5 = study.ersp_h5(subj.subject_id, sess)
            if not h5.is_file():
                continue
            result = load_ersp(h5)
            if channel not in result.channel_names:
                continue
            ci = result.channel_names.index(channel)
            tfr = recenter_baseline(result.ersp[ci], result.times)
            t_mask = (result.times >= erd_window[0]) & (result.times <= erd_window[1])
            profile = tfr[:, t_mask].mean(axis=1)
            sess_profiles.append(profile)
            if ref_freqs is None:
                ref_freqs = result.freqs

        if sess_profiles:
            profiles.append(np.mean(sess_profiles, axis=0))

    if not profiles or ref_freqs is None:
        logger.warning("No frequency profile data for channel=%s group=%s",
                        channel, group)
        return np.array([]), np.array([]), np.array([])

    stacked = np.stack(profiles, axis=0)
    mean_profile = stacked.mean(axis=0)
    se_profile = stacked.std(axis=0, ddof=1) / np.sqrt(len(profiles))

    logger.info("Frequency profile: %s, %d subjects, %d freqs",
                channel, len(profiles), len(ref_freqs))
    return mean_profile, se_profile, ref_freqs


def assemble_ersp_tfr_by_session(
    study: Study,
    sessions: Sequence[int] = BFB_SESSIONS,
    channel: str = "C3",
    group: Optional[str] = None,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray, List[str]]:
    """Stack per-subject ERSP TFR arrays, one stack per session.

    Unlike :func:`assemble_ersp_tfr` which averages sessions together,
    this keeps sessions separate for the groups x sessions heatmap grid.

    Returns
    -------
    by_session : dict[session_int] -> ndarray (n_subjects, n_freqs, n_times)
    freqs : ndarray
    times : ndarray
    subject_ids : list of str  (union across sessions)
    """
    subjects = study.by_group(group) if group else study.included()
    per_session: Dict[int, list] = {s: [] for s in sessions}
    per_session_ids: Dict[int, list] = {s: [] for s in sessions}
    ref_freqs: Optional[np.ndarray] = None
    ref_times: Optional[np.ndarray] = None

    for subj in subjects:
        for sess in sessions:
            h5 = study.ersp_h5(subj.subject_id, sess)
            if not h5.is_file():
                continue
            result = load_ersp(h5)
            if channel not in result.channel_names:
                continue
            ci = result.channel_names.index(channel)
            per_session[sess].append(result.ersp[ci])
            per_session_ids[sess].append(subj.subject_id)
            if ref_freqs is None:
                ref_freqs = result.freqs
                ref_times = result.times

    by_session: Dict[int, np.ndarray] = {}
    all_ids: set = set()
    for sess in sessions:
        if per_session[sess]:
            by_session[sess] = np.stack(per_session[sess], axis=0)
            all_ids.update(per_session_ids[sess])
            logger.info("ERSP TFR session %d: %d subjects at %s, shape %s",
                        sess, len(per_session[sess]), channel,
                        by_session[sess].shape)

    if not by_session:
        logger.warning("No per-session TFR for channel=%s group=%s",
                        channel, group)
        return {}, np.array([]), np.array([]), []

    return by_session, ref_freqs, ref_times, sorted(all_ids)


def assemble_topo_erd(
    study: Study,
    session: int,
    group: Optional[str] = None,
    reward_band: Tuple[float, float] = (12.0, 15.0),
    erd_window: Tuple[float, float] = (0.2, 0.8),
) -> Tuple[np.ndarray, Optional[object], List[str]]:
    """Compute reward-band ERD across all channels from epoch files.

    Uses bandpass filtering + Hilbert transform (fast, memory-efficient)
    restricted to the reward band, with single-trial baseline normalization.
    This bypasses the per-channel ERSP h5 files which only store C3/C4.

    Returns
    -------
    erd_mean : ndarray (n_channels,)
        Group-mean ERD (dB) per channel.
    info : mne.Info or None
        Channel layout for topographic plotting.
    subject_ids : list of str
    """
    import mne
    from scipy.signal import hilbert
    from .config import ERSP as ERSP_CFG

    subjects = study.by_group(group) if group else study.included()
    all_erd: list[np.ndarray] = []
    ids: list[str] = []
    ref_info = None

    for subj in subjects:
        epo_path = study.epochs_fif(subj.subject_id, session)
        if not epo_path.is_file():
            continue

        epochs = mne.read_epochs(epo_path, preload=True, verbose="WARNING")
        if len(epochs) == 0:
            logger.debug("Empty epochs for %s session %d — skipping topo",
                         subj.subject_id, session)
            continue
        epochs = epochs.copy().pick("eeg")

        if ref_info is None:
            ref_info = epochs.info.copy()

        epochs_filt = epochs.copy().filter(
            reward_band[0], reward_band[1],
            method="iir", verbose="WARNING",
        )
        data = epochs_filt.get_data(copy=True)
        times = epochs.times

        analytic = hilbert(data, axis=-1)
        power = np.abs(analytic) ** 2

        bl_mask = (times >= ERSP_CFG.baseline_tmin) & (times <= ERSP_CFG.baseline_tmax)
        bl_mean = power[:, :, bl_mask].mean(axis=-1, keepdims=True)
        bl_mean = np.maximum(bl_mean, np.finfo(float).eps)
        power = np.maximum(power, np.finfo(float).eps)
        ersp_db = 10.0 * np.log10(power / bl_mean)

        ersp_mean = ersp_db.mean(axis=0)
        erd_mask = (times >= erd_window[0]) & (times <= erd_window[1])
        erd = ersp_mean[:, erd_mask].mean(axis=1)

        all_erd.append(erd)
        ids.append(subj.subject_id)

    if not all_erd:
        logger.warning("No topo ERD data for session=%d group=%s",
                        session, group)
        return np.array([]), None, []

    erd_stacked = np.stack(all_erd, axis=0)
    erd_mean = erd_stacked.mean(axis=0)

    logger.info("Topo ERD session %d: %d subjects, %d channels",
                session, len(ids), len(erd_mean))
    return erd_mean, ref_info, ids
