"""BDF import and BioSemi event parsing for the ERSP mechanism paper.

Handles:
- BDF and EDF file loading via MNE (harmonized — both produce identical Raw objects)
- Recording-type name aliases (e.g. EOPORE → EOPRE for subject 129)
- BioSemi status channel bit-mask parsing for reward events (0x0100)
- Event count validation (~600-700 expected per BFB session)
- Metadata extraction (sfreq, n_channels, duration)

EDF/BDF harmonization: MNE reads both formats into the same ``mne.io.Raw``
representation.  The only difference is bit depth (BDF 24-bit ≈ 0.03 µV/LSB,
EDF 16-bit ≈ 8 µV/LSB).  For resting-state Welch PSD the 16-bit quantization
noise is negligible.  See ``audit/pacdel-bdf-audit-full.md §10`` for details.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import (
    CORRUPT_BFBS,
    DATA_ROOT,
    EXPECTED_REWARD_COUNT,
    REC_TYPE_ALIASES,
    REWARD_EVENT_CODE,
)

logger = logging.getLogger(__name__)

MIN_VALID_FILE_BYTES = 100_000  # files below this are likely header-only

# ---------------------------------------------------------------------------
# Dataclasses for structured returns
# ---------------------------------------------------------------------------


@dataclass
class BdfMetadata:
    """Summary information extracted from a BDF/EDF header."""

    path: Path
    subject: str
    recording_type: str          # BFB, LANT, CPT, EOPRE, …
    session: int
    sfreq: float
    n_channels: int
    duration_sec: float
    is_edf: bool


@dataclass
class RewardEvents:
    """Reward events extracted from a BioSemi recording."""

    sample_indices: np.ndarray   # shape (n_events,)
    times_sec: np.ndarray        # shape (n_events,)
    count: int
    is_valid: bool               # True if count within expected range
    message: str                 # human-readable validation note


# ---------------------------------------------------------------------------
# File-path helpers
# ---------------------------------------------------------------------------


def bdf_path(subject: str, rec_type: str, session: int,
             data_root: Path = DATA_ROOT) -> Path:
    """Resolve ``{subject}_{rec_type}_{session}.bdf`` under *data_root*."""
    return data_root / subject / f"{subject}_{rec_type}_{session}.bdf"


def find_recording(subject: str, rec_type: str, session: int,
                   data_root: Path = DATA_ROOT) -> Optional[Path]:
    """Return path to a BDF (preferred) or EDF fallback, or *None*.

    Tries the canonical name first, then known aliases (e.g. EOPORE for
    subject 129's EOPRE files), then EDF variants of both.

    Returns *None* (with a warning) for files in ``CORRUPT_BFBS`` or files
    smaller than ``MIN_VALID_FILE_BYTES``.
    """
    if (subject, session) in CORRUPT_BFBS:
        logger.warning(
            "Corrupt BFB (server-verified unrecoverable): %s session %d",
            subject, session,
        )
        return None

    candidates: List[str] = [rec_type]
    for alias, canonical in REC_TYPE_ALIASES.items():
        if canonical == rec_type and alias not in candidates:
            candidates.append(alias)
    if rec_type in REC_TYPE_ALIASES:
        canonical = REC_TYPE_ALIASES[rec_type]
        if canonical not in candidates:
            candidates.insert(0, canonical)

    for rt in candidates:
        p = bdf_path(subject, rt, session, data_root)
        if p.is_file():
            if p.stat().st_size < MIN_VALID_FILE_BYTES:
                logger.warning(
                    "File suspiciously small (%d bytes), skipping: %s",
                    p.stat().st_size, p,
                )
                return None
            if rt != rec_type:
                logger.info("Resolved alias %s → %s for %s session %d",
                            rec_type, rt, subject, session)
            return p

    for rt in candidates:
        edf = bdf_path(subject, rt, session, data_root).with_suffix(".edf")
        if edf.is_file():
            if edf.stat().st_size < MIN_VALID_FILE_BYTES:
                logger.warning(
                    "EDF suspiciously small (%d bytes), skipping: %s",
                    edf.stat().st_size, edf,
                )
                return None
            logger.warning(
                "BDF missing, falling back to EDF (16-bit, harmonized): %s",
                edf,
            )
            return edf

    return None


def parse_filename(path: Path) -> Tuple[str, str, int]:
    """Extract ``(subject, rec_type, session)`` from BDF/EDF naming.

    Normalizes known aliases (e.g. EOPORE → EOPRE) to canonical names.
    """
    stem = path.stem                       # e.g. "101_BFB_1"
    parts = stem.rsplit("_", maxsplit=2)
    if len(parts) != 3:
        raise ValueError(f"Cannot parse filename: {path.name}")
    subject, rec_type, sess_str = parts
    rec_type = REC_TYPE_ALIASES.get(rec_type, rec_type)
    return subject, rec_type, int(sess_str)


# ---------------------------------------------------------------------------
# BDF / EDF loading
# ---------------------------------------------------------------------------


def load_raw(path: Path, preload: bool = True, verbose: str = "WARNING"):
    """Load a BDF or EDF file via MNE and return the Raw object.

    Parameters
    ----------
    path : Path
        Full path to a ``.bdf`` or ``.edf`` file.
    preload : bool
        Whether to preload data into memory.
    verbose : str
        MNE verbosity level.

    Returns
    -------
    mne.io.Raw
    """
    import mne

    suffix = path.suffix.lower()
    if suffix == ".bdf":
        raw = mne.io.read_raw_bdf(path, preload=preload, verbose=verbose)
    elif suffix == ".edf":
        raw = mne.io.read_raw_edf(path, preload=preload, verbose=verbose)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    return raw


def extract_metadata(path: Path) -> BdfMetadata:
    """Read header-level metadata without loading full data."""
    import mne

    raw = load_raw(path, preload=False, verbose="ERROR")
    subject, rec_type, session = parse_filename(path)
    return BdfMetadata(
        path=path,
        subject=subject,
        recording_type=rec_type,
        session=session,
        sfreq=raw.info["sfreq"],
        n_channels=len(raw.ch_names),
        duration_sec=raw.times[-1],
        is_edf=path.suffix.lower() == ".edf",
    )


# ---------------------------------------------------------------------------
# Event extraction
# ---------------------------------------------------------------------------


def extract_reward_events(
    raw,
    event_code: int = REWARD_EVENT_CODE,
    stim_channel: Optional[str] = None,
) -> RewardEvents:
    """Parse reward events from a BioSemi status channel.

    MNE's ``find_events`` reads the status/trigger channel and returns an
    (n_events, 3) array.  We match events where the reward bit (0x0100) is
    set, which captures both code 256 (0x0100) and 511 (0x01FF) -- the
    latter appears in many Session 6 recordings where extra trigger bits
    are set alongside the reward bit.

    Parameters
    ----------
    raw : mne.io.Raw
        Loaded Raw object (BDF or EDF).
    event_code : int
        Trigger bit-pattern for reward onset (default 0x0100).  Events are
        matched via ``(code & event_code) == event_code`` so that any
        event with the reward bit set is captured.
    stim_channel : str or None
        Explicit stim channel name.  If *None*, MNE auto-detects
        (typically ``STI 014`` or the BioSemi Status channel).

    Returns
    -------
    RewardEvents
    """
    import mne

    kwargs: Dict = {"min_duration": 0}
    if stim_channel is not None:
        kwargs["stim_channel"] = stim_channel

    all_events = mne.find_events(raw, verbose="WARNING", **kwargs)

    # Bit-mask match on lower 16 bits only (strip BioSemi system flags in bits 16+)
    codes_lo16 = all_events[:, 2] & 0xFFFF
    mask = (codes_lo16 & event_code) == event_code
    reward_events = all_events[mask]

    matched_codes = set(int(c) for c in reward_events[:, 2]) if len(reward_events) > 0 else set()
    if len(matched_codes) > 1:
        logger.info(
            "Reward events matched multiple codes via bit-mask 0x%04X: %s",
            event_code, {f"0x{c:04X}({c})": int((reward_events[:, 2] == c).sum()) for c in matched_codes},
        )

    count = len(reward_events)
    lo, hi = EXPECTED_REWARD_COUNT
    if lo <= count <= hi:
        valid = True
        msg = f"{count} reward events (within expected {lo}-{hi})"
    elif count == 0:
        valid = False
        msg = "No reward events found — check stim channel or event code"
    else:
        valid = True  # usable but noteworthy
        msg = f"{count} reward events (outside typical {lo}-{hi}; review)"

    logger.info("extract_reward_events: %s", msg)

    sfreq = raw.info["sfreq"]
    samples = reward_events[:, 0] if count > 0 else np.array([], dtype=int)
    times = samples / sfreq if count > 0 else np.array([], dtype=float)

    return RewardEvents(
        sample_indices=samples,
        times_sec=times,
        count=count,
        is_valid=valid,
        message=msg,
    )


def events_to_mne(reward: RewardEvents, event_id: int = 1) -> np.ndarray:
    """Convert *RewardEvents* to the MNE (n, 3) events array format.

    Column layout: ``[sample, 0, event_id]``.
    """
    n = reward.count
    if n == 0:
        return np.empty((0, 3), dtype=int)
    events = np.zeros((n, 3), dtype=int)
    events[:, 0] = reward.sample_indices
    events[:, 2] = event_id
    return events


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def validate_subject_bfb(
    subject: str,
    sessions: Sequence[int] = (1, 3, 5, 6),
    data_root: Path = DATA_ROOT,
) -> List[BdfMetadata]:
    """Check that all expected BFB files exist for *subject*."""
    results: List[BdfMetadata] = []
    for sess in sessions:
        p = find_recording(subject, "BFB", sess, data_root)
        if p is None:
            logger.error("Missing BFB session %d for subject %s", sess, subject)
            continue
        meta = extract_metadata(p)
        results.append(meta)
        logger.info(
            "%s: %.1fs, %d ch, sfreq=%.0f%s",
            p.name, meta.duration_sec, meta.n_channels, meta.sfreq,
            " [EDF]" if meta.is_edf else "",
        )
    return results
