"""EEG preprocessing pipeline for BFB recordings (PIPELINE.md §3).

Stages (all wrapped as composable functions operating on MNE Raw):
  1. Set channel types (EEG + EOG for ch 65-66)
  2. Assign BioSemi-64 montage
  3. High-pass filter (0.1 Hz FIR, zero-phase) + 60 Hz notch
  4. Common-average re-reference excluding EOG
  5. Bad channel detection (RANSAC-like) + interpolation
  6. ICA (extended Infomax) + ICLabel auto-rejection
  7. Save cleaned raw as .fif

A *sensitivity branch* (0.16 Hz highpass, no ICA) for Attack 5 comparison
is available via :func:`preprocess_minimal`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import EOG_CHANNEL_INDICES, PreprocessConfig, PREPROCESS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PreprocessResult:
    """Metadata returned after preprocessing a single recording."""
    subject: str
    session: int
    bad_channels: List[str]
    n_ica_components: int
    rejected_ic_labels: List[str]
    output_path: Path


# ---------------------------------------------------------------------------
# Channel setup
# ---------------------------------------------------------------------------


_EXG_MISC = frozenset({"EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8"})


def set_channel_types(raw, eog_indices: Sequence[int] = EOG_CHANNEL_INDICES):
    """Mark all channels as EEG, then override EOG and misc channels.

    BioSemi EXG1/EXG2 are typically the mastoid references (handled
    elsewhere).  EXG3-8 are auxiliary inputs used for EMG or unused — they
    lack montage positions and must be set to ``misc`` so RANSAC and
    montage fitting do not choke on them.

    Parameters
    ----------
    raw : mne.io.Raw
        In-place modification.
    eog_indices : sequence of int
        0-based indices of the EOG channels (default: 64, 65 for VEOU/VEOL).
    """
    ch_types: Dict[str, str] = {}
    for i, name in enumerate(raw.ch_names):
        if i in eog_indices:
            ch_types[name] = "eog"
        elif name in _EXG_MISC:
            ch_types[name] = "misc"
        elif raw.get_channel_types([name])[0] in ("stim", "misc"):
            continue  # leave stim/misc as-is
        else:
            ch_types[name] = "eeg"
    raw.set_channel_types(ch_types)
    return raw


def set_montage(raw, montage_name: str = "biosemi64"):
    """Assign standard electrode positions."""
    import mne
    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage, on_missing="warn")
    return raw


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def apply_filters(
    raw,
    highpass_hz: float = 0.1,
    notch_hz: Optional[float] = 60.0,
):
    """Apply high-pass FIR filter and optional notch.

    No low-pass is applied before time-frequency analysis (preserve up to 40 Hz).
    """
    raw.filter(l_freq=highpass_hz, h_freq=None, method="fir",
               phase="zero-double", verbose="WARNING")
    if notch_hz is not None:
        raw.notch_filter(freqs=notch_hz, verbose="WARNING")
    return raw


# ---------------------------------------------------------------------------
# Re-reference
# ---------------------------------------------------------------------------


def rereference(raw, exclude_eog: bool = True):
    """Common-average re-reference, optionally excluding EOG channels."""
    import mne

    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False)
    ref_channels = [raw.ch_names[i] for i in eeg_picks]
    raw.set_eeg_reference(ref_channels=ref_channels, verbose="WARNING")
    return raw


# ---------------------------------------------------------------------------
# Bad channel detection + interpolation
# ---------------------------------------------------------------------------


def detect_and_interpolate_bad_channels(raw, method: str = "ransac"):
    """Detect and interpolate bad EEG channels.

    Parameters
    ----------
    method : str
        ``"ransac"`` — pyprep RANSAC detection (slow, thorough).
        ``"variance"`` — simple z-score variance outlier detection.
        ``"none"`` — skip detection; only interpolate pre-seeded bads.
    """
    import mne

    if method == "none":
        logger.info("Bad channel detection skipped (method='none')")
    elif method == "ransac":
        try:
            from pyprep.find_noisy_channels import NoisyChannels
            seeded = list(raw.info["bads"])
            nd = NoisyChannels(raw, do_detrend=False)
            nd.find_bad_by_ransac()
            merged = sorted(set(seeded) | set(nd.bad_by_ransac))
            raw.info["bads"] = merged
        except ImportError:
            logger.warning("pyprep not installed; falling back to variance method")
            method = "variance"

    if method == "variance":
        eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False)
        data = raw.get_data(picks=eeg_picks)
        var = np.var(data, axis=1)
        median_var = np.median(var)
        mad = np.median(np.abs(var - median_var))
        threshold = median_var + 5 * 1.4826 * mad
        bad_idx = np.where(var > threshold)[0]
        raw.info["bads"] = [raw.ch_names[eeg_picks[i]] for i in bad_idx]

    if raw.info["bads"]:
        logger.info("Bad channels: %s", raw.info["bads"])
        raw.interpolate_bads(reset_bads=True)
    else:
        logger.info("No bad channels detected")

    return raw


# ---------------------------------------------------------------------------
# ICA + ICLabel
# ---------------------------------------------------------------------------


def run_ica(
    raw,
    method: str = "infomax",
    n_components: float = 0.99,
    iclabel_threshold: float = 0.80,
    reject_classes: Sequence[str] = PREPROCESS.iclabel_reject_classes,
):
    """Run ICA and auto-reject artifact components via ICLabel.

    Returns
    -------
    ica : mne.preprocessing.ICA
    rejected : list of str
        Human-readable labels for each rejected component.
    """
    import mne
    from mne.preprocessing import ICA

    fit_params = {}
    if method == "infomax":
        fit_params["extended"] = True

    ica = ICA(
        n_components=n_components,
        method=method,
        fit_params=fit_params,
        random_state=42,
        verbose="WARNING",
    )
    try:
        ica.fit(raw, picks="eeg", verbose="WARNING")
    except RuntimeError as e:
        if "One PCA component" in str(e):
            logger.warning("ICA variance threshold too aggressive, retrying with n_components=15")
            ica = ICA(
                n_components=min(15, len(mne.pick_types(raw.info, eeg=True)) - 1),
                method=method,
                fit_params=fit_params,
                random_state=42,
                verbose="WARNING",
            )
            ica.fit(raw, picks="eeg", verbose="WARNING")
        else:
            raise
    logger.info("ICA fitted: %d components", ica.n_components_)

    rejected_labels: List[str] = []

    # ICLabel classification
    try:
        from mne_icalabel import label_components
    except ImportError:
        logger.warning("mne-icalabel not installed; skipping auto-rejection")
        label_components = None

    if label_components is not None:
        try:
            result = label_components(raw, ica, method="iclabel")
            ic_labels = result["labels"]
            ic_probs = result["y_pred_proba"]

            exclude_idx: List[int] = []
            for i, (lbl, prob) in enumerate(zip(ic_labels, ic_probs)):
                if lbl != "brain" and float(prob) >= iclabel_threshold:
                    exclude_idx.append(i)
                    rejected_labels.append(f"IC{i}:{lbl}({prob:.2f})")

            ica.exclude = exclude_idx
            logger.info("ICLabel rejected %d/%d components: %s",
                         len(exclude_idx), ica.n_components_, rejected_labels)
        except Exception:
            logger.warning("ICLabel classification failed", exc_info=True)

    return ica, rejected_labels


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def preprocess(
    raw,
    cfg: PreprocessConfig = PREPROCESS,
    output_path: Optional[Path] = None,
    subject: str = "",
    session: int = 0,
    known_bad_channels: Optional[Sequence[str]] = None,
) -> PreprocessResult:
    """Run the full preprocessing pipeline on a single BFB recording.

    Parameters
    ----------
    raw : mne.io.Raw
        Loaded raw BDF/EDF (will be modified in place).
    cfg : PreprocessConfig
        Pipeline parameters.
    output_path : Path, optional
        Where to save the cleaned ``.fif``.  Skipped if *None*.
    subject, session : str, int
        For logging and the result record.
    known_bad_channels : sequence of str, optional
        Channel names known to be bad from external metadata (e.g., the
        original study's bad-channel spreadsheet).  These are marked bad
        before RANSAC detection to seed the algorithm.

    Returns
    -------
    PreprocessResult
    """
    logger.info("Preprocessing %s session %d", subject, session)

    # 1. Channel types
    set_channel_types(raw)

    # 2. Montage
    set_montage(raw, cfg.montage_name)

    # 3. Filters
    apply_filters(raw, cfg.highpass_hz, cfg.notch_hz)

    # 4. Re-reference
    rereference(raw, exclude_eog=cfg.exclude_eog)

    # 5. Bad channels — seed with known bad channels before detection
    if known_bad_channels:
        existing = set(raw.info["bads"])
        available = set(raw.ch_names)
        to_add = [ch for ch in known_bad_channels if ch in available and ch not in existing]
        if to_add:
            raw.info["bads"].extend(to_add)
            logger.info("Seeded %d known bad channels: %s", len(to_add), to_add)
    detect_and_interpolate_bad_channels(raw, cfg.bad_channel_method)
    bad_channels = list(raw.info["bads"])

    # 6. ICA (unless skipped for sensitivity branch)
    rejected_labels: List[str] = []
    n_ica = 0
    if cfg.ica_method != "none":
        ica, rejected_labels = run_ica(
            raw,
            method=cfg.ica_method,
            n_components=cfg.ica_n_components,
            iclabel_threshold=cfg.iclabel_threshold,
            reject_classes=cfg.iclabel_reject_classes,
        )
        n_ica = int(ica.n_components_)
        raw = ica.apply(raw, verbose="WARNING")

    # 7. Save
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw.save(output_path, overwrite=True, verbose="WARNING")
        logger.info("Saved cleaned raw: %s", output_path)

    result = PreprocessResult(
        subject=subject,
        session=session,
        bad_channels=[str(ch) for ch in bad_channels],
        n_ica_components=n_ica,
        rejected_ic_labels=rejected_labels,
        output_path=output_path or Path(),
    )

    if output_path is not None:
        save_provenance(result, cfg)

    return result


def save_provenance(result: PreprocessResult, cfg: PreprocessConfig) -> None:
    """Write a JSON sidecar with preprocessing parameters and outcomes."""
    import datetime
    import json

    import mne

    prov = {
        "subject": result.subject,
        "session": result.session,
        "bad_channels": result.bad_channels,
        "n_ica_components": result.n_ica_components,
        "rejected_ics": result.rejected_ic_labels,
        "highpass_hz": cfg.highpass_hz,
        "notch_hz": cfg.notch_hz,
        "ica_method": cfg.ica_method,
        "iclabel_threshold": cfg.iclabel_threshold,
        "mne_version": mne.__version__,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    path = result.output_path.with_name(
        result.output_path.stem + "_provenance.json"
    )
    path.write_text(json.dumps(prov, indent=2) + "\n")
    logger.info("Provenance saved: %s", path)


def preprocess_minimal(
    raw,
    output_path: Optional[Path] = None,
    subject: str = "",
    session: int = 0,
) -> PreprocessResult:
    """Sensitivity branch: dissertation-era processing (0.16 Hz, no ICA)."""
    from .config import sensitivity_minimal_preprocess
    cfg = sensitivity_minimal_preprocess()
    return preprocess(raw, cfg=cfg, output_path=output_path,
                      subject=subject, session=session)
