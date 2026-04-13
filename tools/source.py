"""Source localization utilities for exploratory eLORETA visualization.

Builds a forward model from fsaverage (BioSemi-64), computes noise
covariance from pre-stimulus baseline, and applies eLORETA to evoked data.
All source estimates are illustrative (grand-average, no group-level stats).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import mne
import numpy as np
from mne.minimum_norm import apply_inverse, make_inverse_operator

logger = logging.getLogger(__name__)

FSAVERAGE_DIR = Path(
    mne.datasets.fetch_fsaverage(verbose=False)
)  # e.g. ~/mne_data/MNE-fsaverage-data/fsaverage

ICO_ORDER = 5
CONDUCTIVITIES = (0.3, 0.006, 0.3)  # scalp, skull, brain (S/m)
SNR = 3.0
LAMBDA2 = 1.0 / (SNR ** 2)


def build_forward_model(
    info: mne.Info,
    subjects_dir: Optional[Path] = None,
) -> Tuple[mne.Forward, mne.SourceSpaces]:
    """Build a BEM forward model on fsaverage for EEG.

    Returns the forward solution and source spaces. The forward solution
    uses surface orientation (free, constrained to fixed via loose=0.0 in
    the inverse operator, as required by MNE for depth-weighted eLORETA).
    """
    if subjects_dir is None:
        subjects_dir = FSAVERAGE_DIR.parent

    logger.info("Setting up ico-%d source space on fsaverage ...", ICO_ORDER)
    src = mne.setup_source_space(
        "fsaverage", spacing=f"ico{ICO_ORDER}",
        subjects_dir=str(subjects_dir), add_dist=False,
    )
    n_src = sum(s["nuse"] for s in src)
    logger.info("Source space: %d active vertices", n_src)

    logger.info("Building 3-layer BEM ...")
    model = mne.make_bem_model(
        "fsaverage", conductivity=CONDUCTIVITIES,
        subjects_dir=str(subjects_dir),
    )
    bem = mne.make_bem_solution(model)

    logger.info("Computing forward solution ...")
    fwd = mne.make_forward_solution(
        info, trans="fsaverage", src=src, bem=bem,
        eeg=True, meg=False, verbose=False,
    )
    fwd = mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=False, verbose=False,
    )
    logger.info("Forward: %d sources, %d channels",
                fwd["nsource"], fwd["nchan"])
    return fwd, src


def compute_group_noise_cov(
    epochs_list: Sequence[mne.Epochs],
) -> mne.Covariance:
    """Compute shrunk noise covariance from pre-stimulus baseline.

    Concatenates all epochs and estimates covariance from tmin to 0.
    """
    all_epochs = mne.concatenate_epochs(list(epochs_list))
    all_epochs.apply_baseline((None, 0.0))
    logger.info("Computing noise covariance from %d epochs ...", len(all_epochs))
    noise_cov = mne.compute_covariance(
        all_epochs, tmax=0.0,
        method=["shrunk", "empirical"],
        rank=None, verbose=False,
    )
    return noise_cov


def make_group_inverse(
    info: mne.Info,
    fwd: mne.Forward,
    noise_cov: mne.Covariance,
    depth: float = 0.8,
) -> mne.minimum_norm.InverseOperator:
    """Create an eLORETA-ready inverse operator.

    Parameters
    ----------
    depth : float
        Depth weighting (0.0 = none, 0.8 = standard MNE default).
        Compare both to check sensitivity of source patterns.
    """
    inv = make_inverse_operator(
        info, fwd, noise_cov,
        loose=0.0, depth=depth, verbose=False,
    )
    logger.info("Inverse operator ready (depth=%.1f)", depth)
    return inv


def apply_eloreta(
    evoked: mne.Evoked,
    inv: mne.minimum_norm.InverseOperator,
    lambda2: float = LAMBDA2,
) -> mne.SourceEstimate:
    """Apply eLORETA to an evoked response."""
    stc = apply_inverse(
        evoked, inv, lambda2,
        method="eLORETA", pick_ori=None, verbose=False,
    )
    return stc


def load_group_epochs(
    study,
    group: str,
    sessions: Sequence[int] = (1, 3, 5),
    picks: str = "eeg",
) -> List[mne.Epochs]:
    """Load reward-locked epochs for all subjects in a group.

    Returns a list of Epochs objects (one per subject-session).
    Adds an average-reference projector required by MNE inverse solvers.
    """
    subjects = study.by_group(group)
    epoch_list = []
    for subj in subjects:
        for sess in sessions:
            fif_path = study.epochs_fif(subj.subject_id, sess)
            if not fif_path.is_file():
                logger.warning("Missing: %s", fif_path)
                continue
            ep = mne.read_epochs(str(fif_path), preload=True, verbose=False)
            ep.pick(picks)
            ep.set_eeg_reference(projection=True, verbose=False)
            ep.apply_proj()
            epoch_list.append(ep)
    logger.info("Loaded %d epoch files for group=%s", len(epoch_list), group)
    return epoch_list


def group_evoked(epochs_list: Sequence[mne.Epochs]) -> mne.Evoked:
    """Grand-average evoked from a list of Epochs."""
    evokeds = [ep.average() for ep in epochs_list]
    grand = mne.grand_average(evokeds)
    return grand


def bandpass_evoked(
    evoked: mne.Evoked,
    l_freq: float,
    h_freq: float,
) -> mne.Evoked:
    """Band-filter an evoked response (returns a copy)."""
    filtered = evoked.copy()
    n_samples = filtered.get_data().shape[1]
    sfreq = filtered.info["sfreq"]
    max_fir_len = min(int(sfreq * 0.5), n_samples - 1)
    fir_design = "firwin" if max_fir_len > 50 else "firwin2"
    try:
        filtered.filter(
            l_freq, h_freq, method="fir", phase="zero-double",
            fir_design=fir_design, verbose=False,
        )
    except ValueError:
        filtered.filter(
            l_freq, h_freq, method="iir", verbose=False,
        )
    return filtered
