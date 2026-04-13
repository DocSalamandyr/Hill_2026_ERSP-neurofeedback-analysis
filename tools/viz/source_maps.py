"""Supplementary Figures S11-S12: Exploratory eLORETA source maps.

S11: Band-specific source power — beta-band (C3 Beta) vs SMR-band (C3 SMR)
S12: P2 evoked source maps — C3 SMR vs C3 Beta

Uses nilearn for matplotlib-based cortical surface rendering (no display
server required). All visualizations are illustrative grand averages;
inferential statistics are at the sensor level in the main text.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

import mne

from ..config import BANDS, ErpConfig, ERP as ERP_CFG
from ..source import (
    apply_eloreta,
    bandpass_evoked,
    compute_group_noise_cov,
    group_evoked,
    make_group_inverse,
)
from .style import GROUP_LABELS, apply_style, save_fig

logger = logging.getLogger(__name__)


def _gini(values: np.ndarray) -> float:
    """Gini coefficient: 0 = perfectly uniform, 1 = maximally focal."""
    v = np.abs(values).flatten()
    if v.sum() == 0:
        return 0.0
    v = np.sort(v)
    n = len(v)
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * v) / (n * np.sum(v))) - (n + 1) / n


def _stc_power_in_window(
    stc: mne.SourceEstimate,
    tmin: float,
    tmax: float,
) -> np.ndarray:
    """Mean absolute power in a time window, per vertex."""
    stc_crop = stc.copy().crop(tmin, tmax)
    return np.mean(stc_crop.data ** 2, axis=1)


def _get_fsaverage_surfaces(subjects_dir: str):
    """Load fsaverage inflated surfaces and sulc maps for nilearn plotting."""
    from pathlib import Path
    fs_dir = Path(subjects_dir) / "fsaverage"
    return {
        "lh_infl": str(fs_dir / "surf" / "lh.inflated"),
        "rh_infl": str(fs_dir / "surf" / "rh.inflated"),
        "lh_sulc": str(fs_dir / "surf" / "lh.sulc"),
        "rh_sulc": str(fs_dir / "surf" / "rh.sulc"),
    }


def _sparse_to_full(
    data: np.ndarray,
    vtx: np.ndarray,
    surf_path: str,
    smoothing: int = 3,
) -> np.ndarray:
    """Expand sparse source-space data to full surface mesh.

    Uses the surface mesh adjacency to propagate values from source
    vertices to their neighbors, averaging over `smoothing` steps.
    """
    import nibabel as nib
    from scipy.sparse import csr_matrix

    coords, faces = nib.freesurfer.read_geometry(surf_path)
    n_vertices = coords.shape[0]

    full = np.zeros(n_vertices, dtype=np.float64)
    full[vtx] = data

    if smoothing < 1:
        return full

    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2],
                           faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0],
                           faces[:, 0], faces[:, 1], faces[:, 2]])
    adj = csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n_vertices, n_vertices),
    )

    for _ in range(smoothing):
        neighbor_sum = adj.dot(full)
        neighbor_count = adj.dot((full != 0).astype(float))
        mask = neighbor_count > 0
        new_full = full.copy()
        update_mask = mask & (full == 0)
        new_full[update_mask] = neighbor_sum[update_mask] / neighbor_count[update_mask]
        has_val = full != 0
        has_neighbors = neighbor_count > 0
        both = has_val & has_neighbors
        new_full[both] = (full[both] + neighbor_sum[both]) / (1 + neighbor_count[both])
        full = new_full

    return full


def plot_erd_source_maps(
    epochs_by_group: Dict[str, Sequence[mne.Epochs]],
    fwd: mne.Forward,
    fig_dir: Path,
    subjects_dir: Optional[str] = None,
    depth_values: Sequence[float] = (0.8, 0.0),
) -> Optional[plt.Figure]:
    """Generate Figure S11: ERD source maps for C3 Beta vs C3 SMR.

    Computes eLORETA on band-filtered grand-average evoked responses,
    renders side-by-side brain views.

    Parameters
    ----------
    epochs_by_group : dict
        Keys are group names ("c3_beta", "c3_smr"), values are lists
        of Epochs objects.
    fwd : mne.Forward
        Forward solution (fsaverage, fixed orientation).
    fig_dir : Path
        Output directory for figure files.
    depth_values : sequence of float
        Depth weighting values to test. First value is used for the
        final figure; comparison is logged for sensitivity check.
    """
    if subjects_dir is None:
        subjects_dir = str(
            Path(mne.datasets.fetch_fsaverage(verbose=False)).parent
        )

    apply_style()

    band_configs = {
        "c3_beta": ("beta", BANDS["beta"]),
        "c3_smr": ("smr", BANDS["smr"]),
    }

    erd_window = (0.2, 0.8)
    baseline_window = (-0.1, 0.0)

    results = {}  # group -> {depth -> stc}

    for group_key in ("c3_beta", "c3_smr"):
        if group_key not in epochs_by_group:
            logger.warning("No epochs for %s, skipping", group_key)
            return None

        epochs_list = epochs_by_group[group_key]
        band_name, (fmin, fmax) = band_configs[group_key]

        noise_cov = compute_group_noise_cov(epochs_list)
        grand_avg = group_evoked(epochs_list)
        filtered = bandpass_evoked(grand_avg, fmin, fmax)

        results[group_key] = {}
        for depth in depth_values:
            inv = make_group_inverse(grand_avg.info, fwd, noise_cov, depth=depth)
            stc = apply_eloreta(filtered, inv)

            stc_active = stc.copy().crop(*erd_window)
            source_power = np.mean(np.abs(stc_active.data), axis=1)

            stc_power = stc.copy()
            stc_power.data = source_power[:, np.newaxis]
            stc_power.tmin = 0.0
            stc_power.tstep = 1.0
            results[group_key][depth] = stc_power

            peak_vertex = np.argmax(source_power)
            logger.info(
                "ERD source map: group=%s, depth=%.1f, "
                "peak source amplitude=%.2e at vertex %d, "
                "Gini=%.3f (1=focal, 0=uniform)",
                group_key, depth, source_power[peak_vertex], peak_vertex,
                _gini(source_power),
            )

    primary_depth = depth_values[0]

    if len(depth_values) > 1:
        _log_depth_sensitivity(results, depth_values)

    fig = _compose_erd_figure(
        results, primary_depth, subjects_dir, fig_dir,
    )
    return fig


def _log_depth_sensitivity(
    results: Dict[str, Dict[float, mne.SourceEstimate]],
    depth_values: Sequence[float],
) -> None:
    """Log spatial correlation between depth=0.8 and depth=0.0 maps."""
    for group_key in results:
        maps = [results[group_key][d].data[:, 0] for d in depth_values]
        if len(maps) >= 2:
            corr = np.corrcoef(maps[0], maps[1])[0, 1]
            logger.info(
                "Depth sensitivity [%s]: r(depth=%.1f vs depth=%.1f) = %.3f",
                group_key, depth_values[0], depth_values[1], corr,
            )

    d0, d1 = depth_values[0], depth_values[1]
    for group_key in results:
        peak_d0 = np.argmax(results[group_key][d0].data[:, 0])
        peak_d1 = np.argmax(results[group_key][d1].data[:, 0])
        logger.info(
            "Peak source vertex [%s]: depth=%.1f -> %d, depth=%.1f -> %d",
            group_key, d0, peak_d0, d1, peak_d1,
        )


def _compose_erd_figure(
    results: Dict[str, Dict[float, mne.SourceEstimate]],
    depth: float,
    subjects_dir: str,
    fig_dir: Path,
) -> plt.Figure:
    """Render cortical surface maps for ERD source estimates using nilearn."""
    from nilearn.plotting import plot_surf_stat_map

    surfaces = _get_fsaverage_surfaces(subjects_dir)
    groups = ["c3_beta", "c3_smr"]
    band_labels = {"c3_beta": "Beta (15-18 Hz)", "c3_smr": "SMR (12-15 Hz)"}
    hemis = [("lh", "left"), ("rh", "right")]

    fig, axes = plt.subplots(
        2, 4, figsize=(16, 8),
        subplot_kw={"projection": "3d"},
    )

    for row, group_key in enumerate(groups):
        stc = results[group_key][depth]
        raw_data = stc.data[:, 0].copy()
        dmax = raw_data.max()
        normed = raw_data / dmax if dmax > 0 else raw_data

        for col_offset, (hemi_key, hemi_label) in enumerate(hemis):
            if hemi_key == "lh":
                data = normed[:stc.vertices[0].size]
                vtx = stc.vertices[0]
            else:
                data = normed[stc.vertices[0].size:]
                vtx = stc.vertices[1]

            surf_mesh = surfaces[f"{hemi_key}_infl"]
            bg_map = surfaces[f"{hemi_key}_sulc"]

            full_data = _sparse_to_full(data, vtx, surf_mesh, smoothing=5)
            nonzero = full_data[full_data > 0]
            thresh = np.percentile(nonzero, 65) if len(nonzero) > 0 else 0.5

            for view_idx, view in enumerate(["lateral", "medial"]):
                ax = axes[row, col_offset * 2 + view_idx]
                plot_surf_stat_map(
                    surf_mesh, full_data,
                    bg_map=bg_map,
                    hemi=hemi_label,
                    view=view,
                    cmap="YlOrRd",
                    vmax=1.0,
                    threshold=thresh,
                    axes=ax,
                    colorbar=False,
                )

        axes[row, 0].set_title(
            f"{GROUP_LABELS[group_key]} — {band_labels[group_key]}",
            fontsize=11, fontweight="bold", loc="left",
        )

    fig.suptitle(
        "Exploratory eLORETA Source Maps — Reward-Locked Band Power\n"
        "L Lateral  |  L Medial  |  R Lateral  |  R Medial",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, fig_dir / "source_erd_maps")
    logger.info("Figure S11 saved to %s", fig_dir / "source_erd_maps.png")
    return fig


def plot_erp_source_maps(
    epochs_by_group: Dict[str, Sequence[mne.Epochs]],
    fwd: mne.Forward,
    fig_dir: Path,
    subjects_dir: Optional[str] = None,
    erp_cfg: ErpConfig = ERP_CFG,
) -> Optional[plt.Figure]:
    """Generate Figure S12: P2 evoked source maps for C3 SMR vs C3 Beta.

    Applies eLORETA to the grand-average ERP (0.5-30 Hz bandpass),
    renders source activity at the P2 peak latency.
    """
    if subjects_dir is None:
        subjects_dir = str(
            Path(mne.datasets.fetch_fsaverage(verbose=False)).parent
        )

    apply_style()

    p2_tmin, p2_tmax = erp_cfg.p2_window
    results = {}

    for group_key in ("c3_smr", "c3_beta"):
        if group_key not in epochs_by_group:
            logger.warning("No epochs for %s, skipping S12", group_key)
            return None

        epochs_list = epochs_by_group[group_key]

        noise_cov = compute_group_noise_cov(epochs_list)
        grand_avg = group_evoked(epochs_list)
        grand_avg.apply_baseline((-0.2, 0.0))

        inv = make_group_inverse(grand_avg.info, fwd, noise_cov, depth=0.8)
        stc = apply_eloreta(grand_avg, inv)

        stc_p2 = stc.copy().crop(p2_tmin, p2_tmax)
        stc_mean = stc_p2.mean()
        results[group_key] = stc_mean

        peak_val = stc_mean.data.max()
        peak_vertex = stc_mean.data.argmax()
        logger.info(
            "P2 source: group=%s, peak=%.2e at vertex %d",
            group_key, peak_val, peak_vertex,
        )

    fig = _compose_erp_figure(results, subjects_dir, fig_dir)
    return fig


def _compose_erp_figure(
    results: Dict[str, mne.SourceEstimate],
    subjects_dir: str,
    fig_dir: Path,
) -> plt.Figure:
    """Render cortical surface maps for P2 source estimates using nilearn."""
    from nilearn.plotting import plot_surf_stat_map

    surfaces = _get_fsaverage_surfaces(subjects_dir)
    groups = ["c3_smr", "c3_beta"]
    hemis = [("lh", "left"), ("rh", "right")]

    fig, axes = plt.subplots(
        2, 4, figsize=(16, 8),
        subplot_kw={"projection": "3d"},
    )

    global_max = max(results[g].data.max() for g in groups)

    for row, group_key in enumerate(groups):
        stc = results[group_key]
        for col_offset, (hemi_key, hemi_label) in enumerate(hemis):
            if hemi_key == "lh":
                data = stc.data[:stc.vertices[0].size, 0]
                vtx = stc.vertices[0]
            else:
                data = stc.data[stc.vertices[0].size:, 0]
                vtx = stc.vertices[1]

            surf_mesh = surfaces[f"{hemi_key}_infl"]
            bg_map = surfaces[f"{hemi_key}_sulc"]

            full_data = _sparse_to_full(data, vtx, surf_mesh, smoothing=5)

            for view_idx, view in enumerate(["lateral", "medial"]):
                ax = axes[row, col_offset * 2 + view_idx]
                plot_surf_stat_map(
                    surf_mesh, full_data,
                    bg_map=bg_map,
                    hemi=hemi_label,
                    view=view,
                    cmap="hot",
                    vmax=global_max,
                    threshold=global_max * 0.15,
                    axes=ax,
                    colorbar=False,
                )

        axes[row, 0].set_title(
            f"{GROUP_LABELS[group_key]} — P2 (140–260 ms)",
            fontsize=11, fontweight="bold", loc="left",
        )

    fig.suptitle(
        "Exploratory eLORETA Source Maps — P2 Component (140–260 ms)\n"
        "L Lateral  |  L Medial  |  R Lateral  |  R Medial",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, fig_dir / "source_erp_p2_maps")
    logger.info("Figure S12 saved to %s", fig_dir / "source_erp_p2_maps.png")
    return fig

