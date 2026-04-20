#!/usr/bin/env python3
"""Master analysis pipeline: BDF -> preprocessing -> ERSP/ITC/ERP -> figures.

Implements the stages described in ERSP/PIPELINE.md and ERSP/PAPER.md,
wiring together all modules in ERSP/tools/.

Usage examples:
  python pipeline.py --stage validate   --subject 101
  python pipeline.py --stage preprocess --subject 101 --sessions 1,3,5,6
  python pipeline.py --stage epochs     --subject 101
  python pipeline.py --stage ersp       --subject 101
  python pipeline.py --stage erp        --subject 101
  python pipeline.py --stage prs        --subject 101
  python pipeline.py --stage resting    --subject 101
  python pipeline.py --stage stats      --study-json study.json
  python pipeline.py --stage prs_stats  --study-json study.json
  python pipeline.py --stage figures    --study-json study.json
  python pipeline.py --stage source     --study-json study.json
  python pipeline.py --stage pdf
  python pipeline.py --stage all        --subject 101

  # Preprocessing mode (default: minimal = dissertation-era, no ICA):
  python pipeline.py --preprocess-mode minimal --stage all --all-subjects --study-json data/study.json
  python pipeline.py --preprocess-mode ica     --stage all --all-subjects --study-json data/study.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

# Ensure the ERSP package is importable
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent                      # ERSP/
REPO_ROOT = PROJECT_ROOT.parent                        # research/
sys.path.insert(0, str(PROJECT_ROOT.parent))           # so "ERSP.tools" resolves

# Use absolute imports via the tools package
sys.path.insert(0, str(PROJECT_ROOT))
from tools.config import (
    ANALYSES_ROOT as _ANALYSES_ROOT_BASE,
    BFB_SESSIONS, DATA_ROOT,
    DERIVATIVES_ROOT as _DERIVATIVES_ROOT_BASE,
    ERSP as _ERSP_DEFAULT, ERP as ERP_CFG, PREPROCESS, RESTING as RESTING_CFG,
    RESTING_SESSIONS, RESTING_TYPES,
    ErspConfig, PreprocessConfig,
    sensitivity_minimal_preprocess, sensitivity_minimal_ersp,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-20s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Mode-aware output paths (set by main() based on --preprocess-mode)
# ---------------------------------------------------------------------------

ANALYSES_ROOT: Path = _ANALYSES_ROOT_BASE
DERIVATIVES_ROOT: Path = _DERIVATIVES_ROOT_BASE
PREPROCESS_CFG: PreprocessConfig = PREPROCESS
ERSP_CFG: ErspConfig = _ERSP_DEFAULT
PREPROCESS_MODE: str = "ica"


def _set_preprocess_mode(mode: str) -> None:
    """Configure output paths, preprocessing config, and ERSP config for the given mode."""
    global ANALYSES_ROOT, DERIVATIVES_ROOT, PREPROCESS_CFG, ERSP_CFG, PREPROCESS_MODE
    PREPROCESS_MODE = mode
    ANALYSES_ROOT = _ANALYSES_ROOT_BASE / mode
    DERIVATIVES_ROOT = _DERIVATIVES_ROOT_BASE / mode
    if mode == "minimal":
        PREPROCESS_CFG = sensitivity_minimal_preprocess()
        ERSP_CFG = sensitivity_minimal_ersp()
    else:
        PREPROCESS_CFG = PREPROCESS
        ERSP_CFG = _ERSP_DEFAULT
    logger.info(
        "Preprocess mode: %s | Analyses: %s | Derivatives: %s | "
        "Epoch reject: %s (%.0f µV)",
        mode, ANALYSES_ROOT, DERIVATIVES_ROOT,
        ERSP_CFG.epoch_reject_method, ERSP_CFG.reject_peak_to_peak_uv,
    )


# ── validate ───────────────────────────────────────────────────────────────


def stage_validate(subject: str, sessions: Sequence[int]) -> int:
    """Check BDF presence, event counts, sampling rate."""
    from tools.config import CORRUPT_BFBS, TRUNCATED_BFBS
    from tools.io import extract_reward_events, load_raw, validate_subject_bfb

    for sess in sessions:
        if (subject, sess) in CORRUPT_BFBS:
            logger.warning(
                "CORRUPT (server-verified): %s session %d — skipping",
                subject, sess,
            )
    if subject in TRUNCATED_BFBS:
        for sess in TRUNCATED_BFBS[subject]:
            logger.warning(
                "TRUNCATED (~1/3 normal size): %s session %d — "
                "will check trial count downstream", subject, sess,
            )

    metas = validate_subject_bfb(subject, sessions)
    if not metas:
        logger.error("No BFB files found for subject %s", subject)
        return 1

    for meta in metas:
        raw = load_raw(meta.path, preload=False)
        reward = extract_reward_events(raw)
        logger.info(
            "%s: %.1fs, %d ch, sfreq=%.0f | %s",
            meta.path.name, meta.duration_sec, meta.n_channels,
            meta.sfreq, reward.message,
        )
    return 0


# ── preprocess ─────────────────────────────────────────────────────────────


def stage_preprocess(subject: str, sessions: Sequence[int]) -> int:
    """Filter, re-reference, bad channels, (optionally) ICA, save cleaned raw + event sidecar."""
    import json
    import numpy as np
    from tools.io import (
        extract_reward_events, events_to_mne, find_recording, load_raw,
    )
    from tools.preprocess import preprocess
    from tools.config import PROJECT_ROOT

    known_bads: list[str] = []
    study_path = PROJECT_ROOT / "data" / "study.json"
    if study_path.is_file():
        for rec in json.loads(study_path.read_text()):
            if rec["subject_id"] == subject:
                known_bads = rec.get("known_bad_channels", [])
                break

    for sess in sessions:
        path = find_recording(subject, "BFB", sess)
        if path is None:
            logger.error("Missing BFB_%d for %s", sess, subject)
            continue

        raw = load_raw(path)

        reward = extract_reward_events(raw)
        events = events_to_mne(reward)
        events_path = (
            ANALYSES_ROOT / subject / "preprocessed"
            / f"{subject}_BFB_{sess}_events.npy"
        )
        events_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(events_path, events)
        logger.info("Saved %d events (%s mode) to %s", len(events), PREPROCESS_MODE, events_path)

        out_path = ANALYSES_ROOT / subject / "preprocessed" / f"{subject}_BFB_{sess}_clean-raw.fif"
        result = preprocess(raw, cfg=PREPROCESS_CFG, output_path=out_path,
                            subject=subject, session=sess,
                            known_bad_channels=known_bads)
        logger.info(
            "Preprocessed %s session %d [%s]: %d bad ch, %d ICA rejected",
            subject, sess, PREPROCESS_MODE,
            len(result.bad_channels), len(result.rejected_ic_labels),
        )
    return 0


# ── epochs ─────────────────────────────────────────────────────────────────


def stage_epochs(subject: str, sessions: Sequence[int]) -> int:
    """Extract reward-locked epochs from preprocessed data."""
    import mne
    import numpy as np
    from tools.config import TRUNCATED_BFBS
    from tools.epochs import extract_and_split

    for sess in sessions:
        clean_path = ANALYSES_ROOT / subject / "preprocessed" / f"{subject}_BFB_{sess}_clean-raw.fif"
        if not clean_path.is_file():
            logger.error("Preprocessed file missing: %s", clean_path)
            continue

        events_path = (
            ANALYSES_ROOT / subject / "preprocessed"
            / f"{subject}_BFB_{sess}_events.npy"
        )
        if not events_path.is_file():
            logger.error("Events sidecar missing: %s", events_path)
            continue

        raw = mne.io.read_raw_fif(clean_path, preload=True, verbose="WARNING")
        events = np.load(events_path)

        if len(events) == 0:
            logger.warning("No events in %s session %d — skipping", subject, sess)
            continue

        save_path = ANALYSES_ROOT / subject / "epochs" / f"{subject}_BFB_{sess}_reward-epo.fif"
        try:
            epochs, early, late, meta = extract_and_split(
                raw, events, cfg=ERSP_CFG, subject=subject, session=sess,
                save_path=save_path,
            )
        except (ValueError, IndexError) as e:
            if "No matching events" in str(e) or "index" in str(e).lower():
                logger.warning("No valid epochs for %s session %d — skipping: %s", subject, sess, e)
                continue
            raise

        truncated_sessions = TRUNCATED_BFBS.get(subject, ())
        if sess in truncated_sessions and meta.n_good < ERSP_CFG.min_clean_trials:
            logger.warning(
                "TRUNCATED file %s session %d: only %d good trials "
                "(< %d minimum) — results may be unreliable",
                subject, sess, meta.n_good, ERSP_CFG.min_clean_trials,
            )

        logger.info(
            "Epochs %s session %d: %d good (%d early, %d late), %.1f%% rejected",
            subject, sess, meta.n_good, len(early), len(late),
            meta.rejection_rate * 100,
        )
    return 0


# ── ersp ───────────────────────────────────────────────────────────────────


def stage_ersp(subject: str, sessions: Sequence[int],
               reward_band: str = "smr") -> int:
    """Compute ERSP/ITC from saved epochs."""
    import mne
    from tools.ersp import compute_ersp, save_ersp
    from tools.config import PRIMARY_CHANNELS

    for sess in sessions:
        epo_path = ANALYSES_ROOT / subject / "epochs" / f"{subject}_BFB_{sess}_reward-epo.fif"
        if not epo_path.is_file():
            logger.error("Epochs file missing: %s", epo_path)
            continue

        epochs = mne.read_epochs(epo_path, preload=True, verbose="WARNING")
        if len(epochs) == 0:
            logger.warning("Empty epochs for %s session %d — skipping ERSP", subject, sess)
            continue
        result = compute_ersp(
            epochs, reward_band=reward_band, cfg=ERSP_CFG,
            subject=subject, session=sess,
            channel_picks=list(PRIMARY_CHANNELS),
        )
        out_path = DERIVATIVES_ROOT / subject / "ersp" / f"{subject}_BFB_{sess}_ersp.h5"
        save_ersp(result, out_path)
        logger.info(
            "ERSP %s session %d: %d trials, primary_erd_C3=%.3f dB",
            subject, sess, result.n_trials,
            result.scalars.get("primary_erd_C3", float("nan")),
        )
    return 0


# ── prs ────────────────────────────────────────────────────────────────────


def stage_prs(subject: str, sessions: Sequence[int]) -> int:
    """Compute post-reinforcement synchronization scalars from saved epochs.

    Pre-specified in ``audit/PRS_ANALYSIS.md``. Reuses the existing
    reward-locked epochs; does not re-preprocess or re-epoch.
    """
    import mne
    from tools.prs import PRS_CHANNELS, compute_prs, save_prs

    for sess in sessions:
        epo_path = ANALYSES_ROOT / subject / "epochs" / f"{subject}_BFB_{sess}_reward-epo.fif"
        if not epo_path.is_file():
            logger.error("Epochs file missing: %s", epo_path)
            continue

        epochs = mne.read_epochs(epo_path, preload=True, verbose="WARNING")
        if len(epochs) == 0:
            logger.warning("Empty epochs for %s session %d — skipping PRS", subject, sess)
            continue

        result = compute_prs(
            epochs, cfg=ERSP_CFG,
            subject=subject, session=sess,
            channel_picks=list(PRS_CHANNELS),
        )

        out_path = DERIVATIVES_ROOT / subject / "prs" / f"{subject}_BFB_{sess}_prs.h5"
        save_prs(result, out_path)
        logger.info(
            "PRS %s session %d: %d trials, prs_alpha_Pz=%.3f dB",
            subject, sess, result.n_trials,
            result.scalars.get("prs_alpha_Pz", float("nan")),
        )
    return 0


def stage_prs_stats(study_json: str) -> int:
    """Group-level PRS contrasts (pooled Active vs Sham, per-group vs Sham,
    SMR-vs-Beta linear contrast, ERD–PRS correlation, LME, and cluster
    permutation at Pz). Writes ``prs_group_stats.json`` under
    ``DERIVATIVES_ROOT / prs``.
    """
    import json
    from tools.prs import load_prs
    from tools.study import Study

    study = Study.from_json(
        Path(study_json),
        analyses_root=ANALYSES_ROOT,
        derivatives_root=DERIVATIVES_ROOT,
    )
    subjects = study.included()

    subject_rows: list[dict] = []
    tfrs_Pz: list[np.ndarray] = []  # type: ignore[name-defined]
    tfr_meta: list[dict] = []
    freqs_ref = None
    times_ref = None

    import numpy as np

    for subj in subjects:
        per_session_scalars: dict[str, list[float]] = {}
        per_session_pz_tfrs: list[np.ndarray] = []
        n_sessions_loaded = 0

        for sess in (1, 3, 5, 6):
            path = DERIVATIVES_ROOT / subj.subject_id / "prs" / f"{subj.subject_id}_BFB_{sess}_prs.h5"
            if not path.is_file():
                continue
            r = load_prs(path)
            n_sessions_loaded += 1
            for k, v in r.scalars.items():
                per_session_scalars.setdefault(k, []).append(v)

            if "Pz" in r.channel_names:
                pz_idx = r.channel_names.index("Pz")
                per_session_pz_tfrs.append(r.ersp[pz_idx])
                if freqs_ref is None:
                    freqs_ref = r.freqs
                    times_ref = r.times

        if n_sessions_loaded == 0:
            continue

        row = {
            "subject": subj.subject_id,
            "group": subj.group,
            "n_sessions": n_sessions_loaded,
        }
        for k, vals in per_session_scalars.items():
            vals_f = [v for v in vals if not np.isnan(v)]
            row[k] = float(np.mean(vals_f)) if vals_f else float("nan")
        subject_rows.append(row)

        if per_session_pz_tfrs:
            tfrs_Pz.append(np.mean(np.stack(per_session_pz_tfrs, axis=0), axis=0))
            tfr_meta.append({"subject": subj.subject_id, "group": subj.group})

    def _vals(rows: list[dict], key: str, groups: tuple[str, ...] | None = None) -> np.ndarray:
        if groups is None:
            filtered = rows
        else:
            filtered = [r for r in rows if r["group"] in groups]
        vals = [r.get(key, float("nan")) for r in filtered]
        vals = np.array([v for v in vals if not np.isnan(v)], dtype=float)
        return vals

    from scipy import stats as spstats

    def _welch(a: np.ndarray, b: np.ndarray) -> dict:
        if len(a) < 2 or len(b) < 2:
            return {"n_a": int(len(a)), "n_b": int(len(b)), "t": float("nan"),
                    "df": float("nan"), "p": float("nan"), "d": float("nan")}
        t, p = spstats.ttest_ind(a, b, equal_var=False)
        sa, sb = a.std(ddof=1), b.std(ddof=1)
        na, nb = len(a), len(b)
        s_pooled = np.sqrt(((na - 1) * sa ** 2 + (nb - 1) * sb ** 2) / (na + nb - 2))
        d = float((a.mean() - b.mean()) / s_pooled) if s_pooled > 0 else float("nan")
        df_welch = ((sa ** 2 / na + sb ** 2 / nb) ** 2) / (
            ((sa ** 2 / na) ** 2) / (na - 1) + ((sb ** 2 / nb) ** 2) / (nb - 1)
        ) if sa > 0 and sb > 0 else float("nan")
        return {"n_a": int(na), "n_b": int(nb), "t": float(t), "df": float(df_welch),
                "p": float(p), "d": d, "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                "sd_a": float(sa), "sd_b": float(sb)}

    from tools.stats import bayes_factor_ttest

    def _bf01_two_sample(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 2 or len(b) < 2:
            return float("nan")
        return float(bayes_factor_ttest(a, b, paired=False))

    def _bf01_one_sample(a: np.ndarray) -> float:
        if len(a) < 2:
            return float("nan")
        return float(bayes_factor_ttest(a, b=None, paired=False))

    def _fdr_bh(pvals: list[float]) -> list[float]:
        p = np.array(pvals, dtype=float)
        n = len(p)
        order = np.argsort(p)
        ranked = p[order]
        adj = ranked * n / (np.arange(n) + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        out = np.empty(n)
        out[order] = adj
        return [float(x) for x in out]

    ACTIVE_GROUPS = ("c3_smr", "c3_beta", "c4_smr")
    SMR_GROUPS = ("c3_smr", "c4_smr")

    results: dict = {"n_subjects_loaded": len(subject_rows),
                     "channels": ["Pz", "P3", "P4", "POz", "PO3", "PO4", "Fz"],
                     "band_alpha_hz": [8.0, 12.0],
                     "window_ms": [800, 1500],
                     "baseline_ms": [-100, 0]}

    descriptives: dict = {}
    for grp in ("c3_smr", "c3_beta", "c4_smr", "sham"):
        v = _vals(subject_rows, "prs_alpha_Pz", (grp,))
        descriptives[grp] = {
            "n": int(len(v)),
            "prs_alpha_Pz_mean": float(v.mean()) if len(v) else float("nan"),
            "prs_alpha_Pz_sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
        }
    results["descriptives_Pz"] = descriptives

    contrasts: dict = {}
    active_pz = _vals(subject_rows, "prs_alpha_Pz", ACTIVE_GROUPS)
    sham_pz = _vals(subject_rows, "prs_alpha_Pz", ("sham",))
    pooled = _welch(active_pz, sham_pz)
    pooled["bf01"] = _bf01_two_sample(active_pz, sham_pz)
    contrasts["pooled_active_vs_sham_Pz"] = pooled

    per_group = {}
    ps = []
    keys = []
    for grp in ACTIVE_GROUPS:
        v = _vals(subject_rows, "prs_alpha_Pz", (grp,))
        res = _welch(v, sham_pz)
        res["bf01"] = _bf01_two_sample(v, sham_pz)
        per_group[f"{grp}_vs_sham_Pz"] = res
        ps.append(res["p"]); keys.append(f"{grp}_vs_sham_Pz")
    p_adj = _fdr_bh(ps)
    for k, pa in zip(keys, p_adj):
        per_group[k]["p_adj"] = pa
    contrasts["per_group_vs_sham_Pz"] = per_group

    smr_group_mean = _vals(subject_rows, "prs_alpha_Pz", SMR_GROUPS)
    beta_group = _vals(subject_rows, "prs_alpha_Pz", ("c3_beta",))
    contrasts["smr_vs_beta_Pz"] = _welch(smr_group_mean, beta_group)
    contrasts["smr_vs_beta_Pz"]["bf01"] = _bf01_two_sample(smr_group_mean, beta_group)

    def _one_sample(a: np.ndarray) -> dict:
        if len(a) < 2:
            return {"n": int(len(a)), "t": float("nan"), "p": float("nan"),
                    "mean": float("nan"), "sd": float("nan"), "bf01": float("nan")}
        t, p = spstats.ttest_1samp(a, 0.0)
        return {"n": int(len(a)), "t": float(t), "p": float(p),
                "mean": float(a.mean()), "sd": float(a.std(ddof=1)),
                "bf01": _bf01_one_sample(a)}

    one_sample = {grp: _one_sample(_vals(subject_rows, "prs_alpha_Pz", (grp,)))
                  for grp in ("c3_smr", "c3_beta", "c4_smr", "sham")}
    contrasts["one_sample_vs_zero_Pz"] = one_sample

    erd_vals, prs_vals = [], []
    for r in subject_rows:
        if r["group"] not in ACTIVE_GROUPS:
            continue
        erd_key = "primary_erd_C4" if r["group"] == "c4_smr" else "primary_erd_C3"
        ersp_path = DERIVATIVES_ROOT / r["subject"] / "ersp"
        erd_subj = []
        if ersp_path.is_dir():
            for sess in (1, 3, 5, 6):
                p = ersp_path / f"{r['subject']}_BFB_{sess}_ersp.h5"
                if not p.is_file():
                    continue
                import h5py
                with h5py.File(p, "r") as fh:
                    key = f"scalar/{erd_key}"
                    if key in fh.attrs:
                        v = float(fh.attrs[key])
                        if not np.isnan(v):
                            erd_subj.append(v)
        if erd_subj and not np.isnan(r.get("prs_alpha_Pz", float("nan"))):
            erd_vals.append(np.mean(erd_subj))
            prs_vals.append(r["prs_alpha_Pz"])
    if len(erd_vals) >= 3:
        pearson_r, pearson_p = spstats.pearsonr(erd_vals, prs_vals)
        spearman_r, spearman_p = spstats.spearmanr(erd_vals, prs_vals)
        contrasts["erd_prs_correlation_active"] = {
            "n": int(len(erd_vals)),
            "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
        }
    else:
        contrasts["erd_prs_correlation_active"] = {"n": int(len(erd_vals)),
                                                     "note": "insufficient data"}

    results["contrasts_Pz"] = contrasts

    active_fz = _vals(subject_rows, "prs_alpha_Fz", ACTIVE_GROUPS)
    sham_fz = _vals(subject_rows, "prs_alpha_Fz", ("sham",))
    control_fz_pooled = _welch(active_fz, sham_fz)
    control_fz_pooled["bf01"] = _bf01_two_sample(active_fz, sham_fz)
    results["control_Fz_pooled_active_vs_sham"] = control_fz_pooled

    cluster_result = {"status": "not-run"}
    if tfrs_Pz and freqs_ref is not None and times_ref is not None:
        try:
            from mne.stats import permutation_cluster_test
            tfrs_arr = np.stack(tfrs_Pz, axis=0)  # (n_subjects, n_freqs, n_times)
            groups_arr = np.array([m["group"] for m in tfr_meta])
            active_mask = np.isin(groups_arr, ACTIVE_GROUPS)
            sham_mask = groups_arr == "sham"
            if active_mask.sum() >= 2 and sham_mask.sum() >= 2:
                t_win = (times_ref >= 0.0) & (times_ref <= 1.5)
                f_win = (freqs_ref >= 3.0) & (freqs_ref <= 20.0)
                a_tfr = tfrs_arr[np.ix_(active_mask, f_win, t_win)]
                s_tfr = tfrs_arr[np.ix_(sham_mask, f_win, t_win)]
                T_obs, clusters, cluster_pv, _ = permutation_cluster_test(
                    [a_tfr, s_tfr], n_permutations=1000, threshold=2.0,
                    tail=1, n_jobs=1, verbose="WARNING",
                    out_type="mask",
                )
                cluster_result = {
                    "status": "ok",
                    "n_clusters": int(len(clusters)),
                    "cluster_pvalues": [float(p) for p in cluster_pv],
                    "min_cluster_p": (float(min(cluster_pv)) if len(cluster_pv) else float("nan")),
                    "window_ms": [0, 1500], "freq_hz": [3, 20],
                    "tail": "one-sided (active > sham)",
                    "threshold_t": 2.0, "n_permutations": 1000,
                    "n_subjects_active": int(active_mask.sum()),
                    "n_subjects_sham": int(sham_mask.sum()),
                }
            else:
                cluster_result = {"status": "insufficient-subjects"}
        except Exception as e:  # noqa: BLE001
            logger.warning("Cluster permutation at Pz failed: %s", e)
            cluster_result = {"status": f"failed: {e}"}
    results["cluster_permutation_Pz"] = cluster_result

    out_dir = DERIVATIVES_ROOT / "prs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "prs_group_stats.json"
    with open(out_json, "w") as f:
        json.dump({"subjects": subject_rows, **results}, f, indent=2)
    logger.info("PRS group stats written: %s", out_json)

    if tfrs_Pz and freqs_ref is not None and times_ref is not None:
        import h5py
        tfr_path = out_dir / "prs_Pz_tfrs.h5"
        with h5py.File(tfr_path, "w") as f:
            f.create_dataset("tfrs", data=np.stack(tfrs_Pz, axis=0))
            f.create_dataset("freqs", data=freqs_ref)
            f.create_dataset("times", data=times_ref)
            f.attrs["subjects"] = [m["subject"] for m in tfr_meta]
            f.attrs["groups"] = [m["group"] for m in tfr_meta]
        logger.info("PRS Pz TFRs written: %s", tfr_path)

    return 0


# ── erp ────────────────────────────────────────────────────────────────────


def stage_erp(subject: str, sessions: Sequence[int]) -> int:
    """Compute reward-evoked ERPs from saved epochs."""
    import mne
    import numpy as np
    from tools.epochs import create_reward_epochs
    from tools.erp import bandpass_for_erp, compute_erp, save_erp

    for sess in sessions:
        clean_path = ANALYSES_ROOT / subject / "preprocessed" / f"{subject}_BFB_{sess}_clean-raw.fif"
        if not clean_path.is_file():
            logger.error("Preprocessed file missing: %s", clean_path)
            continue

        events_path = (
            ANALYSES_ROOT / subject / "preprocessed"
            / f"{subject}_BFB_{sess}_events.npy"
        )
        if not events_path.is_file():
            logger.error("Events sidecar missing: %s", events_path)
            continue

        raw = mne.io.read_raw_fif(clean_path, preload=True, verbose="WARNING")
        raw_erp = bandpass_for_erp(raw)
        events = np.load(events_path)

        if len(events) == 0:
            logger.warning("No events for %s session %d — skipping ERP", subject, sess)
            continue

        try:
            epochs = create_reward_epochs(raw_erp, events, cfg=ERSP_CFG)
        except ValueError:
            logger.warning("No valid ERP epochs for %s session %d — skipping", subject, sess)
            continue

        if len(epochs) == 0:
            logger.warning("Empty ERP epochs for %s session %d — skipping", subject, sess)
            continue

        result = compute_erp(epochs, cfg=ERP_CFG, subject=subject, session=sess)

        out_path = DERIVATIVES_ROOT / subject / "erp" / f"{subject}_BFB_{sess}_erp.h5"
        save_erp(result, out_path)
        logger.info("ERP %s session %d: %d trials, %d components measured",
                     subject, sess, result.n_trials, len(result.components))
    return 0


# ── resting ────────────────────────────────────────────────────────────────


def stage_resting(subject: str, sessions: Sequence[int]) -> int:
    """Compute resting-state PSD for EC/EO recordings with full preprocessing."""
    from tools.io import find_recording, load_raw
    from tools.preprocess import preprocess
    from tools.resting import analyze_resting_clean, save_psd

    for rtype in RESTING_TYPES:
        for sess in sessions:
            path = find_recording(subject, rtype, sess)
            if path is None:
                logger.warning("Missing %s session %d for %s", rtype, sess, subject)
                continue

            raw = load_raw(path)

            prep_result = preprocess(
                raw, cfg=PREPROCESS_CFG, subject=subject, session=sess,
            )
            logger.info(
                "Resting preprocess %s %s session %d: %d bad ch, %d ICA rejected",
                subject, rtype, sess,
                len(prep_result.bad_channels), len(prep_result.rejected_ic_labels),
            )

            result = analyze_resting_clean(
                raw, rtype, sess, cfg=RESTING_CFG, subject=subject,
            )
            out_path = DERIVATIVES_ROOT / subject / "resting" / f"{subject}_{rtype}_{sess}_psd.h5"
            save_psd(result, out_path)
            logger.info(
                "Resting %s %s session %d: PSD computed (%d/%d segments clean)",
                subject, rtype, sess,
                result.n_clean_segments, result.n_total_segments,
            )
    return 0


# ── stats ──────────────────────────────────────────────────────────────────


def stage_stats(study_json: str) -> int:
    """Run group-level statistics (requires completed ERSP for all subjects)."""
    from tools.study import Study
    from tools.ersp import load_ersp
    from tools.stats import (
        run_all_planned_contrasts,
        fit_mixed_model_with_slope,
        resting_persistence,
        erd_predicts_resting_change,
        retention_paired_contrasts,
    )
    from tools.group import assemble_resting_change, assemble_ersp_scalars
    from tools.config import STATS as STATS_CFG, GROUPS
    import numpy as np
    import pandas as pd

    study = Study.from_json(
        Path(study_json),
        analyses_root=ANALYSES_ROOT,
        derivatives_root=DERIVATIVES_ROOT,
    )
    logger.info(study.summary())

    stats_dir = DERIVATIVES_ROOT / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect per-subject per-session ERD at C3 ─────────────────────
    group_erd: dict = {gkey: [] for gkey in ("c3_smr", "c3_beta", "c4_smr", "sham")}
    long_rows: list[dict] = []

    for subj in study.included():
        subj_erds = []
        for sess in BFB_SESSIONS:
            h5 = study.ersp_h5(subj.subject_id, sess)
            if not h5.is_file():
                logger.warning("Missing ERSP: %s session %d", subj.subject_id, sess)
                continue
            result = load_ersp(h5)
            val = result.scalars.get("primary_erd_C3", float("nan"))
            subj_erds.append(val)
            if not np.isnan(val):
                long_rows.append({
                    "subject": subj.subject_id,
                    "group": subj.group,
                    "session": sess,
                    "erd": val,
                })

        if subj_erds:
            group_erd[subj.group].append(np.nanmean(subj_erds))

    # ── Pairwise planned contrasts (FDR-corrected) ────────────────────
    group_arrays = {k: np.array(v) for k, v in group_erd.items() if v}
    contrasts = run_all_planned_contrasts(group_arrays, STATS_CFG)

    logger.info("=== Planned Contrasts (session-averaged ERD at C3) ===")
    for c in contrasts:
        logger.info(
            "  %s: d=%.3f [%.3f, %.3f], p=%.4f, p_adj=%.4f, BF01=%.2f",
            c.name, c.cohens_d, c.ci_low, c.ci_high, c.p_value,
            c.p_adjusted if c.p_adjusted is not None else float("nan"),
            c.bf01 if c.bf01 is not None else float("nan"),
        )

    # ── LME omnibus: Group × Session on per-session ERD ───────────────
    long_df = pd.DataFrame(long_rows)
    if len(long_df) > 0:
        logger.info("=== LME Omnibus: Group × Session on ERD at C3 ===")
        logger.info("  Long-format data: %d observations, %d subjects",
                     len(long_df), long_df["subject"].nunique())

        lme_result = fit_mixed_model_with_slope(
            long_df, dv="erd", group_col="group",
            session_col="session", subject_col="subject",
        )

        logger.info("  LME formula: %s", lme_result.formula)
        logger.info("  AIC=%.1f  BIC=%.1f  n_obs=%d  n_groups=%d",
                     lme_result.aic, lme_result.bic,
                     lme_result.n_obs, lme_result.n_groups)

        logger.info("  Fixed effects:")
        for term, coef in lme_result.coefficients.items():
            p = lme_result.p_values.get(term, float("nan"))
            se = lme_result.std_errors.get(term, float("nan"))
            logger.info("    %-50s  β=%7.4f  SE=%6.4f  p=%.4f", term, coef, se, p)

        if lme_result.eta_squared:
            logger.info("  Partial eta-squared (from mixed ANOVA):")
            for source, eta in lme_result.eta_squared.items():
                logger.info("    %-20s  ηp²=%.4f", source, eta)

        lme_path = stats_dir / "lme_erd_c3.txt"
        with open(lme_path, "w") as f:
            f.write("LME Omnibus: Group × Session on ERD at C3\n")
            f.write(f"Formula: {lme_result.formula}\n")
            f.write(f"AIC={lme_result.aic:.1f}  BIC={lme_result.bic:.1f}\n")
            f.write(f"n_obs={lme_result.n_obs}  n_subjects={lme_result.n_groups}\n\n")
            f.write("Fixed effects:\n")
            for term, coef in lme_result.coefficients.items():
                p = lme_result.p_values.get(term, float("nan"))
                se = lme_result.std_errors.get(term, float("nan"))
                f.write(f"  {term:50s}  β={coef:7.4f}  SE={se:6.4f}  p={p:.4f}\n")
            if lme_result.eta_squared:
                f.write("\nPartial eta-squared (mixed ANOVA):\n")
                for source, eta in lme_result.eta_squared.items():
                    f.write(f"  {source:20s}  ηp²={eta:.4f}\n")
        logger.info("  LME results saved to %s", lme_path)

    # ── Resting-state analyses (PAPER.md §2d) ─────────────────────────
    # 2d.3: Change scores (S6 - S1) by group
    logger.info("=== Resting-State Analysis ===")
    for rtype in ("ECPRE", "EOPRE"):
        resting_df = assemble_resting_change(study, resting_type=rtype)
        if resting_df.empty:
            logger.warning("No resting change data for %s", rtype)
            continue

        for band in ("smr", "beta", "theta", "alpha"):
            band_df = resting_df[resting_df["band"] == band]
            if band_df.empty:
                continue
            results = resting_persistence(band_df)
            for c in results:
                logger.info(
                    "  Resting Δ %s %s: d=%.3f, p=%.4f, p_adj=%.4f (%s)",
                    rtype, band, c.cohens_d, c.p_value,
                    c.p_adjusted if c.p_adjusted is not None else float("nan"),
                    c.name,
                )

    # 2d.4: ERD predicts resting change (active subjects only)
    scalars_df = assemble_ersp_scalars(study)
    ec_resting = assemble_resting_change(study, resting_type="ECPRE")
    if not scalars_df.empty and not ec_resting.empty:
        active_erd = scalars_df[
            (scalars_df["group"] != "sham")
            & scalars_df["metric"].str.startswith("primary_erd_C3")
        ]
        active_rest = ec_resting[ec_resting["group"] != "sham"]
        for band in ("smr", "beta"):
            band_rest = active_rest[active_rest["band"] == band]
            if not band_rest.empty:
                reg = erd_predicts_resting_change(active_erd, band_rest)
                logger.info(
                    "  ERD→resting (%s): r²=%.3f, β=%.3f, p=%.4f, n=%d",
                    band, reg["r_squared"], reg["beta"], reg["p_value"], reg["n"],
                )

    # 2d.2: EC vs EO dissociation
    ec_df = assemble_resting_change(study, resting_type="ECPRE")
    eo_df = assemble_resting_change(study, resting_type="EOPRE")
    if not ec_df.empty and not eo_df.empty:
        logger.info("  --- EC vs EO Dissociation ---")
        ec_df = ec_df.copy()
        eo_df = eo_df.copy()
        ec_df["condition"] = "EC"
        eo_df["condition"] = "EO"
        eceo = pd.concat([ec_df, eo_df], ignore_index=True)

        for band in ("smr", "beta", "alpha", "theta"):
            band_eceo = eceo[eceo["band"] == band]
            if band_eceo.empty:
                continue
            subj_counts = band_eceo.groupby("subject")["condition"].nunique()
            complete_subjs = subj_counts[subj_counts == 2].index
            band_complete = band_eceo[band_eceo["subject"].isin(complete_subjs)]
            if band_complete["subject"].nunique() < 4:
                logger.warning("  Too few complete-case subjects for EC/EO %s", band)
                continue
            try:
                import pingouin as pg
                aov = pg.mixed_anova(
                    data=band_complete, dv="delta",
                    within="condition", between="group",
                    subject="subject",
                )
                logger.info("  EC/EO × Group ANOVA (%s band):", band)
                for _, row in aov.iterrows():
                    logger.info(
                        "    %s: F=%.3f, p=%.4f, ηp²=%.4f",
                        row["Source"], row["F"], row["p_unc"], row["np2"],
                    )
            except Exception as exc:
                logger.warning("  EC/EO ANOVA failed for %s: %s", band, exc)

    # Save resting stats summary
    resting_stats_path = stats_dir / "resting_stats.txt"
    with open(resting_stats_path, "w") as f:
        f.write("Resting-State Analysis Summary\n")
        f.write("=" * 60 + "\n\n")
        for rtype in ("ECPRE", "EOPRE"):
            resting_df = assemble_resting_change(study, resting_type=rtype)
            if resting_df.empty:
                continue
            f.write(f"\n{rtype}: S6 - S1 Change Scores\n")
            f.write("-" * 40 + "\n")
            for band in ("smr", "beta", "theta", "alpha"):
                band_df = resting_df[resting_df["band"] == band]
                if band_df.empty:
                    continue
                results = resting_persistence(band_df)
                for c in results:
                    f.write(
                        f"  {c.name}: d={c.cohens_d:.3f}, "
                        f"p={c.p_value:.4f}, "
                        f"p_adj={c.p_adjusted:.4f}\n"
                        if c.p_adjusted is not None
                        else f"  {c.name}: d={c.cohens_d:.3f}, p={c.p_value:.4f}\n"
                    )
    logger.info("  Resting stats saved to %s", resting_stats_path)

    # ── ERP Component ANOVA (Group × Session) ─────────────────────────
    from tools.erp import load_erp
    logger.info("=== ERP Component ANOVA ===")
    erp_rows: list[dict] = []
    for subj in study.included():
        for sess in BFB_SESSIONS:
            h5 = study.erp_h5(subj.subject_id, sess)
            if not h5.is_file():
                continue
            erp = load_erp(h5)
            for m in erp.components:
                erp_rows.append({
                    "subject": subj.subject_id,
                    "group": subj.group,
                    "session": sess,
                    "component": m.component,
                    "channel": m.channel,
                    "peak_uv": m.peak_amplitude_uv,
                    "mean_uv": m.mean_amplitude_uv,
                })

    erp_df = pd.DataFrame(erp_rows)
    erp_results_lines: list[str] = []
    if not erp_df.empty:
        logger.info("  ERP data: %d rows, %d subjects",
                     len(erp_df), erp_df["subject"].nunique())
        try:
            import pingouin as pg
            for comp in ("P50", "N1", "P2"):
                for ch in ("C3", "C4", "Pz"):
                    subset = erp_df[
                        (erp_df["component"] == comp) & (erp_df["channel"] == ch)
                    ].copy()
                    if subset.empty or subset["subject"].nunique() < 4:
                        continue
                    subj_counts = subset.groupby("subject")["session"].nunique()
                    complete = subj_counts[subj_counts == subset["session"].nunique()].index
                    subset = subset[subset["subject"].isin(complete)]
                    if subset["subject"].nunique() < 4:
                        continue
                    try:
                        aov = pg.mixed_anova(
                            data=subset, dv="mean_uv",
                            within="session", between="group",
                            subject="subject",
                        )
                        header = f"  {comp} at {ch}:"
                        logger.info(header)
                        for _, row in aov.iterrows():
                            line = (
                                f"    {row['Source']}: F({row['DF1']:.0f},{row['DF2']:.0f})"
                                f"={row['F']:.3f}, p={row['p_unc']:.4f}, ηp²={row['np2']:.4f}"
                            )
                            logger.info(line)
                            erp_results_lines.append(f"{comp} {ch} {line.strip()}")
                    except Exception as exc:
                        logger.warning("  ERP ANOVA failed for %s %s: %s", comp, ch, exc)
        except ImportError:
            logger.warning("  pingouin not available for ERP ANOVA")

        erp_path = stats_dir / "erp_anova.txt"
        with open(erp_path, "w") as f:
            f.write("ERP Component ANOVA: Group × Session\n")
            f.write("=" * 60 + "\n\n")
            for line in erp_results_lines:
                f.write(line + "\n")
        logger.info("  ERP ANOVA saved to %s", erp_path)

        # ── P2 post-hoc contrasts at C3 ──────────────────────────────
        p2_c3 = erp_df[
            (erp_df["component"] == "P2") & (erp_df["channel"] == "C3")
        ].copy()
        if not p2_c3.empty:
            p2_subj_means = p2_c3.groupby(["subject", "group"])["mean_uv"].mean().reset_index()
            p2_groups: dict = {}
            for gkey in ("c3_smr", "c3_beta", "c4_smr", "sham"):
                vals = p2_subj_means.loc[
                    p2_subj_means["group"] == gkey, "mean_uv"
                ].values
                if len(vals) > 0:
                    p2_groups[gkey] = vals

            if "sham" in p2_groups and len(p2_groups) > 1:
                logger.info("=== P2 Post-Hoc Contrasts at C3 ===")
                p2_contrasts = run_all_planned_contrasts(p2_groups, STATS_CFG)
                p2_lines: list[str] = []
                for c in p2_contrasts:
                    line = (
                        f"  {c.name}: d={c.cohens_d:.3f} [{c.ci_low:.3f}, {c.ci_high:.3f}], "
                        f"p={c.p_value:.4f}, p_adj={c.p_adjusted:.4f}, BF01={c.bf01:.2f}"
                    )
                    logger.info(line)
                    p2_lines.append(line.strip())

                p2_path = stats_dir / "p2_posthoc.txt"
                with open(p2_path, "w") as f:
                    f.write("P2 Post-Hoc Contrasts at C3 (session-averaged mean amplitude)\n")
                    f.write("=" * 60 + "\n\n")
                    for line in p2_lines:
                        f.write(line + "\n")
                logger.info("  P2 post-hoc saved to %s", p2_path)

    # ── Retention contrasts: S5 vs S6 per active group ────────────────
    if len(long_df) > 0:
        logger.info("=== Retention Contrasts: Session 5 → Session 6 ===")
        retention_results = retention_paired_contrasts(long_df)
        for c in retention_results:
            logger.info(
                "  %s: d=%.3f [%.3f, %.3f], p=%.4f, p_adj=%.4f, BF01=%.2f",
                c.name, c.cohens_d, c.ci_low, c.ci_high, c.p_value,
                c.p_adjusted if c.p_adjusted is not None else float("nan"),
                c.bf01 if c.bf01 is not None else float("nan"),
            )
        retention_path = stats_dir / "retention_contrasts.txt"
        with open(retention_path, "w") as f:
            f.write("Retention Contrasts: Session 5 → Session 6 (paired t-tests)\n")
            f.write("=" * 60 + "\n\n")
            for c in retention_results:
                f.write(
                    f"{c.name}: d={c.cohens_d:.3f} [{c.ci_low:.3f}, {c.ci_high:.3f}], "
                    f"p={c.p_value:.4f}, p_adj={c.p_adjusted:.4f}, BF01={c.bf01:.2f}\n"
                )
        logger.info("  Retention contrasts saved to %s", retention_path)

    # ── Resting-state trajectory analyses ─────────────────────────────
    from tools.group import assemble_resting_trajectory
    from scipy.stats import ttest_rel as _ttest_rel

    rest_traj = assemble_resting_trajectory(study)
    if not rest_traj.empty:
        # --- Within-session shifts (POST minus PRE, same day) ---
        logger.info("=== Within-Session Resting Shifts (POST - PRE) ===")
        ws_lines: list[str] = []
        for eyes in ("EC", "EO"):
            pre_cond = f"{eyes}PRE"
            post_cond = f"{eyes}POST"
            pre_df = rest_traj[rest_traj["condition"] == pre_cond]
            post_df = rest_traj[rest_traj["condition"] == post_cond]
            if pre_df.empty or post_df.empty:
                continue
            merged = pre_df.merge(
                post_df, on=["subject", "group", "session", "channel", "band"],
                suffixes=("_pre", "_post"),
            )
            merged["delta"] = merged["power_post"] - merged["power_pre"]

            for band in ("alpha", "smr", "beta", "theta"):
                for ch in ("C3", "C4"):
                    subset = merged[
                        (merged["band"] == band) & (merged["channel"] == ch)
                    ]
                    for grp in ("c3_smr", "c3_beta", "c4_smr", "sham"):
                        gs = subset[subset["group"] == grp]
                        subj_means = gs.groupby("subject")["delta"].mean()
                        if len(subj_means) < 3:
                            continue
                        vals = subj_means.values
                        from scipy.stats import ttest_1samp
                        t_stat, p_val = ttest_1samp(vals, 0)
                        mean_d = float(vals.mean())
                        sd = float(vals.std(ddof=1))
                        d_val = mean_d / sd if sd > 0 else 0.0
                        line = (
                            f"  {eyes} {band} {ch} {grp}: "
                            f"mean Δ={mean_d:.4f}, d={d_val:.3f}, "
                            f"t={t_stat:.3f}, p={p_val:.4f}, n={len(subj_means)}"
                        )
                        logger.info(line)
                        ws_lines.append(line.strip())

        ws_path = stats_dir / "within_session_resting.txt"
        with open(ws_path, "w") as f:
            f.write("Within-Session Resting Shifts (POST - PRE, session-averaged)\n")
            f.write("=" * 60 + "\n\n")
            for line in ws_lines:
                f.write(line + "\n")
        logger.info("  Within-session resting shifts saved to %s", ws_path)

        # --- ERD predicts within-session alpha shift? ---
        logger.info("=== ERD vs Within-Session EC Alpha Shift ===")
        try:
            from scipy.stats import pearsonr as _pearsonr

            ec_pre = rest_traj[rest_traj["condition"] == "ECPRE"]
            ec_post = rest_traj[rest_traj["condition"] == "ECPOST"]
            if not ec_pre.empty and not ec_post.empty:
                ws_merged = ec_pre.merge(
                    ec_post,
                    on=["subject", "group", "session", "channel", "band"],
                    suffixes=("_pre", "_post"),
                )
                ws_merged["delta"] = ws_merged["power_post"] - ws_merged["power_pre"]
                ws_alpha_c3 = ws_merged[
                    (ws_merged["band"] == "alpha") & (ws_merged["channel"] == "C3")
                ]
                ws_subj = ws_alpha_c3.groupby(["subject", "group"])["delta"].mean().reset_index()

                erd_scalars = assemble_ersp_scalars(study)
                erd_c3 = erd_scalars[
                    erd_scalars["metric"].str.startswith("primary_erd_C3")
                ]
                erd_subj = erd_c3.groupby(["subject", "group"])["value"].mean().reset_index()
                erd_subj.rename(columns={"value": "erd"}, inplace=True)

                both = ws_subj.merge(erd_subj[["subject", "erd"]], on="subject")

                for label, df in [("active (n excl sham)", both[both["group"] != "sham"]),
                                  ("all subjects", both)]:
                    if len(df) < 5:
                        continue
                    r, p = _pearsonr(df["erd"].values, df["delta"].values)
                    logger.info(
                        "  ERD vs within-session EC alpha shift (%s): "
                        "r=%.3f, r²=%.3f, p=%.4f, n=%d",
                        label, r, r**2, p, len(df),
                    )
        except Exception as exc:
            logger.warning("  ERD-within-session correlation failed: %s", exc)

        # --- Across-session trajectory (PRE baseline growth curve) ---
        logger.info("=== Resting-State Trajectory (PRE baseline across sessions) ===")
        gc_lines: list[str] = []
        for eyes in ("EC", "EO"):
            pre_cond = f"{eyes}PRE"
            pre_only = rest_traj[rest_traj["condition"] == pre_cond].copy()
            if pre_only.empty:
                continue

            for band in ("alpha", "smr", "beta", "theta"):
                for ch in ("C3", "C4"):
                    subset = pre_only[
                        (pre_only["band"] == band) & (pre_only["channel"] == ch)
                    ]
                    for grp in ("c3_smr", "c3_beta", "c4_smr", "sham"):
                        gs = subset[subset["group"] == grp].copy()
                        if gs["subject"].nunique() < 3:
                            continue
                        pivot = gs.pivot_table(
                            index="subject", columns="session", values="power",
                        )
                        if pivot.shape[1] < 2:
                            continue
                        sessions_present = sorted(pivot.columns)
                        means = [float(pivot[s].mean()) for s in sessions_present]
                        header = (
                            f"  {eyes} {band} {ch} {grp}: "
                            + " → ".join(
                                f"S{s}={m:.4f}" for s, m in zip(sessions_present, means)
                            )
                        )
                        gc_lines.append(header.strip())
                        logger.info(header)

                        if 1 in pivot.columns:
                            for later_s in [3, 5, 6]:
                                if later_s not in pivot.columns:
                                    continue
                                common = pivot[[1, later_s]].dropna()
                                if len(common) < 3:
                                    continue
                                t_stat, p_val = _ttest_rel(
                                    common[later_s].values, common[1].values,
                                )
                                diff = common[later_s].values - common[1].values
                                sd = float(diff.std(ddof=1))
                                d_val = float(diff.mean() / sd) if sd > 0 else 0.0
                                line = (
                                    f"    S{later_s}-S1: d={d_val:.3f}, "
                                    f"t={t_stat:.3f}, p={p_val:.4f}, n={len(common)}"
                                )
                                gc_lines.append(line.strip())
                                logger.info(line)

        # LME growth curve for EC alpha at C3/C4 (focused test)
        try:
            import statsmodels.formula.api as smf
            ec_pre_alpha = rest_traj[
                (rest_traj["condition"] == "ECPRE")
                & (rest_traj["band"] == "alpha")
            ].copy()
            if not ec_pre_alpha.empty:
                ec_pre_alpha["session_c"] = ec_pre_alpha["session"] - 1
                for ch in ("C3", "C4"):
                    ch_df = ec_pre_alpha[ec_pre_alpha["channel"] == ch].copy()
                    if ch_df["subject"].nunique() < 4:
                        continue
                    try:
                        lme = smf.mixedlm(
                            "power ~ C(group) * session_c",
                            data=ch_df,
                            groups=ch_df["subject"],
                        ).fit(reml=True)
                        header = f"\n  LME Growth Curve: EC alpha at {ch}"
                        logger.info(header)
                        gc_lines.append(header.strip())
                        for term in lme.fe_params.index:
                            coef = lme.fe_params[term]
                            pval = lme.pvalues[term]
                            se = lme.bse[term]
                            line = f"    {term}: β={coef:.4f}, SE={se:.4f}, p={pval:.4f}"
                            logger.info(line)
                            gc_lines.append(line.strip())
                    except Exception as exc:
                        logger.warning("  LME failed for EC alpha %s: %s", ch, exc)
        except ImportError:
            logger.warning("  statsmodels not available for LME growth curve")

        gc_path = stats_dir / "resting_trajectory.txt"
        with open(gc_path, "w") as f:
            f.write("Resting-State Trajectory (PRE baseline across sessions)\n")
            f.write("=" * 60 + "\n\n")
            for line in gc_lines:
                f.write(line + "\n")
        logger.info("  Resting trajectory saved to %s", gc_path)

    return 0


# ── figures ────────────────────────────────────────────────────────────────


def stage_figures(study_json: str) -> int:
    """Generate all publication figures from assembled group data."""
    from tools.config import BANDS as BAND_DEFS, GROUPS, GROUP_REWARD_BAND
    from tools.group import (
        assemble_erp_by_session,
        assemble_ersp_scalars,
        assemble_ersp_tfr,
        assemble_ersp_tfr_by_session,
        assemble_frequency_profile,
        assemble_resting_change,
        assemble_topo_erd,
        recenter_baseline,
    )
    from tools.study import Study
    from tools.viz.composite import plot_composite_summary
    from tools.viz.crossover import plot_frequency_crossover
    from tools.viz.erd_lines import plot_erd_by_session
    from tools.viz.erp_waves import plot_erp_waveforms, plot_erp_p2_focus
    from tools.viz.heatmaps import plot_ersp_difference_map, plot_ersp_heatmaps
    from tools.viz.scatter import plot_erd_resting_scatter
    from tools.viz.topomaps import plot_topo_ersp
    from tools.viz.violins import plot_erd_violins

    import numpy as np

    study = Study.from_json(
        Path(study_json),
        analyses_root=ANALYSES_ROOT,
        derivatives_root=DERIVATIVES_ROOT,
    )
    fig_dir = DERIVATIVES_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    scalars = assemble_ersp_scalars(study)
    if scalars.empty:
        logger.error("No ERSP scalar data — cannot generate figures")
        return 1

    # Figure 1-2: ERSP heatmaps per group at C3 and C4
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dtmin, dtmax = ERSP_CFG.display_tmin, ERSP_CFG.display_tmax
    for ch in ("C3", "C4"):
        for grp_key in GROUPS:
            data, freqs, times, _ = assemble_ersp_tfr(
                study, sessions=BFB_SESSIONS, channel=ch, group=grp_key,
            )
            if data.size == 0:
                continue
            data = recenter_baseline(data, times)
            ersp_mean = data.mean(axis=0)  # (n_freqs, n_times)
            disp_mask = (times >= dtmin) & (times <= dtmax)
            clim = max(abs(np.percentile(ersp_mean[:, disp_mask], 2)),
                       abs(np.percentile(ersp_mean[:, disp_mask], 98)))
            clim = round(clim, 2) if clim > 0.1 else 0.5
            fig, ax = plt.subplots(figsize=(8, 4))
            im = ax.pcolormesh(
                times, freqs, ersp_mean,
                cmap="RdBu_r", vmin=-clim, vmax=clim, shading="auto",
            )
            ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
            ax.set_xlim(dtmin, dtmax)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_title(f"ERSP {ch} — {GROUPS[grp_key]} (n={data.shape[0]})")
            fig.colorbar(im, ax=ax, label="Power (dB)")
            fig.tight_layout()
            out = fig_dir / f"ersp_{ch}_{grp_key}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            logger.info("Saved %s", out)
    logger.info("Figures 1-2: ERSP heatmaps saved")

    # Figure 3: ERD magnitude by group and session
    erd_data: dict = {}
    for grp_key in GROUPS:
        grp_scalars = scalars[
            (scalars["group"] == grp_key)
            & (scalars["metric"].str.startswith("primary_erd_C3"))
        ]
        sess_arrs: dict = {}
        for sess in BFB_SESSIONS:
            vals = grp_scalars[grp_scalars["session"] == sess]["value"].values
            if len(vals):
                sess_arrs[sess] = vals
        if sess_arrs:
            erd_data[grp_key] = sess_arrs

    if erd_data:
        plot_erd_by_session(
            erd_data, sessions=BFB_SESSIONS,
            save_path=str(fig_dir / "erd_by_session"),
        )
        logger.info("Figure 3: ERD lines saved")

    # Figure 6: ERP waveforms (per-session grand averages)
    erp_sessions = (1, 5)
    session_labels = [f"Session {s}" for s in erp_sessions]
    erp_groups: dict = {}
    ref_times = None
    ref_ch: list = []
    for grp_key in GROUPS:
        by_sess, t, ch_names, ids = assemble_erp_by_session(
            study, sessions=erp_sessions, group=grp_key,
        )
        if not by_sess:
            continue
        erp_groups[grp_key] = {
            f"Session {s}": avg for s, avg in by_sess.items()
        }
        if ref_times is None:
            ref_times = t
            ref_ch = ch_names

    if erp_groups and ref_times is not None:
        plot_erp_waveforms(
            erp_groups, ref_times, channels=ref_ch,
            sessions_shown=session_labels,
            save_path=str(fig_dir / "erp_waveforms"),
        )
        logger.info("Figure 6: ERP waveforms saved")

        plot_erp_p2_focus(
            erp_groups, ref_times, channels=ref_ch,
            save_path=str(fig_dir / "erp_p2_focus"),
        )
        logger.info("Figure 6b: Focused P2 ERP saved")

    # Figure 7: ERD-resting scatter
    resting_df = assemble_resting_change(study)
    if not resting_df.empty:
        erd_means: dict = {}
        rest_deltas: dict = {}
        for grp_key in ("c3_smr", "c3_beta", "c4_smr"):
            grp_erd = scalars[
                (scalars["group"] == grp_key)
                & (scalars["metric"].str.startswith("primary_erd_C3"))
            ]
            subj_means = grp_erd.groupby("subject")["value"].mean()
            grp_rest = resting_df[
                (resting_df["group"] == grp_key) & (resting_df["band"] == "smr")
            ]
            subj_deltas = grp_rest.groupby("subject")["delta"].mean()
            common = subj_means.index.intersection(subj_deltas.index)
            if len(common) > 1:
                erd_means[grp_key] = subj_means.loc[common].values
                rest_deltas[grp_key] = subj_deltas.loc[common].values

        if erd_means:
            plot_erd_resting_scatter(
                erd_means, rest_deltas,
                save_path=str(fig_dir / "erd_resting_scatter"),
            )
            logger.info("Figure 7: ERD-resting scatter saved")

    # Figure 4: Frequency crossover at C3 (all 4 groups)
    freq_profiles: dict = {}
    freq_profiles_se: dict = {}
    ref_freqs_fp = None
    for grp_key in ("c3_smr", "c3_beta", "c4_smr", "sham"):
        mean_p, se_p, f = assemble_frequency_profile(
            study, sessions=BFB_SESSIONS, channel="C3", group=grp_key,
        )
        if mean_p.size > 0:
            freq_profiles[grp_key] = mean_p
            freq_profiles_se[grp_key] = se_p
            if ref_freqs_fp is None:
                ref_freqs_fp = f

    if freq_profiles and ref_freqs_fp is not None:
        plot_frequency_crossover(
            freq_profiles, freq_profiles_se, ref_freqs_fp,
            save_path=str(fig_dir / "frequency_crossover"),
        )
        logger.info("Figure 4: Frequency crossover saved")

    # Figure 4b: Frequency crossover at C4
    freq_profiles_c4: dict = {}
    freq_profiles_c4_se: dict = {}
    ref_freqs_c4 = None
    for grp_key in ("c3_smr", "c3_beta", "c4_smr", "sham"):
        mean_p, se_p, f = assemble_frequency_profile(
            study, sessions=BFB_SESSIONS, channel="C4", group=grp_key,
        )
        if mean_p.size > 0:
            freq_profiles_c4[grp_key] = mean_p
            freq_profiles_c4_se[grp_key] = se_p
            if ref_freqs_c4 is None:
                ref_freqs_c4 = f

    if freq_profiles_c4 and ref_freqs_c4 is not None:
        plot_frequency_crossover(
            freq_profiles_c4, freq_profiles_c4_se, ref_freqs_c4,
            title="Frequency Crossover at C4",
            save_path=str(fig_dir / "frequency_crossover_C4"),
        )
        logger.info("Figure 4b: Frequency crossover C4 saved")

    # Figure 1-2b: ERSP heatmap grid (4 groups x 4 sessions)
    session_labels = [f"Session {s}" for s in BFB_SESSIONS]
    for ch in ("C3", "C4"):
        ersp_grid_data: dict = {}
        grid_freqs = grid_times = None
        for grp_key in GROUPS:
            by_session, gf, gt, _ = assemble_ersp_tfr_by_session(
                study, sessions=BFB_SESSIONS, channel=ch, group=grp_key,
            )
            if not by_session:
                continue
            ersp_grid_data[grp_key] = {
                f"Session {s}": recenter_baseline(arr, gt).mean(axis=0)
                for s, arr in by_session.items()
            }
            if grid_freqs is None:
                grid_freqs = gf
                grid_times = gt

        if ersp_grid_data and grid_freqs is not None:
            plot_ersp_heatmaps(
                ersp_grid_data, grid_times, grid_freqs, channel=ch,
                segments=session_labels,
                save_path=str(fig_dir / f"ersp_grid_{ch}"),
            )
    logger.info("Figures 1-2b: ERSP session grids saved")

    # Figure 5: Topographic ERD maps (sessions 1 and 5)
    topo_data: dict = {}
    topo_info = None
    topo_sessions = [1, 5]
    topo_labels = [f"Session {s}" for s in topo_sessions]
    for grp_key in GROUPS:
        grp_topo: dict = {}
        rb = BAND_DEFS[GROUP_REWARD_BAND[grp_key]]
        for sess in topo_sessions:
            erd_mean, info, ids = assemble_topo_erd(
                study, session=sess, group=grp_key,
                reward_band=rb,
            )
            if erd_mean.size > 0:
                grp_topo[f"Session {sess}"] = erd_mean
                if topo_info is None:
                    topo_info = info
        if grp_topo:
            topo_data[grp_key] = grp_topo

    if topo_data and topo_info is not None:
        plot_topo_ersp(
            topo_data, topo_info,
            sessions_labels=topo_labels,
            save_path=str(fig_dir / "topomaps_erd"),
        )
        logger.info("Figure 5: Topographic maps saved")

    # Figure 8: Active-Sham difference maps with cluster permutation
    active_sham: dict = {}
    for ch in ("C3", "C4"):
        active_arrays = []
        for grp in ("c3_smr", "c3_beta", "c4_smr"):
            data, freqs, times, _ = assemble_ersp_tfr(
                study, sessions=BFB_SESSIONS, channel=ch, group=grp,
            )
            if data.size > 0:
                active_arrays.append(recenter_baseline(data, times))
        sham_arr, freqs, times, _ = assemble_ersp_tfr(
            study, sessions=BFB_SESSIONS, channel=ch, group="sham",
        )

        if active_arrays and sham_arr.size > 0:
            sham_arr = recenter_baseline(sham_arr, times)
            active_tfr = np.concatenate(active_arrays, axis=0)
            diff_mean = active_tfr.mean(axis=0) - sham_arr.mean(axis=0)
            active_sham[ch] = {
                "active": active_tfr, "sham": sham_arr,
                "diff": diff_mean, "freqs": freqs, "times": times,
            }

            cluster_mask = None
            try:
                from mne.stats import permutation_cluster_test
                T_obs, clusters, cluster_p, _ = permutation_cluster_test(
                    [active_tfr, sham_arr],
                    n_permutations=1000, threshold=2.0, tail=0,
                    verbose="WARNING",
                )
                sig = np.zeros_like(T_obs, dtype=bool)
                for cl, pv in zip(clusters, cluster_p):
                    if pv < 0.05:
                        sig[cl] = True
                n_sig = sum(1 for pv in cluster_p if pv < 0.05)
                logger.info(
                    "  %s cluster permutation: %d clusters found, "
                    "%d significant (p<0.05), min p=%.4f",
                    ch, len(clusters), n_sig,
                    min(cluster_p) if len(cluster_p) else 1.0,
                )
                if sig.any():
                    cluster_mask = sig
                    active_sham[ch]["cluster_mask"] = cluster_mask
            except Exception as e:
                logger.warning("Cluster test failed for %s: %s", ch, e)

            plot_ersp_difference_map(
                diff_mean, times, freqs, channel=ch,
                cluster_mask=cluster_mask,
                save_path=str(fig_dir / f"ersp_diff_{ch}"),
            )
    logger.info("Figure 8: Difference maps saved")

    # Figure 9: Violin plots
    plot_erd_violins(scalars, save_path=str(fig_dir / "erd_violins"))
    logger.info("Figure 9: Violin plots saved")

    # Figure 10: Composite summary
    c3_data = active_sham.get("C3")
    if c3_data and freq_profiles:
        plot_composite_summary(
            ersp_active=c3_data["active"].mean(axis=0),
            ersp_sham=c3_data["sham"].mean(axis=0),
            diff_data=c3_data["diff"],
            times=c3_data["times"],
            freqs=c3_data["freqs"],
            freq_profiles=freq_profiles,
            freq_profiles_se=freq_profiles_se,
            scalars=scalars,
            cluster_mask=c3_data.get("cluster_mask"),
            save_path=str(fig_dir / "composite_summary"),
        )
        logger.info("Figure 10: Composite summary saved")

    logger.info("All figures saved to %s", fig_dir)
    return 0


# ── trial log ──────────────────────────────────────────────────────────────


def stage_trial_log(study_json: str) -> int:
    """Write pipeline_trial_log.tsv summarising trial counts per subject/session."""
    import csv
    import numpy as np
    from tools.study import Study
    from tools.ersp import load_ersp
    from tools.io import extract_reward_events, find_recording, load_raw

    study = Study.from_json(
        Path(study_json),
        analyses_root=ANALYSES_ROOT,
        derivatives_root=DERIVATIVES_ROOT,
    )

    log_path = DERIVATIVES_ROOT / f"pipeline_trial_log.tsv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "subject", "group", "session", "preprocess_mode",
        "raw_events_256", "raw_events_511", "raw_events_total",
        "clean_epochs", "rejection_rate",
        "primary_erd_C3", "primary_erd_C4",
        "n_bad_channels", "n_ica_rejected",
    ]

    rows = []
    for subj in study.included():
        sid = subj.subject_id
        for sess in BFB_SESSIONS:
            row = {
                "subject": sid,
                "group": subj.group,
                "session": sess,
                "preprocess_mode": PREPROCESS_MODE,
                "raw_events_256": "",
                "raw_events_511": "",
                "raw_events_total": "",
                "clean_epochs": "",
                "rejection_rate": "",
                "primary_erd_C3": "",
                "primary_erd_C4": "",
                "n_bad_channels": "",
                "n_ica_rejected": "",
            }

            bdf_path = find_recording(sid, "BFB", sess)
            if bdf_path is not None:
                try:
                    import mne
                    raw = load_raw(bdf_path, preload=False)
                    all_events = mne.find_events(raw, verbose="WARNING", min_duration=0)
                    n_256 = int((all_events[:, 2] == 0x0100).sum())
                    n_511 = int((all_events[:, 2] == 0x01FF).sum())
                    mask = (all_events[:, 2] & 0x0100) == 0x0100
                    n_total = int(mask.sum())
                    row["raw_events_256"] = n_256
                    row["raw_events_511"] = n_511
                    row["raw_events_total"] = n_total
                except Exception as e:
                    logger.warning("Could not read raw events for %s S%d: %s", sid, sess, e)

            epo_path = ANALYSES_ROOT / sid / "epochs" / f"{sid}_BFB_{sess}_reward-epo.fif"
            if epo_path.is_file():
                try:
                    import mne
                    epochs = mne.read_epochs(epo_path, preload=False, verbose="WARNING")
                    n_clean = len(epochs)
                    row["clean_epochs"] = n_clean
                    raw_total = row.get("raw_events_total", "")
                    if raw_total and int(raw_total) > 0:
                        row["rejection_rate"] = f"{1.0 - n_clean / int(raw_total):.3f}"
                except Exception as e:
                    logger.warning("Could not read epochs for %s S%d: %s", sid, sess, e)

            h5 = study.ersp_h5(sid, sess)
            if h5.is_file():
                try:
                    result = load_ersp(h5)
                    row["primary_erd_C3"] = f"{result.scalars.get('primary_erd_C3', float('nan')):.4f}"
                    row["primary_erd_C4"] = f"{result.scalars.get('primary_erd_C4', float('nan')):.4f}"
                    row["n_bad_channels"] = result.scalars.get("n_bad_channels", "")
                    row["n_ica_rejected"] = result.scalars.get("n_ica_rejected", "")
                except Exception as e:
                    logger.warning("Could not read ERSP H5 for %s S%d: %s", sid, sess, e)

            rows.append(row)

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Trial log written: %s (%d rows)", log_path, len(rows))
    return 0


# ── Source localization (exploratory) ───────────────────────────────────────


def stage_source(study_json: str) -> int:
    """Generate exploratory eLORETA source maps (Figures S11, S12)."""
    from tools.source import build_forward_model, load_group_epochs
    from tools.study import Study
    from tools.viz.source_maps import plot_erd_source_maps, plot_erp_source_maps

    study = Study.from_json(
        Path(study_json),
        analyses_root=ANALYSES_ROOT,
        derivatives_root=DERIVATIVES_ROOT,
    )

    fig_dir = DERIVATIVES_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading epochs for source analysis ...")
    epochs_by_group = {}
    for group_key in ("c3_beta", "c3_smr"):
        epochs_by_group[group_key] = load_group_epochs(
            study, group_key, sessions=(1, 3, 5),
        )
        if not epochs_by_group[group_key]:
            logger.error("No epochs for %s — cannot proceed", group_key)
            return 1

    sample_info = epochs_by_group["c3_beta"][0].info
    logger.info("Building forward model ...")
    fwd, src = build_forward_model(sample_info)

    logger.info("=== Figure S11: ERD source maps ===")
    plot_erd_source_maps(epochs_by_group, fwd, fig_dir)

    logger.info("=== Figure S12: P2 evoked source maps ===")
    plot_erp_source_maps(epochs_by_group, fwd, fig_dir)

    logger.info("Source figures complete — inspect before committing to manuscript")
    return 0


# ── PDF generation ─────────────────────────────────────────────────────────

def stage_pdf() -> int:
    """Rebuild manuscript and supplement PDFs."""
    from scripts.make_story_pdf import main as build_pdfs
    try:
        build_pdfs()
    except Exception as exc:
        logger.error("PDF build failed: %s", exc)
        return 1
    return 0


# ── main ───────────────────────────────────────────────────────────────────


STAGES = (
    "validate", "preprocess", "epochs", "ersp", "erp", "prs",
    "resting", "stats", "prs_stats", "figures", "trial_log", "source", "pdf", "all",
)

PER_SUBJECT_STAGES = ("validate", "preprocess", "epochs", "ersp", "erp", "resting")


def _run_subject_pipeline(
    subject_id: str,
    sessions: list[int],
    stages: tuple[str, ...],
    reward_band: str,
) -> int:
    """Run per-subject pipeline stages in order."""
    stage_fns = {
        "validate":   lambda: stage_validate(subject_id, sessions),
        "preprocess": lambda: stage_preprocess(subject_id, sessions),
        "epochs":     lambda: stage_epochs(subject_id, sessions),
        "ersp":       lambda: stage_ersp(subject_id, sessions, reward_band),
        "erp":        lambda: stage_erp(subject_id, sessions),
        "prs":        lambda: stage_prs(subject_id, sessions),
        "resting":    lambda: stage_resting(subject_id, sessions),
    }
    for name in stages:
        if name not in stage_fns:
            continue
        logger.info("=== %s: Stage %s ===", subject_id, name)
        rc = stage_fns[name]()
        if rc != 0:
            logger.error("Stage %s failed for %s (rc=%d)", name, subject_id, rc)
            return rc
    return 0


def _check_prerequisites(subject: str, stage: str) -> bool:
    """Verify prerequisite outputs exist before running a stage."""
    checks = {
        "epochs": lambda: (
            ANALYSES_ROOT / subject / "preprocessed"
            / f"{subject}_BFB_1_clean-raw.fif"
        ).is_file(),
        "ersp": lambda: (
            ANALYSES_ROOT / subject / "epochs"
            / f"{subject}_BFB_1_reward-epo.fif"
        ).is_file(),
        "erp": lambda: (
            ANALYSES_ROOT / subject / "preprocessed"
            / f"{subject}_BFB_1_clean-raw.fif"
        ).is_file(),
        "prs": lambda: (
            ANALYSES_ROOT / subject / "epochs"
            / f"{subject}_BFB_1_reward-epo.fif"
        ).is_file(),
    }
    check = checks.get(stage)
    if check and not check():
        logger.warning(
            "Prerequisite missing for %s stage %s — skipping",
            subject, stage,
        )
        return False
    return True


def main() -> int:
    p = argparse.ArgumentParser(
        description="ERSP mechanism paper pipeline (PIPELINE.md)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--stage", choices=STAGES, default="validate",
                   help="Pipeline stage to run")
    p.add_argument("--subject", default="101",
                   help="Subject ID (e.g. 101)")
    p.add_argument("--sessions", default="1,3,5,6",
                   help="Comma-separated BFB session numbers")
    p.add_argument("--reward-band", default="smr",
                   help="Reward band key for ERSP (smr or beta)")
    p.add_argument("--study-json", default=None,
                   help="Path to study metadata JSON (for stats/figures)")
    p.add_argument("--all-subjects", action="store_true",
                   help="Run on all included subjects (requires --study-json)")
    p.add_argument("--parallel", type=int, default=1, metavar="N",
                   help="Number of subjects to process in parallel (requires --all-subjects)")
    p.add_argument("--preprocess-mode", choices=("minimal", "ica"),
                   default="minimal",
                   help="Preprocessing variant: 'minimal' (dissertation-era, no ICA) "
                        "or 'ica' (full ICA+ICLabel pipeline). Default: minimal")
    args = p.parse_args()

    _set_preprocess_mode(args.preprocess_mode)
    sessions = [int(x) for x in args.sessions.split(",")]

    if args.all_subjects:
        if not args.study_json:
            logger.error("--all-subjects requires --study-json")
            return 1

        from tools.study import Study
        from tools.config import GROUP_REWARD_BAND
        study = Study.from_json(
            Path(args.study_json),
            analyses_root=ANALYSES_ROOT,
            derivatives_root=DERIVATIVES_ROOT,
        )
        logger.info(study.summary())

        subjects = study.included()
        stages = PER_SUBJECT_STAGES if args.stage == "all" else (args.stage,)

        if args.parallel > 1:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=args.parallel, verbose=10)(
                delayed(_run_subject_pipeline)(
                    subj.subject_id, sessions, stages,
                    GROUP_REWARD_BAND[subj.group],
                )
                for subj in subjects
            )
            failures = sum(1 for r in results if r != 0)
        else:
            failures = 0
            for subj in subjects:
                rc = _run_subject_pipeline(
                    subj.subject_id, sessions, stages,
                    GROUP_REWARD_BAND[subj.group],
                )
                if rc != 0:
                    failures += 1

        logger.info(
            "Batch complete: %d/%d subjects succeeded",
            len(subjects) - failures, len(subjects),
        )

        if args.stage == "all":
            logger.info("=== Running group stages (stats, figures, trial_log) ===")
            for group_stage_fn in (stage_stats, stage_figures, stage_trial_log):
                rc = group_stage_fn(args.study_json)
                if rc != 0:
                    logger.error("Group stage %s failed", group_stage_fn.__name__)
                    failures += 1

        return 1 if failures else 0

    # Single-subject mode
    stage_map = {
        "validate":   lambda: stage_validate(args.subject, sessions),
        "preprocess": lambda: stage_preprocess(args.subject, sessions),
        "epochs":     lambda: stage_epochs(args.subject, sessions),
        "ersp":       lambda: stage_ersp(args.subject, sessions, args.reward_band),
        "erp":        lambda: stage_erp(args.subject, sessions),
        "prs":        lambda: stage_prs(args.subject, sessions),
        "resting":    lambda: stage_resting(args.subject, sessions),
        "stats":      lambda: stage_stats(args.study_json) if args.study_json else _missing_study(),
        "prs_stats":  lambda: stage_prs_stats(args.study_json) if args.study_json else _missing_study(),
        "figures":    lambda: stage_figures(args.study_json) if args.study_json else _missing_study(),
        "trial_log":  lambda: stage_trial_log(args.study_json) if args.study_json else _missing_study(),
        "source":     lambda: stage_source(args.study_json) if args.study_json else _missing_study(),
        "pdf":        lambda: stage_pdf(),
    }

    if args.stage == "all":
        for name in PER_SUBJECT_STAGES:
            logger.info("=== Stage: %s ===", name)
            rc = stage_map[name]()
            if rc != 0:
                return rc
        logger.info("Per-subject pipeline complete. Run --stage stats with --study-json for group analysis.")
        return 0

    return stage_map[args.stage]()


def _missing_study() -> int:
    logger.error("--study-json is required for stats and figures stages")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
