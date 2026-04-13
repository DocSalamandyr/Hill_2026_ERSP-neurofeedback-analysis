"""Preemptive reviewer analyses: peak frequency, ERD-P2 correlation, reward rate.

Reads existing derived data (ERSP h5, ERP h5, trial log) and outputs
three targeted analyses to forestall predictable reviewer objections.

Usage:
    python scripts/preemptive_stats.py --study-json data/study.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-20s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preemptive")

BFB_SESSIONS = (1, 3, 5, 6)
DERIV_ROOT = Path("/path/to/your/ERSP_data/Derivatives")


def peak_frequency_analysis(study, out_dir: Path) -> str:
    """Per-subject peak ERD frequency at C3 for SMR vs Beta groups."""
    from tools.ersp import load_ersp
    from tools.group import recenter_baseline
    from tools.stats import planned_contrast
    from tools.config import STATS as STATS_CFG

    ERD_WINDOW = (0.2, 0.8)
    SEARCH_RANGE = (10.0, 20.0)

    peak_freqs: Dict[str, List[float]] = {"c3_smr": [], "c3_beta": []}

    for group_key in ("c3_smr", "c3_beta"):
        for subj in study.by_group(group_key):
            sess_profiles = []
            ref_freqs = None
            for sess in BFB_SESSIONS:
                h5 = study.ersp_h5(subj.subject_id, sess)
                if not h5.is_file():
                    continue
                result = load_ersp(h5)
                if "C3" not in result.channel_names:
                    continue
                ci = result.channel_names.index("C3")
                tfr = recenter_baseline(result.ersp[ci], result.times)
                t_mask = (result.times >= ERD_WINDOW[0]) & (result.times <= ERD_WINDOW[1])
                profile = tfr[:, t_mask].mean(axis=1)
                sess_profiles.append(profile)
                if ref_freqs is None:
                    ref_freqs = result.freqs

            if not sess_profiles or ref_freqs is None:
                continue

            avg_profile = np.mean(sess_profiles, axis=0)
            search_mask = (ref_freqs >= SEARCH_RANGE[0]) & (ref_freqs <= SEARCH_RANGE[1])
            search_profile = avg_profile[search_mask]
            search_freqs = ref_freqs[search_mask]
            peak_idx = np.argmin(search_profile)
            peak_freq = float(search_freqs[peak_idx])
            peak_freqs[group_key].append(peak_freq)
            logger.info("  %s %s: peak ERD at %.1f Hz (%.2f dB)",
                        group_key, subj.subject_id, peak_freq,
                        search_profile[peak_idx])

    smr_peaks = np.array(peak_freqs["c3_smr"])
    beta_peaks = np.array(peak_freqs["c3_beta"])

    contrast = planned_contrast(smr_peaks, beta_peaks,
                                "Peak freq: C3 SMR vs C3 Beta", STATS_CFG)

    lines = [
        "Peak ERD Frequency Analysis (C3, 200-800 ms, 10-20 Hz search)",
        "=" * 60,
        "",
        f"C3 SMR (n={len(smr_peaks)}): mean={smr_peaks.mean():.1f} ± {smr_peaks.std():.1f} Hz",
        f"C3 Beta (n={len(beta_peaks)}): mean={beta_peaks.mean():.1f} ± {beta_peaks.std():.1f} Hz",
        "",
        f"t = {contrast.statistic:.3f}, p = {contrast.p_value:.4f}",
        f"Cohen's d = {contrast.cohens_d:.3f}",
        f"BF01 = {contrast.bf01:.3f}",
        "",
        "Per-subject peak frequencies (Hz):",
        f"  C3 SMR:  {', '.join(f'{f:.1f}' for f in smr_peaks)}",
        f"  C3 Beta: {', '.join(f'{f:.1f}' for f in beta_peaks)}",
        "",
    ]
    report = "\n".join(lines)
    path = out_dir / "peak_frequency.txt"
    path.write_text(report)
    logger.info("Peak frequency analysis saved to %s", path)
    return report


def erd_p2_correlation(study, out_dir: Path) -> str:
    """Correlation between ERD and P2 amplitude at C3 across subjects."""
    from tools.ersp import load_ersp
    from tools.erp import load_erp

    erd_by_subj: Dict[str, List[float]] = {}
    p2_by_subj: Dict[str, List[float]] = {}

    for subj in study.included():
        for sess in BFB_SESSIONS:
            ersp_h5 = study.ersp_h5(subj.subject_id, sess)
            erp_h5 = (study.derivatives_root / subj.subject_id / "erp"
                       / f"{subj.subject_id}_BFB_{sess}_erp.h5")
            if not ersp_h5.is_file() or not erp_h5.is_file():
                continue

            ersp_result = load_ersp(ersp_h5)
            erd_val = ersp_result.scalars.get("primary_erd_C3")
            if erd_val is None:
                continue

            erp_result = load_erp(erp_h5)
            p2_val = None
            for comp in erp_result.components:
                if comp.component == "P2" and comp.channel == "C3":
                    p2_val = comp.mean_amplitude_uv
                    break
            if p2_val is None:
                continue

            erd_by_subj.setdefault(subj.subject_id, []).append(erd_val)
            p2_by_subj.setdefault(subj.subject_id, []).append(p2_val)

    subjects_both = sorted(set(erd_by_subj) & set(p2_by_subj))
    erd_means = np.array([np.mean(erd_by_subj[s]) for s in subjects_both])
    p2_means = np.array([np.mean(p2_by_subj[s]) for s in subjects_both])

    r_all, p_all = sp_stats.pearsonr(erd_means, p2_means)

    active_ids = {s.subject_id for s in study.active_subjects()}
    active_mask = np.array([s in active_ids for s in subjects_both])
    r_active, p_active = sp_stats.pearsonr(erd_means[active_mask],
                                            p2_means[active_mask])

    lines = [
        "ERD-P2 Correlation at C3 (session-averaged per subject)",
        "=" * 60,
        "",
        f"All subjects (n={len(subjects_both)}):",
        f"  r = {r_all:.3f}, p = {p_all:.4f}",
        "",
        f"Active subjects only (n={active_mask.sum()}):",
        f"  r = {r_active:.3f}, p = {p_active:.4f}",
        "",
    ]

    if abs(r_active) < 0.3 and p_active > 0.05:
        lines.append("Interpretation: ERD and P2 are not significantly correlated,")
        lines.append("supporting their characterization as dissociable mechanisms.")
    else:
        lines.append(f"Interpretation: r = {r_active:.3f} — report and discuss.")

    lines.append("")
    report = "\n".join(lines)
    path = out_dir / "erd_p2_correlation.txt"
    path.write_text(report)
    logger.info("ERD-P2 correlation saved to %s", path)
    return report


def reward_rate_equivalence(out_dir: Path) -> str:
    """Test whether reward rates differed across groups."""
    log_path = DERIV_ROOT / "minimal" / "pipeline_trial_log.tsv"
    if not log_path.is_file():
        logger.error("Trial log not found: %s", log_path)
        return ""

    df = pd.read_csv(log_path, sep="\t")
    df = df[df["session"].isin(BFB_SESSIONS)].copy()

    subj_means = (df.groupby(["subject", "group"])["raw_events_total"]
                  .mean().reset_index())

    groups = ["c3_smr", "c3_beta", "c4_smr", "sham"]
    group_data = {g: subj_means[subj_means["group"] == g]["raw_events_total"].values
                  for g in groups}

    all_vals = subj_means["raw_events_total"]
    grand_mean = all_vals.mean()
    grand_sd = all_vals.std()

    f_stat, p_val = sp_stats.f_oneway(*[group_data[g] for g in groups
                                         if len(group_data[g]) > 0])

    lines = [
        "Reward Rate Equivalence Across Groups",
        "=" * 60,
        "",
        f"Grand mean: {grand_mean:.0f} ± {grand_sd:.0f} events/session",
        "",
        "Per-group means (events/session):",
    ]
    for g in groups:
        vals = group_data[g]
        if len(vals) > 0:
            lines.append(f"  {g:10s}: {vals.mean():.0f} ± {vals.std():.0f} (n={len(vals)})")

    lines.extend([
        "",
        f"One-way ANOVA: F(3,{sum(len(group_data[g]) for g in groups) - 4}) = {f_stat:.3f}, p = {p_val:.4f}",
        "",
    ])

    if p_val > 0.05:
        lines.append("Reward rates did not differ significantly across groups.")
    else:
        lines.append(f"NOTE: Significant group difference (p = {p_val:.4f}) — investigate.")

    lines.append("")
    report = "\n".join(lines)
    path = out_dir / "reward_rate.txt"
    path.write_text(report)
    logger.info("Reward rate analysis saved to %s", path)
    return report


def main():
    parser = argparse.ArgumentParser(description="Preemptive reviewer analyses")
    parser.add_argument("--study-json", required=True)
    args = parser.parse_args()

    from tools.study import Study
    study = Study.from_json(
        Path(args.study_json),
        analyses_root=Path("/path/to/your/ERSP_data/Analyses/minimal"),
        derivatives_root=DERIV_ROOT / "minimal",
    )

    out_dir = DERIV_ROOT / "minimal" / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== 1. Peak Frequency Analysis ===")
    print(peak_frequency_analysis(study, out_dir))

    logger.info("\n=== 2. ERD-P2 Correlation ===")
    print(erd_p2_correlation(study, out_dir))

    logger.info("\n=== 3. Reward Rate Equivalence ===")
    print(reward_rate_equivalence(out_dir))

    logger.info("All preemptive analyses complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
