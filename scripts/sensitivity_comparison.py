"""Compare minimal vs ICA preprocessing pipelines.

Generates a comparison table and figure showing:
- Trial counts per subject × session (minimal vs ICA)
- Group-level effect sizes (Active vs Sham ERD)
- Summary statistics

Usage:
    python scripts/sensitivity_comparison.py --study-json data/study.json
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-20s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("sensitivity")

BFB_SESSIONS = (1, 3, 5, 6)
DERIV_ROOT = Path("/path/to/your/ERSP_data/Derivatives")


def load_trial_log(mode: str) -> pd.DataFrame:
    path = DERIV_ROOT / mode / "pipeline_trial_log.tsv"
    if not path.is_file():
        logger.warning("Trial log not found: %s", path)
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def load_erd_scalars(mode: str, study) -> pd.DataFrame:
    from tools.ersp import load_ersp
    rows = []
    deriv = DERIV_ROOT / mode
    for subj in study.included():
        for sess in BFB_SESSIONS:
            h5 = deriv / subj.subject_id / "ersp" / f"{subj.subject_id}_BFB_{sess}_ersp.h5"
            if not h5.is_file():
                continue
            result = load_ersp(h5)
            val = result.scalars.get("primary_erd_C3", float("nan"))
            rows.append({
                "subject": subj.subject_id,
                "group": subj.group,
                "session": sess,
                "erd": val,
                "mode": mode,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Compare minimal vs ICA pipelines")
    parser.add_argument("--study-json", required=True)
    args = parser.parse_args()

    from tools.study import Study
    study = Study.from_json(
        Path(args.study_json),
        analyses_root=Path("/path/to/your/ERSP_data/Analyses/minimal"),
        derivatives_root=DERIV_ROOT / "minimal",
    )

    fig_dir = DERIV_ROOT / "minimal" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Trial counts
    min_log = load_trial_log("minimal")
    ica_log = load_trial_log("ica")

    if min_log.empty or ica_log.empty:
        logger.error("Need both minimal and ICA trial logs to compare")
        return 1

    min_log = min_log.rename(columns={"clean_epochs": "trials_minimal"})
    ica_log = ica_log.rename(columns={"clean_epochs": "trials_ica"})

    merged = min_log[["subject", "session", "trials_minimal"]].merge(
        ica_log[["subject", "session", "trials_ica"]],
        on=["subject", "session"], how="outer",
    )

    logger.info("=== Trial Count Comparison ===")
    logger.info("  Minimal: median=%.0f, mean=%.0f",
                merged["trials_minimal"].median(), merged["trials_minimal"].mean())
    logger.info("  ICA:     median=%.0f, mean=%.0f",
                merged["trials_ica"].median(), merged["trials_ica"].mean())
    logger.info("  Minimal retains %.1f%% more trials on average",
                100 * (merged["trials_minimal"].mean() / merged["trials_ica"].mean() - 1))

    # ERD effect sizes
    min_erd = load_erd_scalars("minimal", study)
    ica_erd = load_erd_scalars("ica", study)

    if min_erd.empty or ica_erd.empty:
        logger.error("Need both minimal and ICA ERD data")
        return 1

    from tools.stats import planned_contrast
    from tools.config import STATS as STATS_CFG

    logger.info("\n=== Effect Size Comparison (Active pooled vs Sham) ===")
    for mode, df in [("minimal", min_erd), ("ica", ica_erd)]:
        group_means = {}
        for gkey in ("c3_smr", "c3_beta", "c4_smr", "sham"):
            vals = df[df["group"] == gkey].groupby("subject")["erd"].mean().values
            if len(vals) > 0:
                group_means[gkey] = vals

        if "sham" in group_means:
            active = np.concatenate([group_means.get(k, [])
                                     for k in ("c3_smr", "c3_beta", "c4_smr")])
            c = planned_contrast(active, group_means["sham"],
                                 f"{mode}: Active vs Sham", STATS_CFG)
            logger.info("  %s: d=%.3f, p=%.4f, n_active=%d, n_sham=%d",
                        mode, c.cohens_d, c.p_value, len(active), len(group_means["sham"]))

    # Figure: trial count comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(merged["trials_minimal"], merged["trials_ica"],
               s=15, alpha=0.6, edgecolors="none")
    lim = max(merged["trials_minimal"].max(), merged["trials_ica"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Trials (minimal)")
    ax.set_ylabel("Trials (ICA)")
    ax.set_title("Trial Retention: Minimal vs ICA")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    ax = axes[1]
    for mode, color, label in [("minimal", "#2196F3", "Minimal"), ("ica", "#FF5722", "ICA")]:
        df = min_erd if mode == "minimal" else ica_erd
        for gkey in ("c3_smr", "c3_beta", "c4_smr", "sham"):
            vals = df[df["group"] == gkey].groupby("subject")["erd"].mean().values
            if len(vals) == 0:
                continue
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(vals))
            x_base = ["C3 SMR", "C3 Beta", "C4 SMR", "Sham"].index(
                {"c3_smr": "C3 SMR", "c3_beta": "C3 Beta",
                 "c4_smr": "C4 SMR", "sham": "Sham"}[gkey])
            offset = -0.2 if mode == "minimal" else 0.2
            ax.scatter(x_base + offset + jitter * 0.3, vals,
                       s=20, alpha=0.6, color=color, edgecolors="none",
                       label=label if gkey == "c3_smr" else "")

    ax.set_xticks(range(4))
    ax.set_xticklabels(["C3 SMR", "C3 Beta", "C4 SMR", "Sham"])
    ax.set_ylabel("ERD (dB)")
    ax.set_title("ERD by Group: Minimal vs ICA")
    ax.axhline(0, color="0.5", ls=":", lw=0.6)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = fig_dir / "sensitivity_comparison"
    fig.savefig(f"{out_path}.png", dpi=150)
    fig.savefig(f"{out_path}.pdf")
    plt.close(fig)
    logger.info("Comparison figure saved to %s", out_path)

    # Save comparison table
    table_path = DERIV_ROOT / "minimal" / "stats" / "sensitivity_comparison.txt"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w") as f:
        f.write("Sensitivity Analysis: Minimal vs ICA Preprocessing\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Trial counts (minimal): median={merged['trials_minimal'].median():.0f}, "
                f"mean={merged['trials_minimal'].mean():.0f}\n")
        f.write(f"Trial counts (ICA):     median={merged['trials_ica'].median():.0f}, "
                f"mean={merged['trials_ica'].mean():.0f}\n")
        f.write(f"Minimal retains {100 * (merged['trials_minimal'].mean() / merged['trials_ica'].mean() - 1):.1f}% "
                f"more trials on average\n\n")
        f.write("Per-session comparison:\n")
        f.write(merged.to_string(index=False))
        f.write("\n")
    logger.info("Comparison table saved to %s", table_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
