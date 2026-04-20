#!/usr/bin/env python3
"""Session 1 P2 baseline ANOVA at C3.

One-way between-group ANOVA on Session 1 P2 mean amplitude at C3,
testing whether the P2 group effect is present from the first session.
This addresses V6 Critical #1.

Output: stats/p2_session1_baseline.txt
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import h5py
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
DERIVATIVES_ROOT = Path("/path/to/your/ERSP_data/Derivatives/minimal")
STUDY_JSON = PROJECT_ROOT / "data" / "study.json"
STATS_DIR = DERIVATIVES_ROOT / "stats"

with open(STUDY_JSON) as f:
    subjects = json.load(f)

included = [s for s in subjects if not s.get("excluded", False)]

rows: list[dict] = []
for subj in included:
    sid = subj["subject_id"]
    group = subj["group"]
    h5_path = DERIVATIVES_ROOT / sid / "erp" / f"{sid}_BFB_1_erp.h5"
    if not h5_path.is_file():
        print(f"  skip {sid}: no Session 1 ERP", file=sys.stderr)
        continue
    with h5py.File(h5_path, "r") as hf:
        i = 0
        while f"component/{i}" in hf:
            grp = hf[f"component/{i}"]
            comp = str(grp.attrs["component"])
            ch = str(grp.attrs["channel"])
            mean_uv = float(grp.attrs["mean_amplitude_uv"])
            if comp == "P2" and ch == "C3":
                rows.append({"subject": sid, "group": group, "mean_uv": mean_uv})
            i += 1

if not rows:
    print("ERROR: no P2 C3 data found for Session 1", file=sys.stderr)
    sys.exit(1)

import pingouin as pg
import pandas as pd

df = pd.DataFrame(rows)
print(f"Session 1 P2 @ C3: {len(df)} subjects across {df['group'].nunique()} groups")
print(f"Groups: {df['group'].value_counts().to_dict()}")

aov = pg.anova(data=df, dv="mean_uv", between="group", detailed=True)
print("\nOne-way ANOVA:")
print(aov.to_string())

F_val = aov.loc[0, "F"]
p_val = aov.loc[0, [c for c in aov.columns if "p" in c.lower() and "np" not in c.lower()][0]]
np2 = aov.loc[0, "np2"]
df1 = int(aov.loc[0, "DF"] if "DF" in aov.columns else aov.loc[0, [c for c in aov.columns if "df" in c.lower() or "ddof" in c.lower()][0]])
df2 = int(aov.loc[1, "DF"] if "DF" in aov.columns else aov.loc[1, [c for c in aov.columns if "df" in c.lower() or "ddof" in c.lower()][0]])

print(f"\nF({df1},{df2}) = {F_val:.3f}, p = {p_val:.4f}, ηp² = {np2:.4f}")

group_stats = df.groupby("group")["mean_uv"].agg(["mean", "std", "count"])
print("\nPer-group descriptives (Session 1 P2 mean_uv at C3):")
print(group_stats.to_string())

STATS_DIR.mkdir(parents=True, exist_ok=True)
out_path = STATS_DIR / "p2_session1_baseline.txt"
with open(out_path, "w") as f:
    f.write("P2 Session 1 Baseline ANOVA at C3\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"N = {len(df)} subjects, {df['group'].nunique()} groups\n")
    f.write(f"Groups: {df['group'].value_counts().to_dict()}\n\n")
    f.write(f"One-way ANOVA: F({df1},{df2}) = {F_val:.3f}, p = {p_val:.4f}, ηp² = {np2:.4f}\n\n")
    f.write("Per-group descriptives:\n")
    f.write(group_stats.to_string())
    f.write("\n")

print(f"\nSaved to {out_path}")
