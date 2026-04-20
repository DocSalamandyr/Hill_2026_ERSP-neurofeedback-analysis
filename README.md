# ERSP Neurofeedback Analysis

Analysis code for:

> Hill, A. (2026). Frequency-Specific Operant Learning in Neurofeedback Reveals
> Distinct Cortical Mechanisms: Evidence from Double-Blind ERSP and ERP
> Dissociations. *[Preprint]*.

## Repository structure

```
scripts/
  pipeline.py              Main analysis pipeline (preprocessing, stats, figures)
  make_story_pdf.py        Assemble manuscript + supplement PDFs
  sensitivity_comparison.py  ICA vs minimal preprocessing comparison
  session1_p2_anova.py     Session 1 P2 baseline ANOVA (§S3.9)
  session1_erd_anova.py    Session 1 ERD baseline ANOVA (§S3.10)
  requirements.txt         Python dependencies
tools/                     Analysis library
  config.py                Paths, frequency bands, statistical parameters
  preprocess.py            EEG preprocessing (filtering, artifact rejection)
  epochs.py                Epoch extraction around reward events
  ersp.py                  ERSP computation (Morlet wavelets)
  erp.py                   ERP extraction and component measurement
  resting.py               Resting-state PSD computation
  stats.py                 Statistical tests (LME, permutation, Bayes)
  group.py                 Group-level aggregation
  study.py                 Study metadata and file path management
  prs.py                   Post-reinforcement synchronization analysis
  source.py                Source localization (eLORETA) utilities
  io.py                    HDF5 read/write
  viz/                     All figure-generation code
data/
  study.json               Group assignments and session metadata (no PII)
derivatives/               Derived data (~215 MB, sufficient for full reproduction)
  101/                     Per-subject folders (101–140)
    ersp/                  ERSP matrices (h5) — one per session
    erp/                   ERP averages (h5) — one per session
    resting/               Resting-state PSD (h5) — EC/EO pre/post per session
  stats/                   Group-level statistical outputs
  pipeline_trial_log.tsv   Trial counts per subject × session
```

## Setup

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r scripts/requirements.txt
```

### 2. Configure data paths

The pipeline locates raw and derived data via environment variables.
Set these before running:

```bash
export ERSP_DATA_ROOT=/path/to/raw/BDF/files
export ERSP_ANALYSES_ROOT=/path/to/intermediate/analyses
export ERSP_DERIVATIVES_ROOT=/path/to/derivatives
```

See `tools/config.py` for defaults and additional parameters.

### 3. Run the pipeline

```bash
# Full pipeline (all subjects, minimal preprocessing mode):
python scripts/pipeline.py --preprocess-mode minimal --stage all \
  --all-subjects --study-json data/study.json

# ICA preprocessing (sensitivity analysis — all 40 subjects):
python scripts/pipeline.py --preprocess-mode ica --stage all \
  --all-subjects --study-json data/study.json

# Compare minimal vs ICA effect sizes and trial retention:
python scripts/sensitivity_comparison.py --study-json data/study.json

# Figures only:
python scripts/pipeline.py --stage figures --study-json data/study.json

# Rebuild manuscript PDFs:
python scripts/pipeline.py --stage pdf --study-json data/study.json
```

Primary results use the minimal pipeline (high-pass filter + statistical
artifact rejection). The ICA pipeline (extended Infomax + ICLabel) was run
on all 40 subjects as a pre-registered sensitivity check. Both pipelines
converge on the same findings; the minimal pipeline preserves slightly more
ERD signal (d = −1.23 vs −1.03 for Active vs Sham), consistent with
Delorme (2023). See manuscript §3.5 and Supplementary S6 for details.

## Data availability

Derived data (ERSP matrices, ERP averages, resting-state PSD, and
statistical outputs; ~215 MB) are included in the `derivatives/`
directory of this repository. These files are sufficient to reproduce
all statistics and figures in the paper without access to raw EEG.

Raw EEG recordings (BioSemi BDF, ~36 GB) are available from the
corresponding author subject to a data use agreement. The data were
collected at UCLA in 2010–2011 under an IRB protocol that did not
include provisions for unrestricted public sharing.

## Citation

```bibtex
@article{hill2026ersp,
  author  = {Hill, Andrew},
  title   = {Frequency-Specific Operant Learning in Neurofeedback Reveals
             Distinct Cortical Mechanisms: Evidence from Double-Blind
             ERSP and ERP Dissociations},
  year    = {2026},
  note    = {Preprint},
  url     = {https://github.com/DocSalamandyr/Hill_2026_ERSP-neurofeedback-analysis}
}
```

## License

MIT
