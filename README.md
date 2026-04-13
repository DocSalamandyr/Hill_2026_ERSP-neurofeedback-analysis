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
  io.py                    HDF5 read/write
  viz/                     All figure-generation code
data/
  study.json               Group assignments and session metadata (no PII)
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

# Figures only:
python scripts/pipeline.py --stage figures --study-json data/study.json

# Rebuild manuscript PDFs:
python scripts/pipeline.py --stage pdf --study-json data/study.json
```

## Data availability

Raw EEG data (BioSemi BDF files) are available from the author upon
reasonable request. The data were collected at the University of California,
Los Angeles under IRB approval.

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
