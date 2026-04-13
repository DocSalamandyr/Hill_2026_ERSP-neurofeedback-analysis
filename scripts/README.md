# Analysis Scripts

## Pipeline

| File | Purpose |
|------|---------|
| [pipeline.py](pipeline.py) | Master entry point — `--stage {validate,preprocess,epochs,ersp,erp,resting,stats,figures}`. Figures stage generates 19 publication-quality PNGs including ERSP heatmaps, session grids, difference maps (cluster permutation), frequency crossover, topographic ERD, ERP waveforms, violins, and composite summary. |
| [create_study.py](create_study.py) | Bootstrap `data/study.json` with subject metadata and group assignments |
| [requirements.txt](requirements.txt) | Python dependencies (install into `/path/to/your/ERSP_data/.venv`) |

## Workspace layout

```
/path/to/your/ERSP_data/
  Data/           raw BDFs + hfinf2 sidecars (40 analyzable subjects, read-only)
  Analyses/       intermediate working files (preprocessed .fif, epochs)
  Derivatives/    final pipeline output (ERSP h5, ERP h5, resting PSD, 19 publication figures, stats)
  Related/        curated reference materials (bad channels, PACDEL CSVs, montages, …)
  .venv/          Python virtual environment
```

Config in `tools/config.py` — override paths via `ERSP_DATA_ROOT`, `ERSP_ANALYSES_ROOT`, `ERSP_DERIVATIVES_ROOT`, `ERSP_RELATED_ROOT` env vars.

## Running

```bash
source /path/to/your/ERSP_data/.venv/bin/activate
cd /path/to/your/ERSP_project

# Single subject
python scripts/pipeline.py --stage validate --subject 105
python scripts/pipeline.py --stage preprocess --subject 105
python scripts/pipeline.py --stage epochs --subject 105
python scripts/pipeline.py --stage ersp --subject 105
python scripts/pipeline.py --stage resting --subject 105

# Batch (all 40 analyzable subjects)
python scripts/pipeline.py --stage preprocess --all-subjects --parallel 4
```

## Original MATLAB/EEGLAB scripts (recovered, reference only)

| Script | Purpose |
|--------|---------|
| `extract_epochs_LANT.m` | LANT epoch extraction (sLANT conditions) |
| `extract_epochs_twofiles_newio_LANT.m` | Same, two-file format |
| `ERSPITCgen.m` | ERSP/ITC plotting via EEGLAB STUDY |
| `eeglabhist-83109.m` | EEGLAB history (Aug 2009 session) |

MATLAB EDF conversion scripts from UCLA are archived in `Related/edf_process/`.

## Canonical references

- [../PIPELINE.md](../PIPELINE.md) — preprocessing and analysis methods
- [../PAPER.md](../PAPER.md) — paper plan
- [../tools/config.py](../tools/config.py) — all tuneable parameters
