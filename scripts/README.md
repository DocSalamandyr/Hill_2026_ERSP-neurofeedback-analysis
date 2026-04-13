# Analysis Scripts

| File | Purpose |
|------|---------|
| [pipeline.py](pipeline.py) | Master entry point — `--stage {validate,preprocess,epochs,ersp,erp,resting,stats,figures,pdf}` |
| [make_story_pdf.py](make_story_pdf.py) | Assemble manuscript + supplement PDFs from Markdown drafts |
| [sensitivity_comparison.py](sensitivity_comparison.py) | Compare minimal vs ICA preprocessing pipelines |
| [requirements.txt](requirements.txt) | Python dependencies |

## Running

```bash
# Activate your virtual environment
source .venv/bin/activate

# Set data paths (see tools/config.py for details)
export ERSP_DATA_ROOT=/path/to/raw/BDF/files
export ERSP_ANALYSES_ROOT=/path/to/intermediate/analyses
export ERSP_DERIVATIVES_ROOT=/path/to/derivatives

# Single subject
python scripts/pipeline.py --stage validate --subject 105
python scripts/pipeline.py --stage preprocess --subject 105

# Batch (all subjects)
python scripts/pipeline.py --stage preprocess --all-subjects --parallel 4

# Figures + stats
python scripts/pipeline.py --stage figures --study-json data/study.json
```

Configuration lives in `tools/config.py`. Override paths via environment
variables: `ERSP_DATA_ROOT`, `ERSP_ANALYSES_ROOT`, `ERSP_DERIVATIVES_ROOT`,
`ERSP_RELATED_ROOT`.
