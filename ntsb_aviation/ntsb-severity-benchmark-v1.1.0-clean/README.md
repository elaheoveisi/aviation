# NTSB Severity Benchmark

Clean, reproducible code and final saved results for the manuscript:

**Temporal Validation and Predictor Leakage in Aviation Accident Severity Prediction: A Benchmark Study Using NTSB Structured Event Data**

## Repository contents

```text
config/      Final model and feature-registry configuration
scripts/     End-to-end analysis and figure-generation scripts
src/         Reusable Python package
results/     Final numerical results used in the manuscript
figures/     Final manuscript figures and key diagnostics
tests/       Automated reproducibility tests
data/        Instructions for obtaining and placing NTSB source data
```

The repository intentionally excludes raw NTSB files, record-level derived datasets, duplicate outputs, development notes, obsolete audits, and intermediate debugging artifacts.

## Study design

- 30,087 unique NTSB events from 2008–2025
- Severe outcome: at least one fatal or serious injury
- Main early-enriched registry: 73 predictors
- Strict-initial sensitivity registry: 52 predictors
- Models: logistic regression, random forest, and XGBoost
- Random 80/20 and future-year temporal validation
- Training-only preprocessing and oversampling
- Prevalence correction after oversampling
- Development-only threshold selection and fixed threshold transfer
- Bootstrap confidence intervals, subgroup analysis, sensitivity analyses, and SHAP

## Main findings

For the main XGBoost model:

| Validation | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|
| Random 80/20 | 0.821 | 0.690 | 0.154 |
| Temporal 2020–2025 | 0.761 | 0.563 | 0.170 |

At the fixed safety threshold of 0.257, FNR increased from 0.185 under random validation to 0.248 under temporal validation.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
PYTHONPATH=src pytest -q
```

## Run the workflow

Place the required NTSB files in `data/raw/` as described in `data/README.md`, then run:

```bash
PYTHONPATH=src python scripts/run_all.py --data-dir data/raw
```

The workflow regenerates an `outputs/` directory locally. That generated directory is not committed because the final, manuscript-relevant outputs are already curated under `results/` and `figures/`.

## Saved results

- `results/main/`: final early-enriched benchmark, threshold, leakage, rolling, subgroup, SHAP, and sensitivity outputs
- `results/strict_initial/`: key strict-initial sensitivity results
- `results/manuscript_tables/`: final tables used in the manuscript
- `results/audit/`: only the feature, missingness, and split audits needed for transparent reproduction
- `figures/`: final manuscript figures and the threshold trade-off diagnostic

## Data availability

Raw data are publicly available from the NTSB Aviation Accident Database but are not redistributed here. See `data/README.md`.

## Citation and archival release

Before journal submission, update `CITATION.cff` and `CODE_AVAILABILITY.md` with the final GitHub URL. Create a GitHub release and archive it with Zenodo to obtain a DOI.
