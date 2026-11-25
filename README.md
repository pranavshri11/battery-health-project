# Battery Health Prediction

Streamlit app and reproducible machine-learning workflow for capacity-informed battery State of Health (SOH) estimation. The local dataset has 1,415 cycle records and 34 battery IDs.

## Run the app

Tested locally with Windows and Python 3.14.6. Dependencies are pinned to the versions used for the saved model. A clean installation and remote CI are separate checks from the existing working environment.

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m streamlit run app.py
```

For an existing environment, use the last two commands. On Linux/macOS, use `.venv/bin/python` instead. The UI provides four measurements, a dark result panel with a percentage gauge, a real sample, reset, training-range messages, CSV export, and expandable model/validation details. Editing measurements hides the old result until it is recalculated.

## Train and check

From the repository root:

```powershell
python -m src.data_preprocessing
python -m src.model_training
python -m src.model_evaluation
python -m src.eda
python -m unittest discover -s tests -v
```

`src.feature_engineering` optionally exports the raw splits. No manual preprocessing step is required before training. `src.hyperparameter_tuning` runs the same complete training workflow. Direct scripts such as `python C:/path/to/project/src/data_diagnostics.py` also work from other directories. Each script supports `--help`.

The CSV validator rejects missing fields, empty IDs, duplicate battery/cycle keys, non-finite values, non-integer counts, nonpositive voltage/capacity/SOH, and physically impossible temperatures. No source measurements are dropped or clipped.

## Validation results

Selection uses minimum grouped-CV MAE on early development cycles only. The scaler is fitted inside each fold. Seven entire batteries are held out. Later development cycles are also withheld from selection and evaluated with a separately fitted temporal model.

| RF validation | Records | Batteries | MAE, percentage points | RMSE, SOH ratio | R² |
|---|---:|---:|---:|---:|---:|
| Unseen batteries | 277 | 7 | 6.05 | 0.11221 | 0.15929 |
| Later cycles of development batteries | 233 | 20 | 3.00 | 0.04120 | 0.88015 |

Seed 42; 27 development batteries; 905 early-cycle selection records and 233 later-cycle records; five grouped CV folds; 12 RF parameter candidates. Tuned RF wins CV and the app serves the model fitted on all 1,138 development records. Holdout labels do not determine the winner. This single-split result shows limited generalization to new batteries.

The dataset labels satisfy `soh = capacity / first_observed_capacity_per_battery`. Because measured capacity is also an input, this is capacity-informed estimation, not independent sensor-only forecasting. The 124 SOH labels above 1 are retained: first observed capacity is not necessarily rated capacity. RUL is `last_observed_cycle - cycle`; no physical failure-life model or confidence interval is claimed.

The project identifies the CSV as NASA battery ageing data, but the raw NASA files, extraction code, and licensing evidence are not included. Independent provenance verification remains open. Full findings and caveats: [code/data/model audit](docs/CODE_DATA_MODEL_AUDIT.md).

## Files

- `app.py`, `assets/app.css`, `.streamlit/config.toml`: primary Streamlit interface.
- `src/`: validation, splitting, training, evaluation, EDA, and inference.
- `models/random_forest_pipeline.joblib`: the app's single scaler/model pipeline.
- `models/model_metadata.json`: feature order, training ranges, package versions, checksums, split membership, and scores.
- `models/temporal_rf_pipeline.joblib`: separate model for later-cycle evaluation.
- `models/selected_model_pipeline.joblib`: CV comparison winner.
- `output/validated/`: new split manifest, CV/tuning tables, per-battery predictions, evaluation report, and generated charts.
- `tests/`: data, inference, leakage-boundary, serialization, and UI regression checks.

Original `.pkl` model/scaler files, `output/model_comparison.csv`, reports, and presentations remain historical evidence. They are not the new app model or current validation results. The separate nested `sites/battery-lab` repository still serves its legacy exported model.

## Deployment and troubleshooting

Existing Streamlit URL: https://battery-health-project-dbpxtfwvq9apwhzjkqxga4.streamlit.app/

Deploy root `app.py` with its `src`, `assets`, theme, pinned requirements, dataset, RF pipeline, and metadata. Configure a Python version compatible with these pinned packages. The pipeline checksum must match metadata. Publish matching artifact and metadata files together; the app reloads resources when their file modification times change. A missing, mismatched, or incompatible artifact gives a readable error rather than a fabricated result.

GitHub Actions runs tests and metric reproduction on push/PR. Its remote result and Streamlit rebuilding must be verified after publication; local checks do not establish deployment success. To reproduce older report results, use the preserved legacy artifacts and the corresponding Git revision rather than substituting the new holdout scores.
