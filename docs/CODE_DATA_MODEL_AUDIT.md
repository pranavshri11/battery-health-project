# Code, dataset, and model audit — 25 November 2026

The primary application is `app.py`. The original aggregated CSV and legacy `.pkl` artifacts are preserved. Existing presentations and reports retain their historical results; they have not been silently rewritten to match this audit.

## Verified dataset

- 1,415 rows, 34 battery IDs, seven columns; no missing/non-finite measurements or duplicate battery/cycle keys.
- SHA-256: `96bd7cdc54ce19ae36f4b1cb0027438783abb26be835ff92069cfb6fadd2c251`. CSV checksums normalize LF/CRLF so Windows checkout conversion does not produce a false mismatch. Binary artifact checksums remain byte-exact.
- SOH equals `capacity / first_observed_capacity_for_battery`, with maximum reconstruction error below `5e-16`. Capacity is therefore related directly to the target. This task is capacity-informed estimation, not independent sensor-only forecasting.
- RUL equals `last_observed_cycle - cycle`. It is the recorded endpoint distance, not an independently verified physical end-of-life label. No RUL model was added.
- 124 SOH values exceed 1.0. They are retained because the reference is the first observation, not necessarily rated capacity. There is no evidence to justify deleting or clipping these measurements.
- The repository contains an aggregated CSV identified by the project as NASA battery ageing data. Original raw source files, extraction code, and licensing evidence still need to be supplied to verify provenance independently.

## Problems corrected

| Previous behavior | Correction |
|---|---|
| Random row split mixed cycles of the same batteries between training and test | Complete held-out battery groups, plus a separate chronological development check |
| Data scaled before CV | StandardScaler and estimator saved in one Pipeline and fitted within each CV fold |
| Winner chosen using test scores | Selection uses grouped CV MAE on early development cycles only |
| Evaluation depended on untracked processed CSVs | Training starts directly from validated CSV and records exact split membership |
| Independent model/scaler files could become mismatched | App loads one pipeline, verifies feature order and artifact hash against metadata |
| Scripts depended on current directory and blocked on plot windows | Paths resolve from the source location, CLI entry points use main guards, plots use Agg and close after saving |
| Dataset inspection accepted invalid values silently | Reusable validator rejects missing, duplicate, non-finite, impossible, and malformed records |
| Dependencies were unbounded | Exact versions recorded from the tested environment; model package versions are also in metadata |

## Validation design

Seed 42; GroupShuffleSplit reserves seven batteries (277 records) as the unseen-battery test. The remaining 27 batteries provide 1,138 development records. The earliest 80% of each development battery (905 records) is used for grouped five-fold CV and a deterministic 12-candidate RF search. The remaining later cycles (233 records from 20 batteries) never enter model or parameter selection. Single-record batteries enter only the early partition.

After selection is frozen, one RF fitted on the early partition evaluates the later cycles. A separate RF fitted on all development records evaluates the seven held-out batteries and is the model served by Streamlit. Those two evaluations must not be presented as measurements of the same fitted artifact.

| Evaluation | Records | Batteries | MAE, SOH ratio | RMSE | R² |
|---|---:|---:|---:|---:|---:|
| RF, unseen batteries | 277 | 7 | 0.060492 | 0.112210 | 0.159291 |
| RF, later development cycles | 233 | 20 | 0.029951 | 0.041204 | 0.880149 |
| Mean baseline, unseen batteries | 277 | 7 | 0.104271 | 0.122652 | -0.004450 |
| RF without capacity, exploratory ablation | 277 | 7 | 0.063225 | 0.106740 | 0.239261 |

Multiplying MAE by 100 gives percentage points: 6.05 for unseen batteries and 3.00 for later cycles. The sensor-only ablation uses fixed RF settings rather than independent tuning and is not a replacement model. This is one deterministic holdout; uncertainty across additional split seeds has not been estimated. No calibrated prediction intervals are available.

The tuned RF wins development CV. Its settings are 200 trees, max depth 20, min samples per leaf 4, max features sqrt, random state 42. All nine original model/baseline candidates and the tuned RF are recorded in `output/validated/model_comparison.csv`. Historical `output/model_comparison.csv` remains the old row-split comparison and is not current validation evidence.

## Artifacts and checks

`models/random_forest_pipeline.joblib` is the app artifact. Metadata records its checksum, dataset checksum, software versions, feature order, training ranges, battery membership, CV folds, parameter selection, and measured scores. Separate temporal and comparison-winner artifacts support reproducible checks. New reports, predictions, split membership, per-battery scores, tuning logs, and charts are under `output/validated/`.

The checks cover dataset rejection, split disjointness, temporal ordering, training-only scaling, serialization, inference ordering and input rejection, metadata mismatch, metric reproduction, app prediction, sample/reset, stale results, range messages, and model-loading errors. `pip check` passes in the existing Windows Python 3.14.6 environment. A new clean dependency installation and the remote CI run must be assessed separately.

The older `sites/battery-lab` is a separate nested Git repository serving the historical exported model. Its existing API/SQLite/33-sample parity check still passes with zero Python prediction difference. It has not been replaced with the new Streamlit model, and its files are not included in the root-repository commit.

## Reproduce

```powershell
python -m src.data_diagnostics
python -m src.feature_engineering
python -m src.model_training
python -m src.model_evaluation
python -m src.eda
python -m unittest discover -s tests -v
python -m streamlit run app.py
```

The training command is authoritative; tuning also delegates to that same workflow. Scripts accept `--help`, and direct absolute script paths work from other working directories. Retraining creates a matching artifact/metadata pair; deploy those files together. Original source data and legacy artifacts remain unchanged for rollback and historical reproduction.
