"""Compare and tune only on development data, then evaluate frozen selections."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from .data_preprocessing import DATASET, FEATURES, ROOT, data_report, dataset_sha256, load_dataset, sha256
    from .feature_engineering import make_splits, split_manifest
except ImportError:
    from data_preprocessing import DATASET, FEATURES, ROOT, data_report, dataset_sha256, load_dataset, sha256
    from feature_engineering import make_splits, split_manifest


def make_pipeline(estimator):
    return Pipeline([("scaler", StandardScaler()), ("regressor", estimator)])


def model_candidates(seed=42):
    from xgboost import XGBRegressor
    return {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.001, max_iter=10000),
        "Decision Tree": DecisionTreeRegressor(random_state=seed),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=seed),
        "XGBoost": XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=seed, n_jobs=1),
    }


def metrics(actual, predicted):
    return {"MAE": float(mean_absolute_error(actual, predicted)),
            "MSE": float(mean_squared_error(actual, predicted)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "R2": float(r2_score(actual, predicted)) if len(actual) > 1 and np.var(actual) > 0 else None}


def evaluate(pipeline, data, label, output):
    prediction = pipeline.predict(data[FEATURES])
    result = metrics(data["soh"], prediction)
    rows = data[["battery_id", "cycle", "soh"]].copy()
    rows["predicted_soh"] = prediction
    rows["residual"] = rows["soh"] - prediction
    rows.to_csv(output / f"{label}_predictions.csv", index=False)
    grouped = []
    for battery, part in rows.groupby("battery_id"):
        grouped.append({"battery_id": battery, "records": len(part), **metrics(part["soh"], part["predicted_soh"])})
    pd.DataFrame(grouped).to_csv(output / f"{label}_per_battery.csv", index=False)
    result["macro_battery_MAE"] = float(np.mean([row["MAE"] for row in grouped]))
    result["records"] = len(data)
    result["batteries"] = int(data["battery_id"].nunique())
    return result


def train(dataset=DATASET, output=ROOT / "output" / "validated", model_dir=ROOT / "models", seed=42, jobs=2, search_iterations=12):
    output, model_dir = Path(output), Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset(dataset)
    report = data_report(data, dataset)
    (output / "data_quality.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    splits = make_splits(data, seed=seed)
    split_manifest(data, splits).to_csv(output / "split_manifest.csv", index=False)
    early = data.iloc[splits["development_early"]]
    later = data.iloc[splits["development_later"]]
    development = data.iloc[splits["development"]]
    unseen = data.iloc[splits["unseen_batteries"]]
    X, y, groups = early[FEATURES], early["soh"], early["battery_id"]
    fold_indices = list(GroupKFold(n_splits=min(5, groups.nunique())).split(X, y, groups))
    fold_manifest = [{"fold": number, "train_batteries": sorted(set(groups.iloc[tr])),
                      "validation_batteries": sorted(set(groups.iloc[va]))}
                     for number, (tr, va) in enumerate(fold_indices)]
    candidates = {name: make_pipeline(estimator) for name, estimator in model_candidates(seed).items()}
    rows = []
    for name, candidate in candidates.items():
        print(f"Grouped CV: {name}", flush=True)
        scores = cross_validate(candidate, X, y, cv=fold_indices, scoring={"MAE": "neg_mean_absolute_error", "R2": "r2"}, n_jobs=jobs, error_score="raise")
        rows.append({"Model": name, "CV MAE": float(-scores["test_MAE"].mean()),
                     "CV MAE std": float(scores["test_MAE"].std()), "CV R2": float(scores["test_R2"].mean())})

    print("Tuning Random Forest inside grouped CV (early development cycles only)", flush=True)
    search = RandomizedSearchCV(candidates["Random Forest"], {
        "regressor__n_estimators": [100, 200], "regressor__max_depth": [None, 10, 20],
        "regressor__min_samples_leaf": [1, 2, 4], "regressor__max_features": [1.0, "sqrt"],
    }, n_iter=search_iterations, random_state=seed, cv=fold_indices,
        scoring={"MAE": "neg_mean_absolute_error", "R2": "r2"}, n_jobs=jobs, error_score="raise", refit="MAE")
    search.fit(X, y)
    pd.DataFrame(search.cv_results_).to_csv(output / "rf_tuning.csv", index=False)
    candidates["Tuned Random Forest"] = clone(search.best_estimator_)
    rows.append({"Model": "Tuned Random Forest", "CV MAE": float(-search.best_score_),
                 "CV MAE std": float(search.cv_results_["std_test_MAE"][search.best_index_]),
                 "CV R2": float(search.cv_results_["mean_test_R2"][search.best_index_])})
    comparison = pd.DataFrame(rows).sort_values("CV MAE", kind="stable").reset_index(drop=True)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    best_name = comparison.iloc[0]["Model"]
    rf_name = comparison[comparison["Model"].isin(["Random Forest", "Tuned Random Forest"])].iloc[0]["Model"]

    # Selection is now frozen. Neither later-cycle nor unseen-battery labels were used above.
    temporal_model = clone(candidates[rf_name]).fit(X, y)
    temporal = evaluate(temporal_model, later, "rf_later_cycles", output)
    joblib.dump(temporal_model, model_dir / "temporal_rf_pipeline.joblib", compress=3)
    rf = clone(candidates[rf_name]).fit(development[FEATURES], development["soh"])
    unseen_metrics = evaluate(rf, unseen, "rf_unseen_batteries", output)
    best = clone(candidates[best_name]).fit(development[FEATURES], development["soh"])
    selected_metrics = evaluate(best, unseen, "selected_unseen_batteries", output)
    mean_baseline = make_pipeline(DummyRegressor()).fit(development[FEATURES], development["soh"])
    baseline_metrics = evaluate(mean_baseline, unseen, "mean_unseen_batteries", output)

    # This ablation reveals how much the capacity-derived target contributes to the task.
    sensor_features = ["cycle", "voltage", "temperature"]
    sensor_model = clone(candidates[rf_name]).fit(development[sensor_features], development["soh"])
    sensor_metrics = metrics(unseen["soh"], sensor_model.predict(unseen[sensor_features]))
    sensor_metrics["note"] = "Fixed RF settings; exploratory ablation, not an independently tuned sensor-only model."

    artifact = model_dir / "random_forest_pipeline.joblib"
    joblib.dump(rf, artifact, compress=3)
    joblib.dump(best, model_dir / "selected_model_pipeline.joblib", compress=3)
    metadata = {
        "schema_version": 1, "model_name": "Random Forest", "rf_selection": rf_name,
        "selected_comparison_model": best_name, "features": FEATURES,
        "target": "soh", "target_unit": "ratio", "target_definition": "capacity / first observed capacity per battery",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
        "dataset_sha256": dataset_sha256(dataset), "artifact_sha256": sha256(artifact),
        "python_version": platform.python_version(),
        "package_versions": {name: importlib.metadata.version(name) for name in ["scikit-learn", "pandas", "numpy", "joblib", "xgboost", "streamlit"]},
        "rf_parameters": rf.named_steps["regressor"].get_params(),
        "development_batteries": sorted(set(development["battery_id"])),
        "holdout_batteries": sorted(set(unseen["battery_id"])),
        "split_counts": {key: len(value) for key, value in splits.items()},
        "selection_protocol": "Minimum mean MAE in grouped CV on early development cycles only. Scaling is fitted inside every fold.",
        "unseen_protocol": "Refit on all development cycles, evaluate on complete held-out batteries; never fit on held-out batteries.",
        "temporal_protocol": "Fit on the earliest 80% of each development battery, evaluate on its remaining later cycles. Single-cycle batteries are training-only.",
        "cv_folds": fold_manifest,
        "training_ranges": {key: {"min": float(development[key].min()), "max": float(development[key].max())} for key in FEATURES},
        "metrics": {"rf_unseen_batteries": unseen_metrics, "rf_later_cycles": temporal,
                    "selected_unseen_batteries": selected_metrics, "mean_unseen_batteries": baseline_metrics,
                    "sensor_only_rf_unseen_batteries": sensor_metrics},
        "limitations": report["interpretation"],
        "artifact_policy": "The app serves the validated RF, even if another model wins CV. Legacy model/scaler files are unchanged for historical reproduction.",
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "evaluation.json").write_text(json.dumps(metadata["metrics"], indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(comparison.to_string(index=False), flush=True)
    print(json.dumps(metadata["metrics"], indent=2, allow_nan=False), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "validated")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--search-iterations", type=int, default=12)
    args = parser.parse_args()
    train(args.dataset, args.output, args.models, args.seed, args.jobs, args.search_iterations)


if __name__ == "__main__":
    main()
