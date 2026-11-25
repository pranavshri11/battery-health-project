"""Re-evaluate saved validation models against their recorded partitions."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    from .data_preprocessing import DATASET, FEATURES, ROOT, dataset_sha256, load_dataset
    from .model_training import evaluate
    from .prediction import load_predictor
except ImportError:
    from data_preprocessing import DATASET, FEATURES, ROOT, dataset_sha256, load_dataset
    from model_training import evaluate
    from prediction import load_predictor


def save_plots(model, data, output):
    prediction = model.predict(data[FEATURES])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(data["soh"], prediction, s=12, alpha=0.6)
    low = min(data["soh"].min(), prediction.min())
    high = max(data["soh"].max(), prediction.max())
    ax.plot([low, high], [low, high], "k--", linewidth=1)
    ax.set(xlabel="Actual SOH (ratio)", ylabel="Predicted SOH (ratio)", title="Random Forest: unseen batteries")
    fig.tight_layout()
    fig.savefig(output / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(prediction, data["soh"] - prediction, s=12, alpha=0.6)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Predicted SOH (ratio)", ylabel="Actual minus predicted", title="Unseen-battery residuals")
    fig.tight_layout()
    fig.savefig(output / "residuals.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    pd.Series(model.named_steps["regressor"].feature_importances_, index=FEATURES).sort_values().plot.barh(ax=ax)
    ax.set(xlabel="Impurity-based importance (not causal)", title="Random Forest feature importance")
    fig.tight_layout()
    fig.savefig(output / "feature_importance.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "validated")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    data = load_dataset(args.dataset)
    model, metadata = load_predictor(args.models / "random_forest_pipeline.joblib", args.models / "model_metadata.json")
    if metadata["dataset_sha256"] != dataset_sha256(args.dataset):
        raise ValueError("Dataset changed since training; regenerate the models and split manifest.")
    manifest = pd.read_csv(args.output / "split_manifest.csv")
    if len(manifest) != len(data) or not manifest[["battery_id", "cycle"]].equals(data[["battery_id", "cycle"]]):
        raise ValueError("The split manifest does not match the validated dataset.")
    unseen = data[manifest["partition"].eq("unseen_batteries")]
    later = data[manifest["partition"].eq("development_later")]
    if set(unseen["battery_id"]) != set(metadata["holdout_batteries"]):
        raise ValueError("The recorded holdout batteries do not match the split manifest.")
    temporal_model = joblib.load(args.models / "temporal_rf_pipeline.joblib")
    result = {"rf_unseen_batteries": evaluate(model, unseen, "rf_unseen_batteries", args.output),
              "rf_later_cycles": evaluate(temporal_model, later, "rf_later_cycles", args.output)}
    for protocol, measured in result.items():
        for metric in ["MAE", "RMSE", "R2"]:
            if abs(measured[metric] - metadata["metrics"][protocol][metric]) > 1e-12:
                raise ValueError(f"Saved {protocol} {metric} is not reproducible from the saved model.")
    save_plots(model, unseen, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
