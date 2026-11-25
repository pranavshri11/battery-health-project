"""Validate the aggregated dataset without silently changing its measurements."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "dataset.csv"
FEATURES = ["cycle", "voltage", "temperature", "capacity"]
TARGET = "soh"
REQUIRED = ["battery_id", *FEATURES, TARGET, "rul"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_sha256(path):
    """Treat Git's LF/CRLF checkout conversion as identical CSV content."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def validate_dataset(frame):
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing dataset columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("The dataset is empty.")
    data = frame.copy()
    if data[REQUIRED].isna().any().any():
        raise ValueError("Dataset contains missing values; correct the source instead of silently dropping rows.")
    data["battery_id"] = data["battery_id"].astype(str).str.strip()
    if data["battery_id"].eq("").any():
        raise ValueError("Battery IDs must not be blank.")
    numeric = [*FEATURES, TARGET, "rul"]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if not np.isfinite(data[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains non-finite numeric values.")
    if data.duplicated(["battery_id", "cycle"]).any():
        raise ValueError("Duplicate battery_id/cycle records require an explicit source correction.")
    for column in ["cycle", "rul"]:
        if not np.equal(data[column], np.floor(data[column])).all():
            raise ValueError(f"{column} must contain integer counts.")
        data[column] = data[column].astype("int64")
    if (data["cycle"] < 1).any() or (data["rul"] < 0).any():
        raise ValueError("Cycle must be positive and RUL must be non-negative.")
    if (data[["voltage", "capacity", TARGET]] <= 0).any().any():
        raise ValueError("Voltage, capacity, and SOH must be positive.")
    if (data["temperature"] < -273.15).any():
        raise ValueError("Temperature cannot be below absolute zero in degrees Celsius.")
    return data.sort_values(["battery_id", "cycle"], kind="stable").reset_index(drop=True)


def load_dataset(path=DATASET):
    return validate_dataset(pd.read_csv(path))


def data_report(data, source=DATASET):
    initial = data.groupby("battery_id")["capacity"].first()
    reconstructed_soh = data["capacity"] / data["battery_id"].map(initial)
    endpoint = data.groupby("battery_id")["cycle"].max()
    return {
        "source_sha256": dataset_sha256(source), "rows": len(data),
        "batteries": int(data["battery_id"].nunique()), "columns": list(data.columns),
        "missing_values": data.isna().sum().astype(int).to_dict(),
        "duplicate_battery_cycles": int(data.duplicated(["battery_id", "cycle"]).sum()),
        "ranges": {key: {"min": float(data[key].min()), "max": float(data[key].max())}
                   for key in [*FEATURES, TARGET, "rul"]},
        "records_per_battery": data.groupby("battery_id").size().astype(int).to_dict(),
        "soh_above_one": int((data[TARGET] > 1).sum()),
        "soh_formula_max_absolute_error": float((reconstructed_soh - data[TARGET]).abs().max()),
        "rul_endpoint_formula_max_absolute_error": float((data["battery_id"].map(endpoint) - data["cycle"] - data["rul"]).abs().max()),
        "interpretation": [
            "For this CSV, SOH equals capacity divided by the battery's first observed capacity.",
            "Capacity is target-related: this is capacity-informed SOH estimation, not independent sensor-only forecasting.",
            "RUL equals final observed cycle minus current cycle; it is not verified physical end-of-life.",
            "SOH above 1 is retained without clipping; first observed capacity is not necessarily rated capacity.",
            "Original raw NASA files, extraction code, and licensing evidence are not present in this repository.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = data_report(load_dataset(args.dataset), args.dataset)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
