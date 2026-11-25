"""Create battery-disjoint holdouts and chronological development partitions."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

try:
    from .data_preprocessing import DATASET, FEATURES, ROOT, load_dataset
except ImportError:
    from data_preprocessing import DATASET, FEATURES, ROOT, load_dataset


def make_splits(data, seed=42, test_size=0.2, later_fraction=0.2):
    if data["battery_id"].nunique() < 6:
        raise ValueError("At least six distinct batteries are required for grouped validation.")
    if not 0 < later_fraction < 1:
        raise ValueError("later_fraction must be between zero and one.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    development, holdout = next(splitter.split(data[FEATURES], data["soh"], data["battery_id"]))
    early, later = [], []
    for _, battery in data.iloc[development].groupby("battery_id"):
        ordered = battery.sort_values("cycle").index.to_numpy()
        cut = max(1, int(np.floor(len(ordered) * (1 - later_fraction))))
        early.extend(ordered[:cut])
        later.extend(ordered[cut:])
    if not later:
        raise ValueError("No batteries have enough cycles for later-cycle evaluation.")
    return {
        "development": np.sort(development), "unseen_batteries": np.sort(holdout),
        "development_early": np.sort(early), "development_later": np.sort(later),
    }


def split_manifest(data, splits):
    manifest = data[["battery_id", "cycle"]].copy()
    manifest.insert(0, "row_id", data.index)
    manifest["partition"] = ""
    for key in ["development_early", "development_later", "unseen_batteries"]:
        manifest.loc[splits[key], "partition"] = key
    if manifest["partition"].eq("").any():
        raise ValueError("A row was not assigned to a validation partition.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "validated")
    args = parser.parse_args()
    data = load_dataset(args.dataset)
    splits = make_splits(data)
    args.output.mkdir(parents=True, exist_ok=True)
    split_manifest(data, splits).to_csv(args.output / "split_manifest.csv", index=False)
    for label, key in [("train", "development"), ("test", "unseen_batteries")]:
        subset = data.iloc[splits[key]]
        subset[FEATURES].to_csv(args.output / f"raw_X_{label}.csv", index=False)
        subset[["soh"]].to_csv(args.output / f"y_{label}.csv", index=False)
    print("Saved raw splits. Scaling is fitted inside each training pipeline, never before cross-validation.")


if __name__ == "__main__":
    main()
