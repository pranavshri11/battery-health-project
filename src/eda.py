"""Reproducible dataset exploration without opening blocking plot windows."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from .data_preprocessing import DATASET, ROOT, load_dataset
except ImportError:
    from data_preprocessing import DATASET, ROOT, load_dataset


def make_plots(data, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    for feature in ["cycle", "voltage", "temperature", "capacity", "soh", "rul"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(data[feature], bins=25, ax=ax)
        ax.set(title=f"{feature.upper()} distribution")
        fig.tight_layout()
        fig.savefig(output / f"{feature}_distribution.png", dpi=150)
        plt.close(fig)
    for target in ["capacity", "soh", "rul"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for _, battery in data.groupby("battery_id"):
            ax.plot(battery["cycle"], battery[target], alpha=0.55, linewidth=1)
        ax.set(xlabel="Discharge cycle", ylabel=target, title=f"{target.upper()} by cycle, one line per battery")
        fig.tight_layout()
        fig.savefig(output / f"cycle_vs_{target}.png", dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature/target correlations (not causal)")
    fig.tight_layout()
    fig.savefig(output / "correlation_heatmap.png", dpi=150)
    plt.close(fig)
    summary = data.groupby("battery_id").agg(records=("cycle", "size"), first_cycle=("cycle", "min"), last_cycle=("cycle", "max"), first_soh=("soh", "first"), last_soh=("soh", "last"))
    summary.to_csv(output / "battery_summary.csv")
    declined = int((summary["last_soh"] < summary["first_soh"]).sum())
    (output / "findings.md").write_text(
        f"# Dataset exploration\n\n{len(data):,} records across {len(summary)} batteries. "
        f"{declined} batteries end with a lower SOH than their first observed record. "
        "Individual traces may fluctuate; no measurements are clipped or removed.\n\n"
        "SOH is capacity divided by first observed capacity for each battery in this CSV. "
        "RUL is the distance to the last recorded cycle, not a validated failure threshold. "
        "Correlation plots include derived targets and must not be interpreted as independent predictive evidence.\n",
        encoding="utf-8")
    print(f"Saved 10 figures and battery summaries to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "validated" / "eda")
    args = parser.parse_args()
    make_plots(load_dataset(args.dataset), args.output)


if __name__ == "__main__":
    main()
