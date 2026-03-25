#!/usr/bin/env python
"""
Aggregate experiment results into a simple CSV table.

Reads:
  - dataset/experiments/baselines.json
  - dataset/experiments/ablations.json

Produces:
  - dataset/experiments/results_table.csv

Columns:
  - Model
  - Input Type
  - Accuracy
  - Macro-F1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate baseline and ablation results into a CSV table."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Path to dataset directory (default: dataset).",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    exp_dir = dataset_dir / "experiments"

    baselines_path = exp_dir / "baselines.json"
    ablations_path = exp_dir / "ablations.json"

    rows = []

    # Baselines
    if baselines_path.exists():
        with baselines_path.open("r", encoding="utf-8") as f:
            baselines = json.load(f)
        for key, entry in baselines.items():
            rows.append(
                {
                    "Model": entry.get("model", key),
                    "Input Type": entry.get("input_type", "unknown"),
                    "Accuracy": entry.get("accuracy", float("nan")),
                    "Macro-F1": entry.get("macro_f1", float("nan")),
                }
            )

    # Ablations
    if ablations_path.exists():
        with ablations_path.open("r", encoding="utf-8") as f:
            ablations = json.load(f)
        for key, entry in ablations.items():
            # Only include entries that have accuracy/macro_f1 fields
            if "accuracy" in entry or "macro_f1" in entry:
                rows.append(
                    {
                        "Model": entry.get("model", key),
                        "Input Type": entry.get("input_type", "unknown"),
                        "Accuracy": entry.get("accuracy", float("nan")),
                        "Macro-F1": entry.get("macro_f1", float("nan")),
                    }
                )

    if not rows:
        raise RuntimeError(
            "No results found in baselines.json or ablations.json. "
            "Run scripts/run_baselines.py and scripts/run_ablations.py first."
        )

    df = pd.DataFrame(rows)
    out_path = exp_dir / "results_table.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[INFO] Aggregated results written to {out_path}")


if __name__ == "__main__":
    main()

