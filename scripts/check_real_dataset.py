#!/usr/bin/env python
"""
Quick inspection tool for a built REAL dataset.

Reports:
- total samples
- split sizes
- class distribution
- number of unique tiles and dates

Usage:
  python scripts/check_real_dataset.py --dataset_dir dataset_real
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a built REAL dataset (sizes, classes, tiles, dates)."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset_real",
        help="Path to dataset directory (default: dataset_real).",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    meta_path = dataset_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found at {meta_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    mode = meta.get("mode", {})
    split_sizes = meta.get("split_sizes", {})
    class_dist = meta.get("class_distribution", {})

    print("=== Dataset overview ===")
    print(f"dataset_dir        : {dataset_dir}")
    print(f"data_mode          : {mode.get('data_mode', 'unknown')}")
    print(f"LST source         : {mode.get('lst_source', 'unknown')}")
    print(f"Imagery source     : {mode.get('image_source', 'unknown')}")
    print(f"Weather source     : {mode.get('weather_source', 'unknown')}")
    print(f"Total samples      : {meta.get('total_samples', 'unknown')}")
    print(f"Split sizes (rows) : {split_sizes}")
    print(f"Class distribution : {class_dist}")

    labels_path = dataset_dir / "labels.csv"
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path)
        n_tiles = labels_df["tile_id"].nunique() if "tile_id" in labels_df.columns else "NA"
        n_dates = labels_df["date"].nunique() if "date" in labels_df.columns else "NA"
        print("=== Labels table details ===")
        print(f"Unique tiles (tile_id): {n_tiles}")
        print(f"Unique dates (date)   : {n_dates}")
    else:
        print(f"[WARN] labels.csv not found at {labels_path}; skipping tile/date counts.")

    print("=== Integrity hints ===")
    if split_sizes.get("test", 0) == 0:
        print(
            "[WARN] Test split is empty. Strict evaluation on a held-out test set "
            "is not possible. Increase dataset size or adjust split ratios."
        )
    if len(class_dist) < 3:
        print(
            "[WARN] Fewer than 3 classes present in labels. Some metrics may be "
            "unstable; consider collecting more diverse LST conditions."
        )


if __name__ == "__main__":
    main()

