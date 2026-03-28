#!/usr/bin/env python
"""
Preview expected train/val/test sizes and grouped split feasibility
for REAL data before running build_dataset.py.

This script inspects raw/satellite_metadata.csv and gives:
- total rows
- unique tiles
- unique dates
- per-class counts based on LST thresholds
- whether grouped split by tile_id is feasible
- approximate expected split sizes for a desired ratio

Usage:
  python scripts/preview_real_splits.py --sat_meta raw/satellite_metadata.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview REAL split feasibility and expected sizes."
    )
    parser.add_argument(
        "--sat_meta",
        type=str,
        default="raw/satellite_metadata.csv",
        help="Path to satellite metadata CSV.",
    )
    parser.add_argument(
        "--low_threshold",
        type=float,
        default=30.0,
        help="LST threshold for low vs medium (default: 30.0).",
    )
    parser.add_argument(
        "--high_threshold",
        type=float,
        default=35.0,
        help="LST threshold for medium vs high (default: 35.0).",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Train split ratio (default: 0.7).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15).",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test split ratio (default: 0.15).",
    )
    args = parser.parse_args()

    sat_meta_path = Path(args.sat_meta)
    if not sat_meta_path.exists():
        raise FileNotFoundError(f"Satellite metadata CSV not found at {sat_meta_path}")

    df = pd.read_csv(sat_meta_path)
    required = ["image_id", "date", "lst", "tile_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Satellite metadata CSV at {sat_meta_path} is missing required columns: {missing}"
        )

    n = len(df)
    n_tiles = df["tile_id"].nunique()
    n_dates = df["date"].nunique()

    print("=== RAW SATELLITE METADATA PREVIEW ===")
    print(f"Path           : {sat_meta_path}")
    print(f"Total rows     : {n}")
    print(f"Unique tiles   : {n_tiles}")
    print(f"Unique dates   : {n_dates}")

    # Derive provisional labels from LST thresholds
    lst = df["lst"].astype(float)
    low_t = args.low_threshold
    high_t = args.high_threshold
    def label_from_lst(x: float) -> int:
        if x < low_t:
            return 0
        if x < high_t:
            return 1
        return 2

    labels = lst.apply(label_from_lst)
    class_counts = labels.value_counts().to_dict()
    print("Class counts (provisional from LST):", class_counts)

    # Approximate split sizes
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        print("[WARN] Train/val/test ratios do not sum to 1.0; skipping split size preview.")
    else:
        est_train = int(round(n * train_ratio))
        est_val = int(round(n * val_ratio))
        est_test = n - est_train - est_val
        print("Expected split sizes (approx):")
        print(f"  train ~ {est_train}, val ~ {est_val}, test ~ {est_test}")
        if est_test == 0:
            print("[WARN] Estimated test size is 0; increase total samples or adjust ratios.")

    # Grouped split feasibility by tile_id
    if n_tiles < 3:
        print(
            "[WARN] Fewer than 3 unique tiles. Grouped split by tile_id will NOT be possible; "
            "add more tiles."
        )
    else:
        print(
            "[OK] Grouped split by tile_id is possible (tiles >= 3). "
            "Exact group sizes will depend on tile sample counts."
        )


if __name__ == "__main__":
    main()

