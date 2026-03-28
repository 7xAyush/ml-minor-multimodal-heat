#!/usr/bin/env python
"""
Validate raw REAL inputs before building the dataset.

Checks:
- raw/satellite_metadata.csv: required columns, duplicate image_id, basic stats
- raw/satellite_images/: missing image files
- raw/weather_downloaded.csv: coverage for all (date, tile_id) pairs

Usage:
  python scripts/validate_real_inputs.py \
    --sat_meta raw/satellite_metadata.csv \
    --images_dir raw/satellite_images \
    --weather_csv raw/weather_downloaded.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Set, Tuple

import pandas as pd


def validate_satellite_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Satellite metadata CSV not found at {path}")
    df = pd.read_csv(path)

    required = ["image_id", "date", "lst", "tile_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Satellite metadata CSV at {path} is missing required columns: {missing}"
        )

    if df["image_id"].duplicated().any():
        dup_ids = df.loc[df["image_id"].duplicated(), "image_id"].unique()
        raise ValueError(
            f"Duplicate image_id values found in satellite metadata: {list(dup_ids)[:10]}"
        )

    print(f"[OK] Satellite metadata loaded from {path} with {len(df)} rows.")
    print(f"     Unique tiles: {df['tile_id'].nunique()}, unique dates: {df['date'].nunique()}")
    return df


def validate_images(df: pd.DataFrame, images_dir: Path) -> None:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    missing: list[str] = []
    for image_id in df["image_id"].astype(str):
        img_path = images_dir / f"{image_id}.png"
        if not img_path.exists():
            missing.append(str(img_path))

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} image files under {images_dir}. "
            f"First few missing: {missing[:5]}"
        )

    print(f"[OK] All {len(df)} image_id entries have corresponding PNG files in {images_dir}.")


def validate_weather(df: pd.DataFrame, weather_csv: Path) -> None:
    if not weather_csv.exists():
        raise FileNotFoundError(f"Weather CSV not found at {weather_csv}")

    w = pd.read_csv(weather_csv)
    required = ["date", "tile_id"]
    missing = [c for c in required if c not in w.columns]
    if missing:
        raise ValueError(
            f"Weather CSV at {weather_csv} is missing required columns: {missing}"
        )

    pairs_needed: Set[Tuple[str, str]] = set(
        zip(df["date"].astype(str), df["tile_id"].astype(str))
    )
    pairs_have: Set[Tuple[str, str]] = set(
        zip(w["date"].astype(str), w["tile_id"].astype(str))
    )
    missing_pairs = pairs_needed - pairs_have

    if missing_pairs:
        missing_list = list(missing_pairs)
        preview = missing_list[:5]
        raise ValueError(
            f"Weather CSV at {weather_csv} is missing {len(missing_pairs)} (date, tile_id) "
            f"pairs required by satellite metadata. First few missing: {preview}"
        )

    print(
        f"[OK] Weather CSV at {weather_csv} covers all {len(pairs_needed)} (date, tile_id) pairs."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate raw REAL inputs for the multimodal heat-risk pipeline."
    )
    parser.add_argument(
        "--sat_meta",
        type=str,
        default="raw/satellite_metadata.csv",
        help="Path to satellite metadata CSV.",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="raw/satellite_images",
        help="Directory containing satellite image chips.",
    )
    parser.add_argument(
        "--weather_csv",
        type=str,
        default="raw/weather_downloaded.csv",
        help="Path to weather CSV.",
    )
    args = parser.parse_args()

    sat_meta_path = Path(args.sat_meta)
    images_dir = Path(args.images_dir)
    weather_path = Path(args.weather_csv)

    df = validate_satellite_metadata(sat_meta_path)
    validate_images(df, images_dir)
    validate_weather(df, weather_path)

    print("[OK] Raw REAL inputs passed all validation checks.")


if __name__ == "__main__":
    main()

