#!/usr/bin/env python
"""
Debugging tool for the REAL data pipeline.

This script is intended to make row-count and key-count changes transparent
across the major REAL pipeline stages, so that issues like unexpected
collapsing from hundreds of rows to a handful can be diagnosed quickly.

It does NOT modify the dataset; it only reports what would happen.

Steps:
  1. Load raw/satellite_metadata.csv
  2. Load raw/weather_downloaded.csv
  3. Report basic counts and duplicate patterns
  4. Simulate:
       - image validation (via clean.validate_and_filter_images)
       - join with weather on (date, tile_id)
       - heat-risk labelling
  5. Report row counts, unique image_id/tile_id/date, and class counts after
     each step.

Usage:

  python scripts/debug_real_pipeline.py --config config_real.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys

# Ensure repository root (parent of scripts/) is on sys.path so that `src`
# can be imported even when this script is executed directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils import load_config, ensure_dir  # type: ignore
from src import clean, label  # type: ignore


def report_stage(name: str, df: pd.DataFrame) -> None:
    print(f"\n=== {name} ===")
    print(f"rows              : {len(df)}")
    if "image_id" in df.columns:
        print(f"unique image_id   : {df['image_id'].nunique()}")
        dup_img = df["image_id"].duplicated().sum()
        print(f"duplicate image_id: {int(dup_img)}")
    if "tile_id" in df.columns:
        print(f"unique tile_id    : {df['tile_id'].nunique()}")
    if "date" in df.columns:
        print(f"unique date       : {df['date'].nunique()}")
    if {"tile_id", "date"} <= set(df.columns):
        dup_tile_date = df.duplicated(subset=["tile_id", "date"]).sum()
        print(f"duplicate (tile_id,date): {int(dup_tile_date)}")
    if "label_class" in df.columns:
        print("class counts      :", df["label_class"].value_counts().to_dict())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug REAL pipeline row counts and key counts."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_real.yaml",
        help="Path to config YAML (default: config_real.yaml).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg = config["paths"]

    sat_meta_path = Path(paths_cfg["satellite_metadata_csv"])
    weather_path = Path(paths_cfg["weather_csv"])
    raw_img_dir = Path(paths_cfg["raw_satellite_dir"])

    if not sat_meta_path.exists():
        raise FileNotFoundError(f"Satellite metadata CSV not found at {sat_meta_path}")
    if not weather_path.exists():
        raise FileNotFoundError(f"Weather CSV not found at {weather_path}")
    if not raw_img_dir.exists():
        raise FileNotFoundError(f"Raw satellite image directory not found at {raw_img_dir}")

    sat = pd.read_csv(sat_meta_path)
    weather = pd.read_csv(weather_path)

    print("=== RAW INPUTS ===")
    print(f"satellite_metadata: {sat_meta_path} (rows={len(sat)})")
    print(f"weather_csv       : {weather_path} (rows={len(weather)})")

    report_stage("Satellite metadata (raw)", sat)
    report_stage("Weather (raw)", weather)

    # Clean weather similarly to the REAL pipeline.
    weather_clean = clean.clean_weather(weather, config)
    report_stage("Weather after clean_weather()", weather_clean)

    # Validate images / resize (uses REAL-mode logic from config).
    images_out_dir = Path(config["paths"]["dataset_root"]) / "debug_images"
    ensure_dir(images_out_dir)
    sat_valid, valid_image_ids, image_stats = clean.validate_and_filter_images(
        sat, raw_img_dir, images_out_dir, config
    )
    print("\n=== Image validation stats ===")
    for k, v in image_stats.items():
        print(f"{k:20s}: {v}")
    report_stage("Satellite metadata after validate_and_filter_images()", sat_valid)

    # Align satellite + weather by (date, tile_id).
    join_keys = ["date", "tile_id"]
    aligned = sat_valid.merge(weather_clean, on=join_keys, how="inner")
    report_stage("Aligned (satellite + weather)", aligned)

    if aligned.empty:
        print("[ERROR] Aligned dataset is empty; check date/tile_id values.")
        return

    # Label using the same logic as real_pipeline.
    labelled = label.add_heat_risk_labels(aligned, config)
    report_stage("After add_heat_risk_labels()", labelled)

    print("\n[INFO] Debugging complete. Use the above stages to locate where rows are lost.")


if __name__ == "__main__":
    main()
