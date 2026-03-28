#!/usr/bin/env python
"""
End-to-end REAL raw-data preparation helper.

This script ties together the local steps **after** you have exported
Landsat-based tiles + composite from Google Earth Engine using:
  - scripts/gee_export_city_tiles.js

It will:
  1) Run scripts/prepare_real_inputs.py to create:
       - raw/satellite_metadata.csv
       - raw/satellite_images/<image_id>.png
  2) Run scripts/fetch_open_meteo_weather.py to create:
       - raw/weather_downloaded.csv
  3) Run scripts/validate_real_inputs.py to ensure:
       - metadata, images, and weather are consistent
  4) Run scripts/preview_real_splits.py to show:
       - total rows, unique tiles/dates, provisional class balance,
         and split feasibility (including grouped-by-tile).

Usage example:

  python scripts/run_real_data_preparation.py \\
      --ee_metadata_csv raw/landsat/ExampleCity_tiles_metadata.csv \\
      --composite_tif   raw/landsat/ExampleCity_landsat_composite.tif

After this succeeds, you can run:

  python build_dataset.py --config config_real.yaml
  python train_multimodal.py --dataset_dir dataset_real
  python -m src.evaluate_test --dataset_dir dataset_real
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(cmd: list[str], step_name: str) -> None:
    print(f"\n========== {step_name} ==========")
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(
            f"[ERROR] Step '{step_name}' failed with exit code {result.returncode}. "
            "Aborting.",
            file=sys.stderr,
        )
        sys.exit(result.returncode)
    print(f"[OK] Step '{step_name}' completed successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end REAL raw-data preparation helper."
    )
    parser.add_argument(
        "--ee_metadata_csv",
        required=True,
        help="Path to Earth Engine tiles metadata CSV (from gee_export_city_tiles.js).",
    )
    parser.add_argument(
        "--composite_tif",
        required=True,
        help="Path to Landsat composite GeoTIFF (from gee_export_city_tiles.js).",
    )
    parser.add_argument(
        "--sat_meta_out",
        default="raw/satellite_metadata.csv",
        help="Output path for REAL satellite metadata CSV "
        "(default: raw/satellite_metadata.csv).",
    )
    parser.add_argument(
        "--images_dir",
        default="raw/satellite_images",
        help="Output directory for satellite image chips "
        "(default: raw/satellite_images).",
    )
    parser.add_argument(
        "--weather_out",
        default="raw/weather_downloaded.csv",
        help="Output path for REAL weather CSV (default: raw/weather_downloaded.csv).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Target image size in pixels (square). Default: 224.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ee_metadata_csv = Path(args.ee_metadata_csv)
    composite_tif = Path(args.composite_tif)

    if not ee_metadata_csv.exists():
        print(f"[ERROR] EE metadata CSV not found at {ee_metadata_csv}", file=sys.stderr)
        sys.exit(1)
    if not composite_tif.exists():
        print(f"[ERROR] Composite GeoTIFF not found at {composite_tif}", file=sys.stderr)
        sys.exit(1)

    # 1) Prepare REAL satellite metadata + image chips.
    run_step(
        [
            sys.executable,
            "scripts/prepare_real_inputs.py",
            "--ee_metadata_csv",
            str(ee_metadata_csv),
            "--composite_tif",
            str(composite_tif),
            "--output_metadata_csv",
            args.sat_meta_out,
            "--output_images_dir",
            args.images_dir,
            "--image_size",
            str(args.image_size),
        ],
        "Step 1: prepare_real_inputs (satellite images + metadata)",
    )

    # 2) Fetch REAL weather from Open-Meteo.
    run_step(
        [
            sys.executable,
            "scripts/fetch_open_meteo_weather.py",
            "--sat_meta",
            args.sat_meta_out,
            "--output_csv",
            args.weather_out,
        ],
        "Step 2: fetch_open_meteo_weather (weather CSV)",
    )

    # 3) Validate all REAL raw inputs.
    run_step(
        [
            sys.executable,
            "scripts/validate_real_inputs.py",
            "--sat_meta",
            args.sat_meta_out,
            "--images_dir",
            args.images_dir,
            "--weather_csv",
            args.weather_out,
        ],
        "Step 3: validate_real_inputs (sanity checks)",
    )

    # 4) Preview splits and class balance.
    run_step(
        [
            sys.executable,
            "scripts/preview_real_splits.py",
            "--sat_meta",
            args.sat_meta_out,
        ],
        "Step 4: preview_real_splits (split feasibility + class counts)",
    )

    print(
        "\n[OK] REAL raw data preparation completed.\n"
        "Next steps:\n"
        "  1) Inspect the preview outputs and ensure you have enough samples "
        "and reasonable class balance.\n"
        "  2) Build the REAL dataset:\n"
        "       python build_dataset.py --config config_real.yaml\n"
        "  3) Train and evaluate the multimodal model as usual.\n"
    )


if __name__ == "__main__":
    main()

