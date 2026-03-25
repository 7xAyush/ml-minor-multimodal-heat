#!/usr/bin/env python
"""
Convert Earth Engine exports (tile-level metadata + composite GeoTIFF)
into REAL mode inputs for this repository:

- raw/satellite_metadata.csv
- raw/satellite_images/<image_id>.png

EE metadata CSV must contain at least:
  - tile_id
  - date  (YYYY-MM-DD)
  - image_id
  - lst   (LST in Celsius, real)
  - min_lon, min_lat, max_lon, max_lat  (tile bounds in EPSG:4326)

Usage example:

python scripts/prepare_real_inputs.py \
  --ee_metadata_csv raw/landsat/ExampleCity_tiles_metadata.csv \
  --composite_tif raw/landsat/ExampleCity_landsat_composite.tif \
  --output_metadata_csv raw/satellite_metadata.csv \
  --output_images_dir raw/satellite_images \
  --tile_size_m 1000 \
  --image_size 224
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare REAL mode inputs from Earth Engine exports "
            "(tile metadata CSV + composite GeoTIFF)."
        )
    )
    parser.add_argument(
        "--ee_metadata_csv",
        required=True,
        help="Path to Earth Engine tile-level metadata CSV (with bounds, lst, date, tile_id, image_id).",
    )
    parser.add_argument(
        "--composite_tif",
        required=True,
        help="Path to exported composite GeoTIFF from Earth Engine.",
    )
    parser.add_argument(
        "--output_metadata_csv",
        default="raw/satellite_metadata.csv",
        help="Output path for REAL mode metadata CSV (default: raw/satellite_metadata.csv).",
    )
    parser.add_argument(
        "--output_images_dir",
        default="raw/satellite_images",
        help="Directory where image chips (PNG) will be written (default: raw/satellite_images).",
    )
    parser.add_argument(
        "--tile_size_m",
        type=float,
        default=1000.0,
        help="Tile size in meters (for reporting only; geometry comes from EE bounds).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Target square image size in pixels (e.g., 224 -> 224x224).",
    )
    parser.add_argument(
        "--use_existing_tile_geometries",
        action="store_true",
        default=True,
        help=(
            "For compatibility only. Geometries are always taken from EE-provided "
            "bounds (min_lon/min_lat/max_lon/max_lat); if these are missing, the "
            "script fails loudly."
        ),
    )
    return parser.parse_args()


def sanitize_image_id(image_id: str) -> str:
    """
    Make image_id filesystem-safe: keep alphanumerics, dash, underscore.
    Replace other characters with underscore.
    """
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    return "".join(ch if ch in safe_chars else "_" for ch in image_id)


def validate_date(date_str: str) -> bool:
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except Exception:
        return False


def normalize_rgb_patch(patch: np.ndarray) -> np.ndarray:
    """
    Normalize an HxWx3 RGB patch to uint8 [0, 255] for PNG export.
    Uses per-channel 2nd-98th percentile scaling.
    """
    if patch.ndim != 3 or patch.shape[2] != 3:
        raise ValueError(f"Expected patch with shape (H,W,3), got {patch.shape}")

    # Handle all-zero or non-finite patches gracefully
    if not np.isfinite(patch).any() or np.all(patch == 0):
        return np.zeros_like(patch, dtype=np.uint8)

    out = np.zeros_like(patch, dtype=np.uint8)
    for c in range(3):
        band = patch[:, :, c].astype(np.float32)
        mask = np.isfinite(band)
        if not mask.any():
            continue
        vmin = np.percentile(band[mask], 2.0)
        vmax = np.percentile(band[mask], 98.0)
        if vmax <= vmin:
            vmax = vmin + 1.0
        band = np.clip((band - vmin) / (vmax - vmin), 0.0, 1.0)
        out[:, :, c] = (band * 255.0).astype(np.uint8)
    return out


def main() -> None:
    args = parse_args()

    ee_csv_path = Path(args.ee_metadata_csv)
    composite_path = Path(args.composite_tif)
    out_meta_path = Path(args.output_metadata_csv)
    out_img_dir = Path(args.output_images_dir)

    if not ee_csv_path.exists():
        print(f"[ERROR] EE metadata CSV not found: {ee_csv_path}", file=sys.stderr)
        sys.exit(1)
    if not composite_path.exists():
        print(f"[ERROR] Composite GeoTIFF not found: {composite_path}", file=sys.stderr)
        sys.exit(1)

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_meta_path.parent.mkdir(parents=True, exist_ok=True)

    # Load EE metadata CSV
    df = pd.read_csv(ee_csv_path)
    n_rows = len(df)
    print(f"[INFO] Loaded EE metadata with {n_rows} rows from {ee_csv_path}")

    required_cols = [
        "tile_id",
        "date",
        "image_id",
        "lst",
        "min_lon",
        "min_lat",
        "max_lon",
        "max_lat",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(
            "[ERROR] EE metadata CSV is missing required columns "
            f"(bounds are mandatory for robust cropping): {missing}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Core type normalization
    df["tile_id"] = df["tile_id"].astype(str)
    df["date"] = df["date"].astype(str)
    df["image_id"] = df["image_id"].astype(str)
    df["lst"] = pd.to_numeric(df["lst"], errors="coerce")

    # Optional extra columns to propagate
    # We support NDVI_mean and/or NDVI; both are treated as extra.
    extra_cols = []
    for col in ["lat", "lon", "NDVI_mean", "NDVI", "scene_id", "cloud_cover_land"]:
        if col in df.columns:
            extra_cols.append(col)

    # Open composite with rasterio
    with rasterio.open(composite_path) as src:
        src_crs = src.crs
        print(f"[INFO] Opened composite {composite_path} with CRS {src_crs}")

        # Assume band order in composite:
        # [RED, GREEN, BLUE, NDVI, LST_C] -> we use first 3 bands for PNG
        rgb_indexes = (1, 2, 3)

        successes = 0
        failures = 0
        missing_geom = 0
        duplicate_ids = 0
        used_ids: set[str] = set()
        out_records: list[dict[str, object]] = []

        for idx, row in df.iterrows():
            tile_id = row["tile_id"]
            date_str = row["date"]
            lst_val = row["lst"]

            if not isinstance(tile_id, str) or not tile_id:
                print(f"[WARN] Row {idx}: missing or empty tile_id; skipping")
                missing_geom += 1
                failures += 1
                continue
            if not validate_date(date_str):
                print(
                    f"[WARN] Row {idx}, tile_id={tile_id}: invalid date '{date_str}'; skipping"
                )
                failures += 1
                continue
            if not np.isfinite(lst_val):
                print(
                    f"[WARN] Row {idx}, tile_id={tile_id}: non-numeric LST '{lst_val}'; skipping"
                )
                failures += 1
                continue

            raw_image_id = row["image_id"]
            if not isinstance(raw_image_id, str) or not raw_image_id:
                raw_image_id = f"{tile_id}_{date_str}"
            image_id = sanitize_image_id(raw_image_id)

            if image_id in used_ids:
                print(
                    f"[WARN] Duplicate image_id '{image_id}' "
                    f"(tile_id={tile_id}, date={date_str}); skipping duplicate"
                )
                duplicate_ids += 1
                failures += 1
                continue
            used_ids.add(image_id)

            try:
                min_lon = float(row["min_lon"])
                min_lat = float(row["min_lat"])
                max_lon = float(row["max_lon"])
                max_lat = float(row["max_lat"])
            except Exception:
                print(
                    f"[WARN] Row {idx}, tile_id={tile_id}: invalid bounds; skipping",
                    file=sys.stderr,
                )
                missing_geom += 1
                failures += 1
                continue

            # Transform bounds from EPSG:4326 to raster CRS
            try:
                left, bottom, right, top = transform_bounds(
                    "EPSG:4326",
                    src_crs,
                    min_lon,
                    min_lat,
                    max_lon,
                    max_lat,
                    densify_pts=2,
                )
            except Exception as exc:
                print(
                    f"[WARN] Row {idx}, tile_id={tile_id}: transform_bounds failed ({exc}); skipping"
                )
                failures += 1
                continue

            # Build window in raster coordinates
            try:
                window = from_bounds(left, bottom, right, top, transform=src.transform)
            except Exception as exc:
                print(
                    f"[WARN] Row {idx}, tile_id={tile_id}: from_bounds failed ({exc}); skipping"
                )
                failures += 1
                continue

            # Read patch (RGB bands)
            try:
                patch = src.read(rgb_indexes, window=window, boundless=True, fill_value=0)
                patch = np.transpose(patch, (1, 2, 0))  # (C,H,W) -> (H,W,C)
                if patch.size == 0 or patch.shape[0] == 0 or patch.shape[1] == 0:
                    print(
                        f"[WARN] Row {idx}, tile_id={tile_id}: empty patch; skipping"
                    )
                    failures += 1
                    continue
            except Exception as exc:
                print(
                    f"[WARN] Row {idx}, tile_id={tile_id}: reading patch failed ({exc}); skipping"
                )
                failures += 1
                continue

            # Normalize to uint8 RGB and resize
            try:
                patch_uint8 = normalize_rgb_patch(patch)
                img = Image.fromarray(patch_uint8, mode="RGB")
                if args.image_size is not None and args.image_size > 0:
                    img = img.resize((args.image_size, args.image_size), Image.BILINEAR)
            except Exception as exc:
                print(
                    f"[WARN] Row {idx}, tile_id={tile_id}: normalization/resize failed ({exc}); skipping"
                )
                failures += 1
                continue

            out_path = out_img_dir / f"{image_id}.png"
            try:
                img.save(out_path)
            except Exception as exc:
                print(
                    f"[WARN] Row {idx}, tile_id={tile_id}: saving PNG failed ({exc}); skipping"
                )
                failures += 1
                continue

            # Build metadata record for REAL mode
            record: dict[str, object] = {
                "image_id": image_id,
                "date": date_str,
                "tile_id": tile_id,
                "lst": float(lst_val),
            }
            for col in extra_cols:
                record[col] = row[col]
            out_records.append(record)
            successes += 1

    if not out_records:
        print(
            "[ERROR] No successful chips extracted; not writing metadata CSV.",
            file=sys.stderr,
        )
        sys.exit(1)

    out_df = pd.DataFrame(out_records)

    # Ensure uniqueness of image_id
    if out_df["image_id"].duplicated().any():
        dup_count = int(out_df["image_id"].duplicated().sum())
        print(
            f"[ERROR] {dup_count} duplicate image_id values in output metadata; aborting.",
            file=sys.stderr,
        )
        sys.exit(1)

    out_df.to_csv(out_meta_path, index=False)
    print(f"[INFO] Wrote REAL mode metadata to {out_meta_path}")

    # Summary report
    print("========== PREPARE_REAL_INPUTS SUMMARY ==========")
    print(f"EE metadata rows:      {n_rows}")
    print(f"Successful chips:      {successes}")
    print(f"Skipped chips:         {failures}")
    print(f"Missing/invalid geom:  {missing_geom}")
    print(f"Duplicate image_ids:   {duplicate_ids}")
    print(f"Final metadata rows:   {len(out_df)}")
    print("=================================================")


if __name__ == "__main__":
    main()

