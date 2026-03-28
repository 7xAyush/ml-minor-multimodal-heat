#!/usr/bin/env python
"""
Overlay a REAL per-pixel heatmap (LST) on top of a satellite RGB tile.

This script uses the original Landsat composite (with RED, GREEN, BLUE,
NDVI, LST_C bands) exported from Google Earth Engine and the EE metadata
CSV to:

  - Extract the RGB patch for a given tile/date (image_id).
  - Extract the corresponding LST_C patch.
  - Render an RGB image with a semi-transparent temperature overlay.

Inputs:
  --ee_metadata_csv  raw/landsat/chennai_tiles_metadata.csv
     Columns required: tile_id, date, image_id, min_lon, min_lat, max_lon, max_lat

  --composite_tif    raw/landsat/chennai_landsat_composite.tif
     Bands expected: [1=RED, 2=GREEN, 3=BLUE, 4=NDVI, 5=LST_C]

  --image_id         image_id to visualize (e.g., tile_r0_c0_2022-05-26)
     If not provided, the first row in the metadata CSV is used.

Output:
  docs/tile_heat_overlays/<image_id>_heat_overlay.png

Usage example:

  python scripts/overlay_tile_heatmap.py \\
      --ee_metadata_csv raw/landsat/chennai_tiles_metadata.csv \\
      --composite_tif   raw/landsat/chennai_landsat_composite.tif \\
      --image_id        tile_r0_c0_2022-05-26
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds


def normalize_rgb(patch: np.ndarray) -> np.ndarray:
    """Per-channel 2–98 percentile scaling to uint8 RGB."""
    if patch.ndim != 3 or patch.shape[2] != 3:
        raise ValueError(f"Expected RGB patch with shape (H,W,3), got {patch.shape}")
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
    parser = argparse.ArgumentParser(
        description="Overlay REAL LST heatmap on top of an RGB tile."
    )
    parser.add_argument(
        "--ee_metadata_csv",
        type=str,
        default="raw/landsat/chennai_tiles_metadata.csv",
        help="Path to Earth Engine tiles metadata CSV.",
    )
    parser.add_argument(
        "--composite_tif",
        type=str,
        default="raw/landsat/chennai_landsat_composite.tif",
        help="Path to composite GeoTIFF with RGB+NDVI+LST_C.",
    )
    parser.add_argument(
        "--image_id",
        type=str,
        default=None,
        help="image_id to visualize (if omitted, first row of metadata is used).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Opacity of the LST overlay (0–1, default 0.5).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="docs/tile_heat_overlays",
        help="Directory to save overlay PNGs.",
    )
    args = parser.parse_args()

    meta_path = Path(args.ee_metadata_csv)
    comp_path = Path(args.composite_tif)
    if not meta_path.exists():
        raise FileNotFoundError(f"EE metadata CSV not found at {meta_path}")
    if not comp_path.exists():
        raise FileNotFoundError(f"Composite GeoTIFF not found at {comp_path}")

    df = pd.read_csv(meta_path)
    required = ["tile_id", "date", "image_id", "min_lon", "min_lat", "max_lon", "max_lat"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"EE metadata CSV at {meta_path} is missing required columns: {missing}"
        )

    if args.image_id is None:
        row = df.iloc[0]
    else:
        sub = df[df["image_id"].astype(str) == str(args.image_id)]
        if sub.empty:
            raise ValueError(
                f"image_id='{args.image_id}' not found in {meta_path}. "
                "Check spelling or choose a different ID."
            )
        row = sub.iloc[0]

    image_id = str(row["image_id"])
    tile_id = str(row["tile_id"])
    date = str(row["date"])
    min_lon = float(row["min_lon"])
    min_lat = float(row["min_lat"])
    max_lon = float(row["max_lon"])
    max_lat = float(row["max_lat"])

    # Read RGB + LST_C patch from composite.
    with rasterio.open(comp_path) as src:
        src_crs = src.crs
        left, bottom, right, top = transform_bounds(
            "EPSG:4326",
            src_crs,
            min_lon,
            min_lat,
            max_lon,
            max_lat,
            densify_pts=2,
        )
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        rgb = src.read((1, 2, 3), window=window, boundless=True, fill_value=0)
        lst = src.read(5, window=window, boundless=True, fill_value=np.nan)

    rgb = np.transpose(rgb, (1, 2, 0))  # (H,W,3)
    lst = lst.squeeze()  # (H,W)

    rgb_uint8 = normalize_rgb(rgb)

    # Prepare LST overlay: mask invalid, scale to colormap.
    lst_mask = np.isfinite(lst)
    if not lst_mask.any():
        raise ValueError("LST patch contains no finite values; cannot overlay heatmap.")

    lst_vals = lst[lst_mask]
    p5, p95 = np.percentile(lst_vals, [5, 95])
    if p95 <= p5:
        p95 = p5 + 1.0
    lst_norm = (lst - p5) / (p95 - p5)
    lst_norm = np.clip(lst_norm, 0.0, 1.0)

    cmap = plt.cm.inferno

    # Create figure
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    # Show RGB
    ax.imshow(rgb_uint8, origin="upper")
    # Show LST overlay
    ax.imshow(
        cmap(lst_norm),
        origin="upper",
        alpha=args.alpha,
    )

    ax.set_title(f"{tile_id} @ {date}\nLST overlay", fontsize=8)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_id}_heat_overlay.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[INFO] Saved heat overlay for {image_id} to {out_path}")


if __name__ == "__main__":
    main()

