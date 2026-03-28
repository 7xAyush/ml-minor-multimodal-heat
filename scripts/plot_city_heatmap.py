#!/usr/bin/env python
"""
Plot a city-level heatmap of tile-level heat risk.

This script uses REAL data already produced by the pipeline to create a
research-style figure showing:

  - The city grid (tiles) drawn on a lat/lon background.
  - Each tile colored by its mean LST (°C) across all dates.
  - Optional circular markers highlighting the hottest tiles.

Inputs (defaults are aligned with REAL mode):

  --tiles_csv   raw/tiles.csv
      Columns expected: tile_id, min_lat, min_lon, max_lat, max_lon, lat, lon

  --labels_csv  dataset_real/labels.csv
      Columns expected: tile_id, lst (LST in Celsius)

Output:

  docs/city_heatmap_true_lst.png

Usage:

  python scripts/plot_city_heatmap.py \
      --tiles_csv raw/tiles.csv \
      --labels_csv dataset_real/labels.csv

The resulting figure is suitable for inclusion in a paper: white background,
Lat/Lon axes, colorbar, and a few hottest tiles highlighted with circles.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a city heatmap of tile-level mean LST."
    )
    parser.add_argument(
        "--tiles_csv",
        type=str,
        default="raw/tiles.csv",
        help="Path to tiles.csv with tile bounds and centroids (default: raw/tiles.csv).",
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default="dataset_real/labels.csv",
        help="Path to labels.csv with tile_id and lst columns (default: dataset_real/labels.csv).",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="docs/city_heatmap_true_lst.png",
        help="Output PNG path for the heatmap figure.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of hottest tiles to highlight with circular markers (default: 3).",
    )
    args = parser.parse_args()

    tiles_path = Path(args.tiles_csv)
    labels_path = Path(args.labels_csv)
    out_path = Path(args.out_path)

    if not tiles_path.exists():
        raise FileNotFoundError(f"tiles_csv not found at {tiles_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"labels_csv not found at {labels_path}")

    tiles = pd.read_csv(tiles_path)
    labels = pd.read_csv(labels_path)

    required_tiles = {"tile_id", "min_lat", "min_lon", "max_lat", "max_lon", "lat", "lon"}
    missing_tiles = required_tiles - set(tiles.columns)
    if missing_tiles:
        raise ValueError(
            f"tiles_csv at {tiles_path} is missing required columns: {sorted(missing_tiles)}"
        )

    if "tile_id" not in labels.columns or "lst" not in labels.columns:
        raise ValueError(
            f"labels_csv at {labels_path} must contain 'tile_id' and 'lst' columns."
        )

    # Compute mean LST per tile_id from REAL labels.
    labels = labels.copy()
    labels["lst"] = pd.to_numeric(labels["lst"], errors="coerce")
    per_tile = labels.groupby("tile_id", as_index=False)["lst"].mean()
    per_tile = per_tile.rename(columns={"lst": "mean_lst"})

    merged = tiles.merge(per_tile, on="tile_id", how="left")
    if merged["mean_lst"].isna().all():
        raise ValueError(
            "All tiles have NaN mean_lst after merge. Check that tile_id values "
            "match between tiles.csv and labels.csv."
        )

    # Determine color scale from available LST values.
    lst_vals = merged["mean_lst"].dropna().values
    vmin = float(np.nanpercentile(lst_vals, 5))
    vmax = float(np.nanpercentile(lst_vals, 95))
    if vmin == vmax:
        vmax = vmin + 1.0

    # Identify hottest tiles to highlight (by mean LST).
    hottest = (
        merged.dropna(subset=["mean_lst"])
        .sort_values("mean_lst", ascending=False)
        .head(max(1, args.top_k))
    )
    hottest_ids = set(hottest["tile_id"].astype(str))

    # Figure setup
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Draw each tile as a rectangle colored by mean LST.
    cmap = plt.cm.hot

    for _, row in merged.iterrows():
        tile_id = str(row["tile_id"])
        min_lon = float(row["min_lon"])
        min_lat = float(row["min_lat"])
        max_lon = float(row["max_lon"])
        max_lat = float(row["max_lat"])
        mean_lst = row["mean_lst"]

        if np.isnan(mean_lst):
            color = "#EEEEEE"
        else:
            norm_val = (mean_lst - vmin) / (vmax - vmin)
            norm_val = float(np.clip(norm_val, 0.0, 1.0))
            color = cmap(norm_val)

        rect = Rectangle(
            (min_lon, min_lat),
            max_lon - min_lon,
            max_lat - min_lat,
            facecolor=color,
            edgecolor="#CCCCCC",
            linewidth=0.8,
        )
        ax.add_patch(rect)

    # Overlay conventional circular markers (scatter) on the hottest tiles.
    # We use a fixed marker size in points so the circles look visually
    # consistent and not distorted by the map extent.
    for _, row in hottest.iterrows():
        cx = float(row["lon"])
        cy = float(row["lat"])
        ax.scatter(
            [cx],
            [cy],
            s=120,  # marker size in points^2
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
        )
        ax.text(
            cx,
            cy,
            str(row["tile_id"]),
            ha="center",
            va="center",
            fontsize=6,
            color="#000000",
        )

    # Axes limits from tiles extent.
    min_lon_all = merged["min_lon"].min()
    max_lon_all = merged["max_lon"].max()
    min_lat_all = merged["min_lat"].min()
    max_lat_all = merged["max_lat"].max()

    ax.set_xlim(min_lon_all - 0.01, max_lon_all + 0.01)
    ax.set_ylim(min_lat_all - 0.01, max_lat_all + 0.01)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Tile-level Mean LST (°C) with Hotspot Highlight", fontsize=10)
    ax.tick_params(axis="both", labelsize=7)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean LST (°C)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved city heatmap to {out_path}")


if __name__ == "__main__":
    main()
