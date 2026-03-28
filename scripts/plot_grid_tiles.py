#!/usr/bin/env python
"""
Generate a clean, research-style visualization of a city divided into grid tiles.

The figure shows:
  - A map-like background (stylized urban area)
  - A square grid of tiles
  - 2–3 highlighted tiles in a different color
  - Tile labels: tile_id and approximate coordinates

The figure is saved as:
  docs/grid_tiles.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main() -> None:
    # Simple synthetic "city" bounding box in lat/lon
    min_lat, max_lat = 12.0, 13.0
    min_lon, max_lon = 77.0, 78.0

    # Grid resolution (rows x cols)
    n_rows, n_cols = 4, 4

    # Figure setup
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Draw a light gray "urban area" background
    city_rect = Rectangle(
        (min_lon, min_lat),
        max_lon - min_lon,
        max_lat - min_lat,
        facecolor="#F5F5F5",
        edgecolor="#CCCCCC",
        linewidth=1.0,
    )
    ax.add_patch(city_rect)

    # Compute tile sizes
    tile_height = (max_lat - min_lat) / n_rows
    tile_width = (max_lon - min_lon) / n_cols

    # Choose a few tiles to highlight
    highlight_tiles = {(0, 0), (1, 2), (3, 3)}

    for r in range(n_rows):
        for c in range(n_cols):
            tile_min_lat = min_lat + r * tile_height
            tile_min_lon = min_lon + c * tile_width

            tile_id = f"tile_r{r}_c{c}"
            centroid_lat = tile_min_lat + tile_height / 2.0
            centroid_lon = tile_min_lon + tile_width / 2.0

            # Highlight selected tiles
            if (r, c) in highlight_tiles:
                face = "#E0ECF8"  # light blue
                edge = "#336699"
                lw = 1.5
            else:
                face = "none"
                edge = "#BBBBBB"
                lw = 1.0

            rect = Rectangle(
                (tile_min_lon, tile_min_lat),
                tile_width,
                tile_height,
                facecolor=face,
                edgecolor=edge,
                linewidth=lw,
            )
            ax.add_patch(rect)

            # Label only highlighted tiles for readability
            if (r, c) in highlight_tiles:
                label = (
                    f"{tile_id}\n"
                    f"({centroid_lat:.3f}, {centroid_lon:.3f})"
                )
                ax.text(
                    centroid_lon,
                    centroid_lat,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#222222",
                )

    # Styling
    ax.set_xlim(min_lon - 0.02, max_lon + 0.02)
    ax.set_ylim(min_lat - 0.02, max_lat + 0.02)
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.set_title("City Tile Grid (Illustrative)", fontsize=10)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_aspect("equal", adjustable="box")

    out_dir = Path("docs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grid_tiles.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid tiles visualization to {out_path}")


if __name__ == "__main__":
    main()

