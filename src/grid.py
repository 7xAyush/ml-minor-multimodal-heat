from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .utils import ensure_dir


logger = logging.getLogger("dataset_builder")


def generate_tiles(config: Dict[str, Any]) -> pd.DataFrame:
    city_cfg = config["city"]
    grid_cfg = config["grid"]
    bbox = city_cfg["bbox"]
    min_lat, min_lon, max_lat, max_lon = bbox

    tile_size_m = float(grid_cfg.get("tile_size_m", 1000))
    max_tiles = grid_cfg.get("max_tiles", None)

    lat_center = (min_lat + max_lat) / 2.0
    deg_lat = tile_size_m / 111320.0
    deg_lon = tile_size_m / (111320.0 * math.cos(math.radians(lat_center)))

    lat_values = []
    lon_values = []
    rows = 0
    cols = 0

    lat = min_lat
    while lat < max_lat:
        lat_values.append(lat)
        lat += deg_lat
    lon = min_lon
    while lon < max_lon:
        lon_values.append(lon)
        lon += deg_lon

    tiles = []
    for r, lat_min in enumerate(lat_values):
        lat_max = min(lat_min + deg_lat, max_lat)
        for c, lon_min in enumerate(lon_values):
            lon_max = min(lon_min + deg_lon, max_lon)
            tile_id = f"tile_r{r}_c{c}"
            tiles.append(
                {
                    "tile_id": tile_id,
                    "min_lat": lat_min,
                    "min_lon": lon_min,
                    "max_lat": lat_max,
                    "max_lon": lon_max,
                    "lat": (lat_min + lat_max) / 2.0,
                    "lon": (lon_min + lon_max) / 2.0,
                }
            )
        cols = max(cols, len(lon_values))
    rows = len(lat_values)

    df = pd.DataFrame(tiles)

    if max_tiles is not None:
        try:
            max_tiles_int = int(max_tiles)
            if max_tiles_int > 0 and max_tiles_int < len(df):
                df = df.head(max_tiles_int).reset_index(drop=True)
                logger.info("Limiting tiles to first %d tiles", max_tiles_int)
        except (TypeError, ValueError):
            pass

    tiles_path = Path("raw") / "tiles.csv"
    ensure_dir(tiles_path.parent)
    df.to_csv(tiles_path, index=False)
    logger.info("Generated %d tiles and saved to %s", len(df), tiles_path)
    return df
