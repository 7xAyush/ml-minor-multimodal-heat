from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import requests

from .utils import ensure_dir, read_csv_if_exists
from . import grid, kaggle_ingest


logger = logging.getLogger("dataset_builder")


def download_weather(config: Dict[str, Any]) -> pd.DataFrame:
    paths_cfg = config["paths"]
    weather_csv = Path(paths_cfg["weather_csv"])
    if weather_csv.exists():
        logger.info("Loading existing weather CSV from %s", weather_csv)
        df = pd.read_csv(weather_csv)
        # Ensure daily aggregation if user provided hourly data
        if "time" in df.columns and "date" not in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df["date"] = df["time"].dt.date.astype(str)
        if "date" in df.columns and "tile_id" in df.columns:
            non_key_cols = [
                c
                for c in df.columns
                if c not in ["date", "tile_id"] and pd.api.types.is_numeric_dtype(df[c])
            ]
            df_daily = (
                df.groupby(["date", "tile_id"], as_index=False)[non_key_cols].mean()
            )
            logger.info(
                "Aggregated provided hourly weather to daily level (%d rows)",
                len(df_daily),
            )
            return df_daily
        return df

    download_csv = Path(paths_cfg["weather_download_csv"])
    ensure_dir(download_csv.parent)

    city_cfg = config["city"]
    time_cfg = config["time"]

    # Tile-based weather download
    tiles_path = download_csv.parent / "tiles.csv"
    if tiles_path.exists():
        tiles_df = pd.read_csv(tiles_path)
        logger.info("Loaded tiles from %s", tiles_path)
    else:
        logger.info("Tiles file not found; generating grid tiles.")
        tiles_df = grid.generate_tiles(config)

    all_daily_records = []
    url = "https://archive-api.open-meteo.com/v1/archive"

    for _, tile in tiles_df.iterrows():
        lat = tile["lat"]
        lon = tile["lon"]
        tile_id = tile["tile_id"]

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": time_cfg["start_date"],
            "end_date": time_cfg["end_date"],
            "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m",
            "timezone": "UTC",
        }

        logger.info(
            "Downloading weather for tile %s at (%.4f, %.4f)", tile_id, lat, lon
        )
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly", {})
        if "time" not in hourly:
            logger.warning(
                "Skipping tile %s due to missing 'time' in Open-Meteo response", tile_id
            )
            continue

        df_tile = pd.DataFrame(hourly)
        df_tile["time"] = pd.to_datetime(df_tile["time"])
        df_tile["date"] = df_tile["time"].dt.date.astype(str)
        df_tile["tile_id"] = tile_id
        all_daily_records.append(df_tile)

    if not all_daily_records:
        raise RuntimeError("No weather data downloaded for any tile.")

    df_all = pd.concat(all_daily_records, ignore_index=True)
    # Aggregate hourly -> daily per tile
    numeric_cols = [
        c
        for c in df_all.columns
        if c not in ["time", "date", "tile_id"]
        and pd.api.types.is_numeric_dtype(df_all[c])
    ]
    df_daily = (
        df_all.groupby(["date", "tile_id"], as_index=False)[numeric_cols].mean()
    )

    df_daily.to_csv(download_csv, index=False)
    logger.info("Saved downloaded daily weather to %s (%d rows)", download_csv, len(df_daily))
    return df_daily


def download_or_load_urban_proxy(config: Dict[str, Any]) -> pd.DataFrame:
    paths_cfg = config["paths"]
    urban_csv = Path(paths_cfg["urban_proxy_csv"])

    df = read_csv_if_exists(urban_csv)
    if df is not None:
        logger.info("Loaded urban proxy CSV from %s", urban_csv)
        return expand_urban_to_daily(df, config)

    # Auto-create minimal urban proxy from tiles
    tiles_path = urban_csv.parent / "tiles.csv"
    if tiles_path.exists():
        tiles_df = pd.read_csv(tiles_path)
        logger.info("Loaded tiles from %s to build urban proxy", tiles_path)
    else:
        logger.info("Tiles file not found; generating grid tiles for urban proxy.")
        tiles_df = grid.generate_tiles(config)

    # Simple demo proxy: constant population density
    tiles_df = tiles_df[["tile_id"]].copy()
    tiles_df["pop_density"] = 10000

    ensure_dir(urban_csv.parent)
    tiles_df.to_csv(urban_csv, index=False)
    logger.info(
        "Generated synthetic urban proxy (constant pop_density) and saved to %s",
        urban_csv,
    )
    return expand_urban_to_daily(tiles_df, config)


def expand_urban_to_daily(urban_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Expand static per-tile urban proxy to daily time series."""
    if "tile_id" not in urban_df.columns:
        raise KeyError("Urban proxy dataframe must contain 'tile_id' column.")

    time_cfg = config["time"]
    dates = pd.date_range(
        start=time_cfg["start_date"],
        end=time_cfg["end_date"],
        freq="D",
    )
    date_str = dates.strftime("%Y-%m-%d")

    tiles = urban_df["tile_id"].unique()
    # Cross join tiles x dates
    cross = pd.MultiIndex.from_product(
        [tiles, date_str], names=["tile_id", "date"]
    ).to_frame(index=False)

    merged = cross.merge(urban_df, on="tile_id", how="left")
    logger.info(
        "Expanded urban proxy to daily: %d tiles x %d days = %d rows",
        len(tiles),
        len(date_str),
        len(merged),
    )
    return merged


def load_satellite_metadata_and_images(
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Path]:
    mode_cfg = config.get("mode", {})
    if mode_cfg.get("use_kaggle_source", False):
        logger.info("Using Kaggle source ingestion flow for satellite data.")
        return kaggle_ingest.build_satellite_images_and_metadata(config)

    paths_cfg = config["paths"]
    meta_csv = Path(paths_cfg["satellite_metadata_csv"])
    raw_img_dir = Path(paths_cfg["raw_satellite_dir"])

    if not meta_csv.exists():
        raise FileNotFoundError(
            f"Satellite metadata CSV not found at '{meta_csv}'. "
            "Expected columns include image_id, date, lst and tile_id."
        )
    if not raw_img_dir.exists():
        raise FileNotFoundError(
            f"Raw satellite image directory not found at '{raw_img_dir}'."
        )

    df = pd.read_csv(meta_csv)
    logger.info(
        "Loaded satellite metadata (%d rows) and images from %s", len(df), raw_img_dir
    )
    return df, raw_img_dir
