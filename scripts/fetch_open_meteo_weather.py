#!/usr/bin/env python
"""
Fetch REAL historical weather from Open-Meteo for all (date, tile_id) pairs
present in raw/satellite_metadata.csv and save to raw/weather_downloaded.csv.

This script is designed to work with the REAL pipeline in this repository.
It uses only public REAL data (Open-Meteo ERA5 archive) and does not generate
any synthetic weather values.

Expected inputs (default paths match config_real.yaml):
  - raw/satellite_metadata.csv
      columns (at least): image_id, date, lst, tile_id, lat, lon

Outputs:
  - raw/weather_downloaded.csv
      columns:
        - date
        - tile_id
        - temperature_2m
        - relativehumidity_2m
        - windspeed_10m
      plus any additional numeric variables returned by Open-Meteo.

Usage example:

  python scripts/fetch_open_meteo_weather.py \\
      --sat_meta raw/satellite_metadata.csv \\
      --output_csv raw/weather_downloaded.csv

The script:
  1) Reads satellite_metadata.csv and finds unique tiles and dates.
  2) For each tile, uses its (lat, lon) to request hourly ERA5 data from
     the Open-Meteo archive API over the date range needed for that tile.
  3) Aggregates hourly -> daily per (date, tile_id) by simple mean.
  4) Filters to the exact (date, tile_id) pairs present in satellite metadata.
  5) Writes raw/weather_downloaded.csv.
  6) Performs a coverage check and prints a summary, including provisional
     class balance (low/medium/high) based on LST thresholds.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import requests


OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch REAL weather from Open-Meteo for all (date, tile_id) "
        "pairs found in raw/satellite_metadata.csv."
    )
    parser.add_argument(
        "--sat_meta",
        type=str,
        default="raw/satellite_metadata.csv",
        help="Path to satellite metadata CSV (with at least image_id,date,lst,tile_id[,lat,lon]).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="raw/weather_downloaded.csv",
        help="Output CSV path for aggregated daily weather.",
    )
    parser.add_argument(
        "--low_threshold",
        type=float,
        default=30.0,
        help="LST threshold for low vs medium class (Celsius).",
    )
    parser.add_argument(
        "--high_threshold",
        type=float,
        default=35.0,
        help="LST threshold for medium vs high class (Celsius).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds for Open-Meteo API.",
    )
    return parser.parse_args()


def load_satellite_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Satellite metadata CSV not found at {path}")

    df = pd.read_csv(path)
    required = ["image_id", "date", "lst", "tile_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Satellite metadata CSV at {path} is missing required columns: {missing}"
        )

    # Normalize types
    df["image_id"] = df["image_id"].astype(str)
    df["tile_id"] = df["tile_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    return df


def build_tile_coordinates(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    Build mapping tile_id -> (lat, lon).

    Prefer lat/lon columns from satellite_metadata.csv (as produced by
    scripts/prepare_real_inputs.py from Earth Engine exports). If they are
    not present, fail loudly instead of guessing.
    """
    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError(
            "Satellite metadata does not contain 'lat' and 'lon' columns. "
            "For REAL weather fetching, please ensure Earth Engine metadata "
            "exports include tile centroid lat/lon and that "
            "scripts/prepare_real_inputs.py propagated them."
        )

    coords: Dict[str, Tuple[float, float]] = {}
    by_tile = df.groupby("tile_id")
    for tile_id, group in by_tile:
        lat_vals = group["lat"].dropna().astype(float).unique()
        lon_vals = group["lon"].dropna().astype(float).unique()
        if len(lat_vals) == 0 or len(lon_vals) == 0:
            raise ValueError(
                f"Missing lat/lon values for tile_id={tile_id} in satellite metadata."
            )
        # Use the first unique pair; all rows for a tile should share centroid.
        coords[tile_id] = (float(lat_vals[0]), float(lon_vals[0]))
    return coords


def fetch_hourly_for_tile(
    tile_id: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    timeout: float,
) -> pd.DataFrame:
    """
    Fetch hourly ERA5 weather for one tile over [start_date, end_date].
    Returns a DataFrame with columns: time, temperature_2m, relativehumidity_2m,
    windspeed_10m, date, tile_id.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m",
        "timezone": "UTC",
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    hourly = data.get("hourly", {})
    if "time" not in hourly:
        raise RuntimeError(
            f"Open-Meteo response for tile_id={tile_id} is missing 'hourly.time'. "
            f"Raw response keys: {list(data.keys())}"
        )

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.strftime("%Y-%m-%d")
    df["tile_id"] = tile_id
    return df


def aggregate_to_daily(df_all: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        c
        for c in df_all.columns
        if c not in ["time", "date", "tile_id"]
        and pd.api.types.is_numeric_dtype(df_all[c])
    ]
    if not numeric_cols:
        raise ValueError("No numeric weather variables found to aggregate.")

    df_daily = (
        df_all.groupby(["date", "tile_id"], as_index=False)[numeric_cols].mean()
    )
    return df_daily


def main() -> None:
    args = parse_args()

    sat_meta_path = Path(args.sat_meta)
    out_path = Path(args.output_csv)

    sat_df = load_satellite_metadata(sat_meta_path)
    print(f"[INFO] Loaded satellite metadata from {sat_meta_path} with {len(sat_df)} rows.")

    # Build tile -> (lat, lon) mapping.
    tile_coords = build_tile_coordinates(sat_df)
    print(f"[INFO] Found {len(tile_coords)} unique tiles with lat/lon.")

    # Determine date range per tile.
    sat_df["date_dt"] = pd.to_datetime(sat_df["date"])
    min_date_global = sat_df["date_dt"].min().strftime("%Y-%m-%d")
    max_date_global = sat_df["date_dt"].max().strftime("%Y-%m-%d")
    print(
        f"[INFO] Global satellite date range: {min_date_global} to {max_date_global}"
    )

    # For efficiency, we fetch per tile over its own min/max date range.
    all_hourly = []
    for tile_id, (lat, lon) in tile_coords.items():
        sat_tile = sat_df[sat_df["tile_id"] == tile_id]
        min_date = sat_tile["date_dt"].min().strftime("%Y-%m-%d")
        max_date = sat_tile["date_dt"].max().strftime("%Y-%m-%d")
        print(
            f"[INFO] Fetching weather for tile {tile_id} at ({lat:.4f}, {lon:.4f}) "
            f"from {min_date} to {max_date}"
        )
        try:
            df_tile = fetch_hourly_for_tile(
                tile_id=tile_id,
                lat=lat,
                lon=lon,
                start_date=min_date,
                end_date=max_date,
                timeout=args.timeout,
            )
        except Exception as exc:
            print(f"[WARN] Skipping tile {tile_id} due to fetch error: {exc}", file=sys.stderr)
            continue
        all_hourly.append(df_tile)

    if not all_hourly:
        print("[ERROR] No weather data fetched for any tile.", file=sys.stderr)
        sys.exit(1)

    hourly_all = pd.concat(all_hourly, ignore_index=True)
    print(
        f"[INFO] Combined hourly weather rows: {len(hourly_all)} "
        f"across {hourly_all['tile_id'].nunique()} tiles."
    )

    daily_all = aggregate_to_daily(hourly_all)
    print(
        f"[INFO] Aggregated to daily weather rows: {len(daily_all)} "
        f"across {daily_all['tile_id'].nunique()} tiles."
    )

    # Restrict to the (date, tile_id) pairs actually needed by satellite metadata.
    sat_pairs = set(zip(sat_df["date"], sat_df["tile_id"]))
    daily_all["date"] = daily_all["date"].astype(str)
    daily_pairs = set(zip(daily_all["date"], daily_all["tile_id"]))

    missing_pairs = sat_pairs - daily_pairs
    if missing_pairs:
        preview = list(missing_pairs)[:5]
        print(
            f"[WARN] Weather data missing for {len(missing_pairs)} (date, tile_id) pairs "
            f"present in satellite_metadata. First few missing: {preview}",
            file=sys.stderr,
        )

    mask_needed = daily_all.apply(
        lambda row: (row["date"], row["tile_id"]) in sat_pairs, axis=1
    )
    daily_needed = daily_all[mask_needed].reset_index(drop=True)

    if daily_needed.empty:
        print(
            "[ERROR] After filtering to satellite (date, tile_id) pairs, "
            "no weather rows remain. Check your satellite dates/tiles.",
            file=sys.stderr,
        )
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily_needed.to_csv(out_path, index=False)
    print(
        f"[INFO] Wrote daily weather for {len(daily_needed)} rows "
        f"to {out_path}."
    )

    # Quick coverage check.
    needed_covered = sat_pairs <= set(zip(daily_needed["date"], daily_needed["tile_id"]))
    if needed_covered:
        print("[OK] Weather CSV covers all (date, tile_id) pairs in satellite metadata.")
    else:
        print(
            "[WARN] Weather CSV does NOT cover all (date, tile_id) pairs "
            "in satellite metadata. Consider re-running or adjusting date range.",
            file=sys.stderr,
        )

    # Provisional class balance from LST thresholds on satellite metadata.
    low_t = args.low_threshold
    high_t = args.high_threshold
    lst_vals = pd.to_numeric(sat_df["lst"], errors="coerce")

    def label_from_lst(x: float) -> int:
        if pd.isna(x):
            return -1
        if x < low_t:
            return 0
        if x < high_t:
            return 1
        return 2

    labels = lst_vals.apply(label_from_lst)
    counts = labels.value_counts().to_dict()
    print("=== PROVISIONAL CLASS COUNTS FROM LST ===")
    print(f"  Low   (0, lst < {low_t:0.1f}): {counts.get(0, 0)}")
    print(f"  Medium(1, {low_t:0.1f} <= lst < {high_t:0.1f}): {counts.get(1, 0)}")
    print(f"  High  (2, lst >= {high_t:0.1f}): {counts.get(2, 0)}")
    print("=========================================")


if __name__ == "__main__":
    main()

