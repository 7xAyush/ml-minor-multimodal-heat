from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import logging

import pandas as pd

from .utils import ensure_dir, save_metadata, load_config, setup_logging
from . import download, clean, label, preprocess, split


def _load_real_satellite_metadata(config: Dict[str, Any]) -> Tuple[pd.DataFrame, Path]:
    """
    Load user-provided satellite metadata and raw images for REAL mode.

    Expected CSV (paths.satellite_metadata_csv) columns:
      - image_id
      - date        (YYYY-MM-DD)
      - lst         (LST in Celsius, physically meaningful)
      - tile_id     (aligned with the tile grid used for weather)
    """
    paths_cfg = config["paths"]
    meta_csv = Path(paths_cfg["satellite_metadata_csv"])
    raw_img_dir = Path(paths_cfg["raw_satellite_dir"])

    if not meta_csv.exists():
        raise FileNotFoundError(
            f"REAL mode expects an existing satellite metadata CSV at '{meta_csv}'. "
            "This file must contain real (non-synthetic) LST labels and tile assignments."
        )
    if not raw_img_dir.exists():
        raise FileNotFoundError(
            f"REAL mode expects a directory of raw satellite image chips at '{raw_img_dir}'."
        )

    df = pd.read_csv(meta_csv)
    required_cols = ["image_id", "date", "lst", "tile_id"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Satellite metadata CSV at '{meta_csv}' is missing required columns: {missing}"
        )

    # Basic type normalization
    df["image_id"] = df["image_id"].astype(str)
    df["date"] = df["date"].astype(str)
    df["tile_id"] = df["tile_id"].astype(str)

    return df, raw_img_dir


def build_real_dataset(config: Dict[str, Any], logger: logging.Logger | None = None) -> None:
    """
    Build a REAL (non-synthetic) multimodal dataset.

    Assumptions:
      - LST values and tile assignments are provided by the user in satellite_metadata_csv.
      - No synthetic brightness-based labels are generated in this path.
      - Weather is obtained via Open-Meteo or user CSV and aligned by (date, tile_id).
      - Urban proxy features are optional and can be added later; the minimal real
        version uses image + weather only to avoid synthetic urban constants.
    """
    if logger is None:
        logger = setup_logging()

    logger.info("Running REAL data pipeline (no synthetic label generation).")

    # Prepare directories
    base_dataset_dir = Path(config["paths"]["dataset_root"])
    images_out_dir = base_dataset_dir / "images"
    splits_out_dir = base_dataset_dir / "splits"
    ensure_dir(base_dataset_dir)
    ensure_dir(images_out_dir)
    ensure_dir(splits_out_dir)

    # 1. Load real satellite metadata (provided by the user)
    sat_meta_df, raw_image_dir = _load_real_satellite_metadata(config)
    logger.info("Loaded real satellite metadata with %d rows", len(sat_meta_df))

    # 2. Load/download weather data
    logger.info("Step 1/5 (REAL): Loading or downloading weather data")
    weather_df = download.download_weather(config)
    logger.info("Weather table has %d rows before cleaning", len(weather_df))

    # 3. Clean weather + validate images
    logger.info("Step 2/5 (REAL): Cleaning weather and validating images")
    weather_df = clean.clean_weather(weather_df, config)
    sat_meta_df, valid_image_ids, image_stats = clean.validate_and_filter_images(
        sat_meta_df, raw_image_dir, images_out_dir, config
    )

    # 4. Align satellite + weather by date and tile_id (no synthetic urban proxy here)
    logger.info("Step 3/5 (REAL): Aligning satellite metadata with weather by ['date', 'tile_id']")
    join_keys = ["date", "tile_id"]
    aligned_df = sat_meta_df.merge(weather_df, on=join_keys, how="inner")
    if aligned_df.empty:
        raise ValueError(
            "Aligned dataset is empty after joining satellite metadata with weather on "
            f"{join_keys}. Check that your tile_id/date values match between the two sources."
        )
    logger.info("Aligned dataset has %d rows after join", len(aligned_df))

    # 5. Label into heat-risk classes from REAL LST
    logger.info("Step 4/5 (REAL): Labelling samples based on real LST")
    labelled_df = label.add_heat_risk_labels(aligned_df, config)

    # Filter to samples with valid processed images only
    image_id_col = config["preprocessing"]["image_id_column"]
    labelled_df = labelled_df[labelled_df[image_id_col].isin(valid_image_ids)].reset_index(drop=True)
    logger.info(
        "After filtering to valid processed images, %d samples remain in REAL mode",
        len(labelled_df),
    )

    # 6. Preprocessing (images + tabular) and splits
    logger.info("Step 5/5 (REAL): Preprocessing and creating splits")
    tabular_df, labels_df, preprocess_stats = preprocess.preprocess_all(
        labelled_df, images_out_dir, config
    )
    train_count, val_count, test_count = split.create_splits(labels_df, config)

    # Save main tabular + labels
    tabular_path = base_dataset_dir / "tabular.csv"
    labels_path = base_dataset_dir / "labels.csv"
    tabular_df.to_csv(tabular_path, index=False)
    labels_df.to_csv(labels_path, index=False)

    # Build metadata and report
    metadata: Dict[str, Any] = {
        "mode": {
            "data_mode": "real",
            "lst_source": config.get("sources", {}).get("lst", "unknown"),
            "image_source": config.get("sources", {}).get("imagery", "unknown"),
            "weather_source": config.get("sources", {}).get("weather", "open_meteo"),
            "urban_source": config.get("sources", {}).get("urban", "none"),
        },
        "paths": {
            "dataset_root": str(base_dataset_dir),
            "images_dir": str(images_out_dir),
            "tabular_csv": str(tabular_path),
            "labels_csv": str(labels_path),
            "splits_dir": str(splits_out_dir),
        },
        "image_stats": image_stats,
        "preprocess_stats": preprocess_stats,
        "split_sizes": {
            "train": int(train_count),
            "val": int(val_count),
            "test": int(test_count),
        },
    }
    metadata.update(clean.compute_dataset_report(tabular_df, labels_df, image_stats))

    metadata_path = base_dataset_dir / "metadata.json"
    save_metadata(metadata, metadata_path)

    # Human-readable report
    clean.print_dataset_report(metadata)
