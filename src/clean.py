from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .utils import ensure_dir, compute_basic_stats, infer_schema


logger = logging.getLogger("dataset_builder")


def clean_weather(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Cleaning weather data, initial rows: %d", len(df))
    df = df.copy()
    # If hourly data present, aggregate to daily per tile
    if "time" in df.columns and "date" not in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df["date"] = df["time"].dt.date.astype(str)

    if "date" in df.columns and "tile_id" in df.columns:
        non_key_cols = [
            c
            for c in df.columns
            if c not in ["time", "date", "tile_id"]
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        df = (
            df.groupby(["date", "tile_id"], as_index=False)[non_key_cols].mean()
        )

    df = df.drop_duplicates()
    df = df.dropna(subset=["date"])
    # Basic outlier clipping for key weather variables if present
    if "temperature_2m" in df.columns:
        df["temperature_2m"] = df["temperature_2m"].clip(lower=-40, upper=60)
    if "relativehumidity_2m" in df.columns:
        df["relativehumidity_2m"] = df["relativehumidity_2m"].clip(lower=0, upper=100)
    if "windspeed_10m" in df.columns:
        df["windspeed_10m"] = df["windspeed_10m"].clip(lower=0, upper=60)
    logger.info("Weather data after cleaning: %d rows", len(df))
    return df


def clean_urban(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Cleaning urban proxy data, initial rows: %d", len(df))
    df = df.copy()
    df = df.drop_duplicates()
    if "tile_id" not in df.columns:
        raise KeyError(
            "Urban proxy dataframe must contain 'tile_id' column."
        )
    df = df.dropna(subset=["tile_id"])
    # Fill numeric NaNs with median
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    logger.info("Urban proxy data after cleaning: %d rows", len(df))
    return df


def _image_hash(path: Path) -> str:
    with path.open("rb") as f:
        data = f.read()
    return hashlib.md5(data).hexdigest()


def _is_image_corrupt(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        # OpenCV second check
        arr = cv2.imread(str(path))
        if arr is None:
            return True
        return False
    except Exception:
        return True


def validate_and_filter_images(
    sat_meta_df: pd.DataFrame,
    raw_image_dir: Path,
    images_out_dir: Path,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    ensure_dir(images_out_dir)
    join_cfg = config["preprocessing"]
    image_id_col = join_cfg["image_id_column"]

    sat_meta_df = sat_meta_df.copy()

    mode_cfg = config.get("mode", {})
    data_mode = mode_cfg.get("data_mode", "synthetic")
    # In REAL mode we must preserve one row per (image_id, date, tile_id),
    # even if multiple rows share identical pixel content (e.g. static chips
    # for multiple dates of the same tile). Hash-based de-duplication is
    # therefore disabled for REAL data to avoid collapsing temporal samples.
    dedup_by_hash = data_mode != "real"

    valid_image_ids: List[str] = []
    hash_seen: Dict[str, str] = {}
    num_corrupt = 0
    num_missing = 0
    num_duplicates = 0

    size = config["images"]["size"]
    target_w, target_h = int(size[0]), int(size[1])

    logger.info("Validating and resizing satellite images...")

    for _, row in tqdm(sat_meta_df.iterrows(), total=len(sat_meta_df)):
        img_id = str(row[image_id_col])
        # Accept various common extensions; try in order
        candidates = [
            raw_image_dir / f"{img_id}.png",
            raw_image_dir / f"{img_id}.jpg",
            raw_image_dir / f"{img_id}.jpeg",
            raw_image_dir / f"{img_id}.tif",
        ]
        img_path = next((p for p in candidates if p.exists()), None)
        if img_path is None:
            num_missing += 1
            continue

        if _is_image_corrupt(img_path):
            num_corrupt += 1
            continue

        # Hash-based de-duplication is only applied when explicitly enabled
        # (currently for synthetic/Kaggle modes). In REAL mode we still count
        # duplicates but keep all rows to avoid collapsing per-date samples.
        hsh = _image_hash(img_path)
        if hsh in hash_seen:
            num_duplicates += 1
            if dedup_by_hash:
                continue
        else:
            hash_seen[hsh] = img_id

        # Load, resize and save to images_out_dir as PNG
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            num_corrupt += 1
            continue

        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        # Keep uint8; normalization should happen at training time
        if resized.dtype != np.uint8:
            resized = np.clip(resized, 0, 255).astype(np.uint8)

        out_path = images_out_dir / f"{img_id}.png"
        cv2.imwrite(str(out_path), resized)

        valid_image_ids.append(img_id)

    filtered_meta = sat_meta_df[sat_meta_df[image_id_col].isin(valid_image_ids)].reset_index(drop=True)

    image_stats = {
        "total_metadata_rows": int(len(sat_meta_df)),
        "valid_images": int(len(valid_image_ids)),
        "missing_images": int(num_missing),
        "corrupt_images": int(num_corrupt),
        "duplicate_images": int(num_duplicates),
        "output_image_size": [target_w, target_h],
    }

    logger.info("Image validation complete. Valid images: %d", len(valid_image_ids))
    return filtered_meta, valid_image_ids, image_stats


def compute_dataset_report(
    tabular_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    image_stats: Dict[str, Any],
) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    report["total_samples"] = int(len(labels_df))
    report["class_distribution"] = (
        labels_df["label_class"].value_counts().to_dict()
        if "label_class" in labels_df.columns
        else {}
    )
    report["tabular_schema"] = infer_schema(tabular_df)
    report["tabular_stats"] = compute_basic_stats(tabular_df)
    report["image_stats"] = image_stats
    report["missing_values"] = tabular_df.isna().sum().to_dict()
    return report


def print_dataset_report(metadata: Dict[str, Any]) -> None:
    logger.info("======== DATASET REPORT ========")
    logger.info("Total samples: %s", metadata.get("total_samples"))
    logger.info("Class distribution: %s", metadata.get("class_distribution"))
    logger.info("Missing values (tabular): %s", metadata.get("missing_values"))
    img_stats = metadata.get("image_stats", {})
    logger.info(
        "Images - valid: %(valid_images)s, missing: %(missing_images)s, "
        "corrupt: %(corrupt_images)s, duplicates: %(duplicate_images)s",
        img_stats,
    )
    logger.info("Tabular numeric stats: %s", metadata.get("tabular_stats"))
    logger.info("======== END OF REPORT ========")
