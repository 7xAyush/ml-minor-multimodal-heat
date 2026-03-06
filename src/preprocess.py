from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


logger = logging.getLogger("dataset_builder")


def preprocess_tabular(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    pre_cfg = config["preprocessing"]
    image_id_col = pre_cfg["image_id_column"]
    label_col = pre_cfg["label_column"]
    lst_col = pre_cfg["lst_column"]
    exclude = set(pre_cfg.get("exclude_from_scaling", [])) | {image_id_col, label_col, lst_col}

    # Separate identifier/label columns
    id_label_cols = [c for c in [image_id_col, label_col, lst_col] if c in df.columns]

    # Identify categorical vs numeric
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
    categorical_cols = [
        c
        for c in df.columns
        if c not in numeric_cols and c not in id_label_cols and c not in exclude
    ]

    logger.info(
        "Preprocessing tabular data. Numeric cols: %s; categorical cols: %s",
        numeric_cols,
        categorical_cols,
    )

    # Scale numeric
    scaler = MinMaxScaler()
    numeric_scaled = pd.DataFrame(
        scaler.fit_transform(df[numeric_cols]),
        columns=numeric_cols,
        index=df.index,
    )

    # Encode categoricals
    if categorical_cols:
        categoricals_encoded = pd.get_dummies(df[categorical_cols].astype("category"))
    else:
        categoricals_encoded = pd.DataFrame(index=df.index)

    # Concatenate all features
    feature_df = pd.concat([numeric_scaled, categoricals_encoded], axis=1)
    # Reattach id/label/lst
    for col in id_label_cols:
        feature_df[col] = df[col]

    stats = {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "feature_columns": list(feature_df.columns),
        "scaler_min_": scaler.data_min_.tolist() if hasattr(scaler, "data_min_") else None,
        "scaler_max_": scaler.data_max_.tolist() if hasattr(scaler, "data_max_") else None,
    }
    return feature_df, stats


def preprocess_all(
    df: pd.DataFrame,
    images_dir: Path,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    pre_cfg = config["preprocessing"]
    image_id_col = pre_cfg["image_id_column"]
    label_col = pre_cfg["label_column"]
    lst_col = pre_cfg["lst_column"]

    if image_id_col not in df.columns:
        raise KeyError(f"image_id column '{image_id_col}' not found in dataframe.")

    # Tabular preprocessing
    features_df, tab_stats = preprocess_tabular(df, config)

    # Labels dataframe: image_id, tile_id, date, label_class, lst
    required_cols = ["tile_id", "date"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Required columns for labels.csv missing from dataframe: {missing}"
        )
    labels_df = df[[image_id_col, "tile_id", "date", label_col, lst_col]].copy()

    # Sanity: ensure each image has a corresponding processed image file
    missing_images = []
    for img_id in labels_df[image_id_col].astype(str).unique():
        img_path = images_dir / f"{img_id}.png"
        if not img_path.exists():
            missing_images.append(img_id)
    if missing_images:
        logger.warning(
            "There are %d image_ids without processed image files. "
            "They will still appear in tabular/labels but images may be missing.",
            len(missing_images),
        )

    preprocess_stats = {
        "tabular": tab_stats,
        "num_samples": int(len(labels_df)),
        "num_missing_processed_images": int(len(missing_images)),
    }
    return features_df, labels_df, preprocess_stats
