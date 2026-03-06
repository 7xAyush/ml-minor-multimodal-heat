from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd


logger = logging.getLogger("dataset_builder")


def align_by_keys(
    sat_meta_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    urban_df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    logger.info("Aligning satellite + weather on ['date','tile_id']")
    merged = sat_meta_df.merge(
        weather_df,
        on=["date", "tile_id"],
        how="inner",
    )

    logger.info("Merging urban proxy on ['date','tile_id']")
    merged = merged.merge(
        urban_df,
        on=["date", "tile_id"],
        how="left",
    )

    if len(merged) == 0:
        raise ValueError("Aligned dataset empty after joins.")

    logger.info("Aligned dataset rows: %d", len(merged))
    return merged


def add_heat_risk_labels(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    cfg = config["preprocessing"]
    lst_col = cfg["lst_column"]
    label_col = cfg["label_column"]

    if lst_col not in df.columns:
        raise KeyError(
            f"LST column '{lst_col}' not found in aligned dataframe. "
            "Ensure your satellite metadata contains this column."
        )

    thresholds = config["heat_risk"]
    low = thresholds["low_threshold"]
    high = thresholds["high_threshold"]

    def label_from_lst(lst: float) -> int:
        if lst < low:
            return 0
        if lst < high:
            return 1
        return 2

    df[label_col] = df[lst_col].astype(float).apply(label_from_lst).astype(int)
    logger.info("Added heat-risk labels using LST with thresholds %.2f, %.2f", low, high)
    return df
