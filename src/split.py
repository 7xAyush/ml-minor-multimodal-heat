from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.model_selection import train_test_split


logger = logging.getLogger("dataset_builder")


def create_splits(labels_df: pd.DataFrame, config: Dict[str, Any]) -> None:
    splits_cfg = config["splits"]
    pre_cfg = config["preprocessing"]
    label_col = pre_cfg["label_column"]
    image_id_col = pre_cfg["image_id_column"]

    train_ratio = splits_cfg["train_ratio"]
    val_ratio = splits_cfg["val_ratio"]
    test_ratio = splits_cfg["test_ratio"]
    random_state = splits_cfg.get("random_state", 42)

    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("Train/val/test ratios must sum to 1.")

    stratify_labels = labels_df[label_col] if label_col in labels_df.columns else None

    # Decide whether to use stratification for the first split
    use_stratify_first = False
    if stratify_labels is not None:
        counts = stratify_labels.value_counts()
        if len(labels_df) < 20:
            logger.warning(
                "Dataset has fewer than 20 samples; using random splits without stratification."
            )
        elif (counts < 2).any():
            logger.warning(
                "Some classes have fewer than 2 samples; using random splits without stratification."
            )
        else:
            use_stratify_first = True

    # First split train vs temp (val+test)
    try:
        train_df, temp_df = train_test_split(
            labels_df,
            test_size=val_ratio + test_ratio,
            random_state=random_state,
            stratify=stratify_labels if use_stratify_first else None,
        )
    except ValueError as exc:
        logger.warning(
            "Stratified train/temp split failed (%s); retrying without stratification.",
            exc,
        )
        train_df, temp_df = train_test_split(
            labels_df,
            test_size=val_ratio + test_ratio,
            random_state=random_state,
            stratify=None,
        )

    # Decide whether to use stratification for the second split
    use_stratify_second = False
    if label_col in temp_df.columns and len(temp_df) >= 20:
        temp_counts = temp_df[label_col].value_counts()
        if (temp_counts >= 2).all():
            use_stratify_second = True

    # Then split temp into val and test
    relative_test_size = test_ratio / (val_ratio + test_ratio)
    stratify_temp = temp_df[label_col] if use_stratify_second else None
    try:
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_size,
            random_state=random_state,
            stratify=stratify_temp,
        )
    except ValueError as exc:
        logger.warning(
            "Stratified val/test split failed (%s); retrying without stratification.",
            exc,
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_size,
            random_state=random_state,
            stratify=None,
        )

    dataset_root = Path(config["paths"]["dataset_root"])
    splits_dir = dataset_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_df[[image_id_col, label_col]].to_csv(splits_dir / "train.csv", index=False)
    val_df[[image_id_col, label_col]].to_csv(splits_dir / "val.csv", index=False)
    test_df[[image_id_col, label_col]].to_csv(splits_dir / "test.csv", index=False)

    logger.info(
        "Created splits: train=%d, val=%d, test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
