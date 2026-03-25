from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


logger = logging.getLogger("dataset_builder")


def create_splits(labels_df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Create train/val/test splits and write them to disk.

    For very small datasets, this function uses safe fallbacks instead of
    letting scikit-learn raise confusing errors:

    - If val_ratio + test_ratio == 0.0:
        All samples are assigned to train; val/test are empty.
    - If n_samples < 3 and (val_ratio + test_ratio) > 0.0:
        All samples are assigned to train; val/test are empty, with a clear warning.

    Returns
    -------
    (n_train, n_val, n_test)
    """
    splits_cfg = config["splits"]
    pre_cfg = config["preprocessing"]
    label_col = pre_cfg["label_column"]
    image_id_col = pre_cfg["image_id_column"]

    train_ratio = float(splits_cfg["train_ratio"])
    val_ratio = float(splits_cfg["val_ratio"])
    test_ratio = float(splits_cfg["test_ratio"])
    random_state = splits_cfg.get("random_state", 42)

    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("Train/val/test ratios must sum to 1.")

    n_samples = len(labels_df)
    if n_samples == 0:
        raise ValueError("Cannot create splits: labels_df is empty.")

    # Handle degenerate case where user disables val/test entirely.
    if (val_ratio + test_ratio) == 0.0:
        logger.info(
            "val_ratio and test_ratio are both 0.0; assigning all %d samples "
            "to the training split and leaving val/test empty.",
            n_samples,
        )
        dataset_root = Path(config["paths"]["dataset_root"])
        splits_dir = dataset_root / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)

        train_df = labels_df.copy()
        empty = labels_df.iloc[0:0].copy()
        train_df[[image_id_col, label_col]].to_csv(splits_dir / "train.csv", index=False)
        empty[[image_id_col, label_col]].to_csv(splits_dir / "val.csv", index=False)
        empty[[image_id_col, label_col]].to_csv(splits_dir / "test.csv", index=False)

        logger.info(
            "Created degenerate splits (train-only): train=%d, val=%d, test=%d",
            len(train_df),
            len(empty),
            len(empty),
        )
        return len(train_df), 0, 0

    # For smoke tests with extremely small datasets, avoid confusing sklearn errors.
    if n_samples < 3:
        logger.warning(
            "Dataset has only %d samples; cannot create meaningful train/val/test "
            "splits with ratios train=%.2f, val=%.2f, test=%.2f. "
            "All samples will be assigned to the training split; val/test will be "
            "empty. Increase the dataset to at least 3 samples for full splitting.",
            n_samples,
            train_ratio,
            val_ratio,
            test_ratio,
        )
        dataset_root = Path(config["paths"]["dataset_root"])
        splits_dir = dataset_root / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)

        train_df = labels_df.copy()
        empty = labels_df.iloc[0:0].copy()
        train_df[[image_id_col, label_col]].to_csv(splits_dir / "train.csv", index=False)
        empty[[image_id_col, label_col]].to_csv(splits_dir / "val.csv", index=False)
        empty[[image_id_col, label_col]].to_csv(splits_dir / "test.csv", index=False)

        logger.info(
            "Created tiny-data splits: train=%d, val=%d, test=%d",
            len(train_df),
            len(empty),
            len(empty),
        )
        return len(train_df), 0, 0

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
    if len(temp_df) < 2:
        # Not enough samples to form separate val and test; assign all temp to val,
        # leave test empty. This keeps smoke tests from crashing while making the
        # limitation explicit.
        logger.warning(
            "Temporary (val+test) split has only %d samples; assigning all to the "
            "validation split and leaving the test split empty. Increase the "
            "dataset to obtain a non-empty test split.",
            len(temp_df),
        )
        val_df = temp_df.copy()
        test_df = temp_df.iloc[0:0].copy()
    else:
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
    return len(train_df), len(val_df), len(test_df)
