import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml


def load_config(path: str) -> Dict[str, Any]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found at '{path}'.")
    with path_obj.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("dataset_builder")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def save_metadata(metadata: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def infer_schema(df: pd.DataFrame) -> Dict[str, str]:
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def compute_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_desc = df.describe(include="number").to_dict()
    return {"numeric_summary": numeric_desc}

