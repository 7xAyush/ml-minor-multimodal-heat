from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .utils import ensure_dir


logger = logging.getLogger("dataset_builder")


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def discover_images(root: Path) -> List[Path]:
    """Recursively discover image files under the given root."""
    if not root.exists():
        raise FileNotFoundError(
            f"Kaggle source directory '{root}' does not exist. "
            "Please unzip your Kaggle dataset there or disable mode.use_kaggle_source."
        )
    images: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    if not images:
        raise FileNotFoundError(
            f"No image files found under '{root}'. "
            "Supported extensions: .jpg, .jpeg, .png, .tif, .tiff"
        )
    logger.info("Discovered %d images under %s", len(images), root)
    return images


def safe_image_id(path: Path, seen: Dict[str, int]) -> str:
    """Return a filesystem-safe, unique image_id derived from filename."""
    base = path.stem
    if base not in seen:
        seen[base] = 1
        return base
    # Duplicate base: append short hash of full path or counter
    seen[base] += 1
    h = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{base}_{h}"


def build_satellite_images_and_metadata(
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Path]:
    paths_cfg = config["paths"]
    kaggle_root = Path(paths_cfg["kaggle_source_dir"])
    out_dir = Path(paths_cfg["raw_satellite_dir"])
    meta_csv = Path(paths_cfg["satellite_metadata_csv"])

    ensure_dir(out_dir)

    images = discover_images(kaggle_root)
    seen_ids: Dict[str, int] = {}

    size_cfg = config["images"]["size"]
    target_w, target_h = int(size_cfg[0]), int(size_cfg[1])

    # Full list of dates in configured range
    dates = pd.date_range(
        config["time"]["start_date"],
        config["time"]["end_date"],
        freq="D",
    ).strftime("%Y-%m-%d").tolist()

    records = []
    warned = False

    for idx, img_path in enumerate(images):
        image_id = safe_image_id(img_path, seen_ids)
        out_path = out_dir / f"{image_id}.png"

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img = img.resize((target_w, target_h))
                img.save(out_path, format="PNG")
                arr = np.array(img, dtype=np.float32)
        except Exception as exc:
            logger.warning("Skipping image %s due to error: %s", img_path, exc)
            continue

        brightness = float(arr.mean()) / 255.0
        random_variation = float(np.random.normal(0.0, 1.5))
        lst = 26.0 + 18.0 * brightness + random_variation
        lst = float(np.clip(lst, 20.0, 50.0))

        # Cycle over dates and 100 tiles to align with weather/urban
        date = dates[idx % len(dates)]
        tile_id = f"tile_r0_c{idx % 100}"

        records.append(
            {
                "image_id": image_id,
                "date": date,
                "lst": lst,
                "tile_id": tile_id,
            }
        )

        if not warned:
            logger.warning(
                "LST is synthetic demo label from brightness; "
                "replace with real LST for valid heat-risk training."
            )
            warned = True

    if not records:
        raise RuntimeError("No valid images were ingested from Kaggle source.")

    df = pd.DataFrame(records, columns=["image_id", "date", "lst", "tile_id"])
    ensure_dir(meta_csv.parent)
    df.to_csv(meta_csv, index=False)
    logger.info(
        "Saved synthetic satellite metadata with %d rows to %s", len(df), meta_csv
    )

    return df, out_dir
