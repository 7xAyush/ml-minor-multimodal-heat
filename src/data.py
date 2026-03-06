from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("multimodal_data")


class MultimodalHeatDataset(Dataset):
    """
    Dataset that yields (image_tensor, tabular_tensor, label_tensor).

    Dataframe `df` must contain:
      - image_id
      - label_class
      - tabular feature columns
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Path,
        feature_cols: List[str],
        transform: Optional[Callable] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.feature_cols = feature_cols
        self.transform = transform

        if "image_id" not in self.df.columns or "label_class" not in self.df.columns:
            raise ValueError("Dataframe must contain 'image_id' and 'label_class' columns.")

        self.image_ids = self.df["image_id"].astype(str).tolist()
        self.labels = self.df["label_class"].to_numpy(dtype=np.int64)
        self.tabular = self.df[self.feature_cols].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        img_path = self.images_dir / f"{image_id}.png"

        if not img_path.exists():
            logger.warning("Missing image file for image_id=%s at %s", image_id, img_path)
            return None

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                if self.transform is not None:
                    image_tensor = self.transform(img)
                else:
                    arr = np.array(img, dtype=np.float32) / 255.0
                    image_tensor = torch.from_numpy(arr).permute(2, 0, 1)
        except Exception as exc:
            logger.warning("Error loading image %s: %s", img_path, exc)
            return None

        tab = torch.from_numpy(self.tabular[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image_tensor, tab, label


def multimodal_collate_fn(
    batch: Sequence[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]
):
    # Filter out any None samples (e.g., missing/corrupt images)
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    images, tabular, labels = zip(*batch)
    images_tensor = torch.stack(images, dim=0)
    tabular_tensor = torch.stack(tabular, dim=0)
    labels_tensor = torch.stack(labels, dim=0)
    return images_tensor, tabular_tensor, labels_tensor
