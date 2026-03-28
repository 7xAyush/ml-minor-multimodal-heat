#!/usr/bin/env python
"""
Create a figure showing sample satellite image patches.

Requirements:
- Display 4 image tiles in a 2x2 grid.
- Each tile represents an urban region (in practice, taken from existing chips).
- Add small labels (optional).

The script tries to load real chips from:
  1. dataset/images/
  2. raw/satellite_images/

If no images are found, it falls back to synthetic patches.

Output:
  docs/sample_patches.png
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def find_image_paths() -> List[Path]:
    candidates: List[Path] = []
    for base in [Path("dataset/images"), Path("raw/satellite_images")]:
        if base.exists():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
                candidates.extend(sorted(base.glob(ext)))
        if len(candidates) >= 4:
            break
    return candidates[:4]


def load_or_synthesize_images() -> List[np.ndarray]:
    paths = find_image_paths()
    imgs: List[np.ndarray] = []

    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            # Resize to a common size for visualization
            img = img.resize((128, 128), Image.BILINEAR)
            imgs.append(np.array(img))
        except Exception:
            continue

    # If we still have fewer than 4, synthesize the rest
    while len(imgs) < 4:
        # Synthetic RGB patch with "urban-like" structure: random blocks
        patch = np.zeros((128, 128, 3), dtype=np.uint8)
        # Roads (gray lines)
        patch[60:68, :] = 160
        patch[:, 60:68] = 160
        # Vegetation (green blocks)
        patch[10:50, 10:50, 1] = 180
        # Buildings (light rectangles)
        patch[80:120, 20:60] = 200
        patch[20:60, 80:120] = 200
        imgs.append(patch)

    return imgs[:4]


def main() -> None:
    imgs = load_or_synthesize_images()

    fig, axes = plt.subplots(2, 2, figsize=(4, 4), dpi=300)
    fig.patch.set_facecolor("white")

    labels = ["(a)", "(b)", "(c)", "(d)"]

    for ax, img, label in zip(axes.ravel(), imgs, labels):
        ax.imshow(img)
        ax.set_axis_off()
        # Optional small label in corner
        ax.text(
            3,
            12,
            label,
            color="white",
            fontsize=7,
            fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.3, pad=1, edgecolor="none"),
        )

    plt.tight_layout(pad=0.1)

    out_dir = Path("docs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sample_patches.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sample patches figure to {out_path}")


if __name__ == "__main__":
    main()

