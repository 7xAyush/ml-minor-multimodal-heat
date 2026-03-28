#!/usr/bin/env python
"""
Generate a deep learning architecture diagram for the multimodal model:

- Left branch: Image -> ResNet18 -> image feature vector
- Right branch: Tabular input -> MLP -> tabular feature vector
- Concatenation of both feature vectors
- Fully connected layers (fusion head)
- Output: 3-class classification

The figure is saved as:
  docs/multimodal_architecture.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch


def add_block(ax, xy, width, height, label, facecolor="#E8F0FF"):
    x, y = xy
    rect = Rectangle(
        (x, y),
        width,
        height,
        linewidth=1.5,
        edgecolor="#444444",
        facecolor=facecolor,
    )
    ax.add_patch(rect)
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=9,
        color="#222222",
    )
    return rect


def add_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="->",
        mutation_scale=10,
        linewidth=1.0,
        color="#444444",
    )
    ax.add_patch(arrow)


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    w = 1.8
    h = 0.8

    # Coordinates for image branch (upper)
    img_in_x, img_in_y = 0.5, 2.7
    resnet_x, resnet_y = 2.5, 2.7
    img_feat_x, img_feat_y = 4.5, 2.7

    # Coordinates for tabular branch (lower)
    tab_in_x, tab_in_y = 0.5, 0.9
    mlp_x, mlp_y = 2.5, 0.9
    tab_feat_x, tab_feat_y = 4.5, 0.9

    # Fusion + head
    concat_x, concat_y = 6.3, 1.8
    fc_x, fc_y = 8.1, 1.8
    out_x, out_y = 9.9 - w, 1.8

    # Blocks: image branch
    img_in_label = "Image Input\n(3×H×W)"
    resnet_label = "ResNet18\nImage Encoder"
    img_feat_label = "Image Feature\nVector"

    add_block(ax, (img_in_x, img_in_y), w, h, img_in_label)
    add_block(ax, (resnet_x, resnet_y), w, h, resnet_label)
    add_block(ax, (img_feat_x, img_feat_y), w, h, img_feat_label)

    # Blocks: tabular branch
    tab_in_label = "Tabular Input\n(weather + urban)"
    mlp_label = "MLP\nTabular Encoder"
    tab_feat_label = "Tabular Feature\nVector"

    add_block(ax, (tab_in_x, tab_in_y), w, h, tab_in_label, facecolor="#F0F3F7")
    add_block(ax, (mlp_x, mlp_y), w, h, mlp_label, facecolor="#E8F0FF")
    add_block(ax, (tab_feat_x, tab_feat_y), w, h, tab_feat_label, facecolor="#F0F3F7")

    # Fusion + head blocks
    concat_label = "Concatenation\n[img_feat; tab_feat]"
    fc_label = "Fusion Head\nFully Connected Layers"
    out_label = "3-Class Output\nHeat-Risk\nLow / Medium / High"

    add_block(ax, (concat_x, concat_y), w, h, concat_label)
    add_block(ax, (fc_x, fc_y), w, h, fc_label)
    add_block(ax, (out_x, out_y), w, h, out_label, facecolor="#F0F3F7")

    # Arrows: image branch
    add_arrow(
        ax,
        (img_in_x + w, img_in_y + h / 2),
        (resnet_x, resnet_y + h / 2),
    )
    add_arrow(
        ax,
        (resnet_x + w, resnet_y + h / 2),
        (img_feat_x, img_feat_y + h / 2),
    )

    # Arrows: tabular branch
    add_arrow(
        ax,
        (tab_in_x + w, tab_in_y + h / 2),
        (mlp_x, mlp_y + h / 2),
    )
    add_arrow(
        ax,
        (mlp_x + w, mlp_y + h / 2),
        (tab_feat_x, tab_feat_y + h / 2),
    )

    # Arrows into concatenation block
    add_arrow(
        ax,
        (img_feat_x + w, img_feat_y + h / 2),
        (concat_x, concat_y + h * 0.75),
    )
    add_arrow(
        ax,
        (tab_feat_x + w, tab_feat_y + h / 2),
        (concat_x, concat_y + h * 0.25),
    )

    # Arrows through fusion head to output
    add_arrow(
        ax,
        (concat_x + w, concat_y + h / 2),
        (fc_x, fc_y + h / 2),
    )
    add_arrow(
        ax,
        (fc_x + w, fc_y + h / 2),
        (out_x, out_y + h / 2),
    )

    out_dir = Path("docs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "multimodal_architecture.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved multimodal architecture diagram to {out_path}")


if __name__ == "__main__":
    main()

