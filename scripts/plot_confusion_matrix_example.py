#!/usr/bin/env python
"""
Generate an example confusion matrix heatmap for a 3-class
heat-risk classification problem.

Classes:
- Low Heat Risk
- Medium Heat Risk
- High Heat Risk

The script uses realistic example values and produces a
clean, academic-style matplotlib figure suitable for papers.

Output:
  docs/confusion_matrix_example.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    # Example confusion matrix counts (rows = true, cols = predicted)
    cm = np.array(
        [
            [45, 5, 0],   # True Low
            [6, 38, 6],   # True Medium
            [0, 7, 30],   # True High
        ],
        dtype=np.int32,
    )

    class_names = ["Low Heat Risk", "Medium Heat Risk", "High Heat Risk"]

    # Slightly wider than tall so labels have space
    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    fig.patch.set_facecolor("white")

    # Use origin="upper" so the first row is at the top,
    # which matches the usual table-style layout.
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, origin="upper")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Sample count", rotation=90, va="center")

    # Axis ticks and labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    # Keep labels horizontal for a cleaner "table" look
    ax.set_xticklabels(class_names, rotation=0, ha="center")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    # Add grid-like separation
    ax.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate each cell with its value
    max_val = cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            # Choose text color based on background intensity
            color = "white" if value > max_val * 0.5 else "black"
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color=color,
                fontsize=7,
            )

    # Make cells roughly square and layout tight
    ax.set_aspect("equal")
    plt.tight_layout(pad=0.4)

    out_dir = Path("docs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "confusion_matrix_example.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix example to {out_path}")


if __name__ == "__main__":
    main()
