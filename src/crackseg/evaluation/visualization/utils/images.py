"""Utility functions for visualization I/O operations.

Lightweight helpers extracted to reduce module size and improve reuse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
from matplotlib.figure import Figure


def load_image_rgb(image_path: str | Path) -> Any:
    """Load an image from disk and convert to RGB array."""
    image_path = str(image_path)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_figure(fig: Any, save_path: str | Path) -> None:
    """Save a Matplotlib or Plotly figure to disk.

    Falls back to calling ``write_image`` if the object is not a Matplotlib
    Figure but exposes the Plotly API.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(fig, Figure):
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return

    # Plotly fallback without importing plotly
    if hasattr(fig, "write_image"):
        fig.write_image(str(save_path))
        return

    raise TypeError(
        "Unsupported figure type. Expected Matplotlib Figure or Plotly figure."
    )
