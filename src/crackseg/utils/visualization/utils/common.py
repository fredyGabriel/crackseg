"""Common visualization utilities for plots.

Shared helpers extracted from large plotting module to reduce size
and duplication while keeping behavior unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def denormalize_image_rgb(img: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Denormalize image from [-1, 1] to [0, 1] if needed and clip."""
    if img.min() < 0:
        img = (img + 1.0) / 2.0
    return np.clip(img, 0.0, 1.0)


def save_current_figure(path: str | Path, dpi: int = 200) -> None:
    """Save current matplotlib figure to path with tight layout and dpi."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
