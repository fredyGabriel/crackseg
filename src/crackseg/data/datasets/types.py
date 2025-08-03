"""Type definitions for crack segmentation datasets.

This module provides type definitions and utilities for the
crack segmentation dataset system.
"""

from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image

# Define SourceType at module level for type hinting
SourceType = str | Path | PIL.Image.Image | np.ndarray[Any, Any]

# Define cache_item_type at module level
CacheItemType = tuple[PIL.Image.Image | None, PIL.Image.Image | None]
