from __future__ import annotations

from typing import Any


def get_target_size_from_config(config: Any) -> tuple[int, int]:
    """Extract target image size from config.

    Supports list or tuple entries and returns a (W, H) tuple.
    """
    target_size = config.data.image_size
    if isinstance(target_size, list | tuple):
        w, h = int(target_size[0]), int(target_size[1])
        return (w, h)
    return target_size  # assumed tuple[int,int]
