"""Storage helpers for traceability JSON persistence.

Extracted from storage.py to reduce size and improve modularity.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def convert_datetime_fields(data: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, datetime):
            converted[key] = value.isoformat()
        elif isinstance(value, Path):
            converted[key] = str(value)
        else:
            converted[key] = value
    return converted


def load_json_list(file_path: Path) -> list[dict[str, Any]]:
    try:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def save_json_list(file_path: Path, data: list[dict[str, Any]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
