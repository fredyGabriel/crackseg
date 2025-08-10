from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _to_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def read_text(path: str | Path, encoding: str = "utf-8") -> str:
    file_path = _to_path(path)
    with open(file_path, encoding=encoding) as f:
        return f.read()


def write_text(
    path: str | Path, content: str, encoding: str = "utf-8"
) -> None:
    file_path = _to_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding=encoding) as f:
        f.write(content)


def read_json(path: str | Path, encoding: str = "utf-8") -> Any:
    file_path = _to_path(path)
    with open(file_path, encoding=encoding) as f:
        return json.load(f)


def write_json(
    path: str | Path,
    data: Any,
    *,
    encoding: str = "utf-8",
    indent: int = 2,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
) -> None:
    file_path = _to_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding=encoding) as f:
        json.dump(
            data,
            f,
            indent=indent,
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
        )
