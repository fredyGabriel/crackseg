"""Helper functions for applying path mappings to content."""

from __future__ import annotations


def apply_import_mapping(content: str, old_path: str, new_path: str) -> str:
    patterns = [
        f"from {old_path}",
        f"import {old_path}",
        f"from {old_path}.",
        f"import {old_path}.",
    ]
    result = content
    for pattern in patterns:
        result = result.replace(pattern, pattern.replace(old_path, new_path))
    return result


def apply_config_mapping(content: str, old_path: str, new_path: str) -> str:
    patterns = [
        f"{old_path}:",
        f"  {old_path}:",
        f"    {old_path}:",
        f"defaults:\n  - {old_path}",
        f"defaults:\n    - {old_path}",
    ]
    result = content
    for pattern in patterns:
        result = result.replace(pattern, pattern.replace(old_path, new_path))
    return result


def apply_docs_mapping(content: str, old_path: str, new_path: str) -> str:
    patterns = [
        f"({old_path})",
        f"[{old_path}]",
        f"`{old_path}`",
        f"```{old_path}```",
    ]
    result = content
    for pattern in patterns:
        result = result.replace(pattern, pattern.replace(old_path, new_path))
    return result


def apply_artifact_mapping(content: str, old_path: str, new_path: str) -> str:
    patterns = [
        f'"{old_path}"',
        f"'{old_path}'",
        f"Path('{old_path}')",
        f"pathlib.Path('{old_path}')",
    ]
    result = content
    for pattern in patterns:
        result = result.replace(pattern, pattern.replace(old_path, new_path))
    return result
