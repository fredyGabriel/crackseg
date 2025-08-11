"""Helper utilities for configuration verification to keep module slim."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config_verifier_utils import (
    calculate_dict_depth,
    check_circular_references,
    check_unused_references,
    count_dict_keys,
    find_environment_variables,
    find_nested_configs,
    parse_basic_yaml,
    verify_path_consistency,
    verify_section_structures,
)


def verify_extension(
    config_path: Path,
    add_metadata: Callable[[str, Any], None],
    add_error: Callable[[str], None],
) -> bool:
    ext = config_path.suffix.lower()
    supported = [".yaml", ".yml", ".json"]
    if ext not in supported:
        add_error(f"Unsupported configuration extension: {ext}")
        return False
    add_metadata("config_format", ext)
    return True


def load_config_content(config_path: Path) -> dict[str, Any]:
    ext = config_path.suffix.lower()
    if ext == ".json":
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    # YAML basic parsing
    with open(config_path, encoding="utf-8") as f:
        content = f.read()
    if not content.strip():
        raise ValueError("Configuration file is empty")
    return parse_basic_yaml(content)


def analyze_yaml_text(content: str) -> dict[str, Any]:
    lines = content.splitlines()
    meta: dict[str, Any] = {
        "config_lines": len(lines),
        "config_size": len(content),
    }
    low = content.lower()
    if "hydra:" in low:
        meta["has_hydra_config"] = True
    if "defaults:" in low:
        meta["has_defaults"] = True
    return meta


def deep_analyze_json(config_data: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "max_depth": calculate_dict_depth(config_data),
        "total_keys": count_dict_keys(config_data),
    }
    nested = find_nested_configs(config_data)
    if nested:
        meta["nested_configs"] = nested
    env_vars = find_environment_variables(config_data)
    if env_vars:
        meta["environment_variables"] = env_vars
    return meta


def cross_check_json(config_data: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    circular = check_circular_references(config_data)
    if circular:
        meta["circular_references"] = circular
    unused = check_unused_references(config_data)
    if unused:
        meta["unused_references"] = unused
    path_issues = verify_path_consistency(config_data)
    if path_issues:
        meta["path_issues"] = path_issues
    return meta


def verify_required_sections(
    config_data: dict[str, Any], required_sections: list[str]
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    existing: list[str] = []
    for section in required_sections:
        if section in config_data:
            existing.append(section)
        else:
            missing.append(section)
    return existing, missing


def verify_structures(
    config_data: dict[str, Any],
    add_warning: Callable[[str], None],
    add_metadata: Callable[[str, Any], None],
) -> None:
    verify_section_structures(config_data, add_warning, add_metadata)
