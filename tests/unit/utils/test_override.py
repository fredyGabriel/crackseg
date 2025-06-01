# ruff: noqa: PLR2004
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from src.utils.config.override import (
    apply_overrides,
    override_config,
    save_config,
)
from src.utils.exceptions import ConfigError


def test_override_config_simple():
    """Test simple override of a top-level config value."""
    cfg = OmegaConf.create({"a": 1, "b": 2})
    overrides: dict[str, object] = {"a": 10}
    result = override_config(cfg, overrides)
    assert result["a"] == 10  # noqa: PLR2004
    assert result["b"] == 2  # noqa: PLR2004


def test_override_config_nested():
    """Test override of a nested config value."""
    cfg = OmegaConf.create({"a": {"b": {"c": 1}}})
    overrides: dict[str, object] = {"a.b.c": 42}
    result = override_config(cfg, overrides)
    assert result["a"]["b"]["c"] == 42


def test_override_config_nonexistent_strict():
    """Test override with non-existent path and strict=True raises error."""
    cfg = OmegaConf.create({"a": 1})
    overrides: dict[str, object] = {"x.y": 5}
    with pytest.raises(ConfigError):
        override_config(cfg, overrides, strict=True)


def test_override_config_nonexistent_nonstrict():
    """Test override with non-existent path and strict=False skips override."""
    cfg = OmegaConf.create({"a": 1})
    overrides: dict[str, object] = {"x.y": 5}
    result = override_config(cfg, overrides, strict=False)
    assert "x" not in result
    assert result["a"] == 1


def test_apply_overrides_hydra_dotlist():
    """Test apply_overrides applies Hydra-style dotlist overrides."""
    cfg = OmegaConf.create({"a": 1, "b": {"c": 2}})
    overrides = ["a=10", "b.c=20"]
    result = apply_overrides(cfg, overrides)
    assert result["a"] == 10
    assert result["b"]["c"] == 20


def test_save_config_creates_yaml(tmp_path: Path):
    """Test save_config writes a valid YAML file."""
    cfg = OmegaConf.create({"foo": 123, "bar": {"baz": 456}})
    file_path = tmp_path / "test_config.yaml"
    save_config(cfg, str(file_path))
    assert file_path.exists()
    with open(file_path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert loaded["foo"] == 123
    assert loaded["bar"]["baz"] == 456
