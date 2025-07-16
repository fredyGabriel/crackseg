"""Unit tests for configuration template creation."""

from pathlib import Path

import pytest
import yaml

from gui.utils.config.exceptions import ConfigError
from gui.utils.config.templates import create_config_from_template


@pytest.fixture
def template_file(tmp_path: Path) -> Path:
    """Fixture to create a sample template file."""
    template_path = tmp_path / "template.yaml"
    template_data: dict[str, object] = {
        "model": {"architecture": "unet", "encoder": {"type": "resnet50"}},
        "training": {"epochs": 50, "batch_size": 16},
    }
    template_path.write_text(yaml.dump(template_data))
    return template_path


class TestConfigTemplates:
    """Test suite for configuration template functions."""

    def test_create_from_template_no_overrides(
        self, template_file: Path, tmp_path: Path
    ):
        """Test creating a config from a template without any overrides."""
        output_path = tmp_path / "output.yaml"
        create_config_from_template(str(template_file), str(output_path))

        assert output_path.exists()
        original_data = yaml.safe_load(template_file.read_text())
        new_data = yaml.safe_load(output_path.read_text())
        assert original_data == new_data

    def test_create_from_template_with_simple_overrides(
        self, template_file: Path, tmp_path: Path
    ):
        """Test creating a config with simple, top-level overrides."""
        output_path = tmp_path / "output_simple.yaml"
        overrides: dict[str, object] = {"training": {"epochs": 100}}
        create_config_from_template(
            str(template_file), str(output_path), overrides
        )

        new_data = yaml.safe_load(output_path.read_text())
        assert new_data["training"]["epochs"] == 100
        assert new_data["training"]["batch_size"] == 16  # Unchanged

    def test_create_from_template_with_nested_overrides(
        self, template_file: Path, tmp_path: Path
    ):
        """Test creating a config with nested key overrides."""
        output_path = tmp_path / "output_nested.yaml"
        overrides: dict[str, object] = {
            "model.encoder.type": "efficientnet",
            "training.batch_size": 32,
        }
        create_config_from_template(
            str(template_file), str(output_path), overrides
        )

        new_data = yaml.safe_load(output_path.read_text())
        assert new_data["model"]["encoder"]["type"] == "efficientnet"
        assert new_data["training"]["batch_size"] == 32

    def test_create_from_template_creates_new_nested_keys(
        self, template_file: Path, tmp_path: Path
    ):
        """Test that overrides can create new nested dictionary structures."""
        output_path = tmp_path / "output_new_keys.yaml"
        overrides: dict[str, object] = {
            "data.augmentation.type": "random_flip"
        }
        create_config_from_template(
            str(template_file), str(output_path), overrides
        )

        new_data = yaml.safe_load(output_path.read_text())
        assert new_data["data"]["augmentation"]["type"] == "random_flip"

    def test_create_from_template_raises_error_on_missing_template(
        self, tmp_path: Path
    ):
        """
        Test that a ConfigError is raised if the template file is not found.
        """
        output_path = tmp_path / "output.yaml"
        with pytest.raises(ConfigError):
            create_config_from_template(
                "/non/existent/template.yaml", str(output_path)
            )
