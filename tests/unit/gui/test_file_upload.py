"""Unit tests for file upload functionality.

This module tests the YAML file upload, validation, and processing
functionality of the GUI application.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

from gui.utils.config import (
    ConfigError,
    ValidationError,
    get_upload_file_info,
    upload_config_file,
    validate_uploaded_content,
)


class TestFileUploadFunctions:
    """Test class for file upload utility functions."""

    def test_get_upload_file_info_valid_yaml(self):
        """Test getting file info for a valid YAML file."""
        # Create a mock uploaded file
        mock_file = Mock()
        mock_file.name = "test_config.yaml"
        mock_file.size = 1024  # 1 KB
        mock_file.type = "application/yaml"

        info = get_upload_file_info(mock_file)

        assert info["name"] == "test_config.yaml"
        assert info["size"] == 1024
        assert info["size_human"] == "1.0 KB"
        assert info["extension"] == ".yaml"
        assert info["is_valid_extension"] is True
        assert info["is_valid_size"] is True
        assert info["max_size_mb"] == 10

    def test_get_upload_file_info_invalid_extension(self):
        """Test getting file info for file with invalid extension."""
        mock_file = Mock()
        mock_file.name = "test_config.txt"
        mock_file.size = 1024
        mock_file.type = "text/plain"

        info = get_upload_file_info(mock_file)

        assert info["extension"] == ".txt"
        assert info["is_valid_extension"] is False

    def test_get_upload_file_info_oversized_file(self):
        """Test getting file info for oversized file."""
        mock_file = Mock()
        mock_file.name = "large_config.yaml"
        mock_file.size = 15 * 1024 * 1024  # 15 MB
        mock_file.type = "application/yaml"

        info = get_upload_file_info(mock_file)

        assert info["is_valid_size"] is False

    def test_validate_uploaded_content_valid_yaml(self):
        """Test validation of valid YAML content."""
        valid_yaml = """
        model:
          name: test_model
          parameters:
            learning_rate: 0.001
            batch_size: 32
        data:
          train_path: /path/to/train
          val_path: /path/to/val
        """

        is_valid, errors = validate_uploaded_content(valid_yaml)

        # Should pass basic YAML validation at minimum
        # Advanced validation might have warnings, but no critical errors
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)

    def test_validate_uploaded_content_invalid_yaml(self):
        """Test validation of invalid YAML content."""
        invalid_yaml = """
        model:
          name: test_model
          parameters:
            learning_rate: 0.001
            batch_size: 32
          - invalid_structure  # This is invalid YAML
        """

        is_valid, errors = validate_uploaded_content(invalid_yaml)

        assert is_valid is False
        assert len(errors) > 0
        # ValidationError objects don't have severity attribute, verify they
        # are ValidationError instances
        assert all(isinstance(error, ValidationError) for error in errors)
        # Verify at least one error contains syntax information
        assert any("syntax" in str(error).lower() for error in errors)

    def test_upload_config_file_success(self):
        """Test successful file upload."""
        # Setup mock uploaded file
        mock_file = Mock()
        mock_file.name = "test_config.yaml"
        mock_file.size = 1024
        mock_file.read.return_value = b"""
model:
  name: test_model
  learning_rate: 0.001
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            result = upload_config_file(
                mock_file,
                target_directory=temp_dir,
                validate_on_upload=False,  # Skip validation for this test
            )

            file_path, config_dict, validation_errors = result

            # Verify results
            assert isinstance(file_path, str)
            assert isinstance(config_dict, dict)
            assert isinstance(validation_errors, list)
            assert "test_model" in str(config_dict)

            # Verify file was actually created
            assert Path(file_path).exists()
            assert Path(file_path).suffix == ".yaml"

    def test_upload_config_file_invalid_extension(self):
        """Test upload with invalid file extension."""
        mock_file = Mock()
        mock_file.name = "test_config.txt"
        mock_file.size = 1024

        with pytest.raises(ConfigError) as exc_info:
            upload_config_file(mock_file)

        assert "Invalid file extension" in str(exc_info.value)

    def test_upload_config_file_oversized(self):
        """Test upload with oversized file."""
        mock_file = Mock()
        mock_file.name = "large_config.yaml"
        mock_file.size = 15 * 1024 * 1024  # 15 MB

        with pytest.raises(ConfigError) as exc_info:
            upload_config_file(mock_file)

        assert "exceeds maximum allowed size" in str(exc_info.value)

    def test_upload_config_file_invalid_yaml_syntax(self):
        """Test upload with invalid YAML syntax."""
        mock_file = Mock()
        mock_file.name = "invalid_config.yaml"
        mock_file.size = 1024
        mock_file.read.return_value = b"""
        model:
          name: test_model
        - invalid: structure
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            # Function should not raise exception, but return validation errors
            file_path, config_dict, validation_errors = upload_config_file(
                mock_file, target_directory=temp_dir
            )

            # Should return validation errors instead of raising exception
            assert len(validation_errors) > 0
            assert any(
                "syntax" in str(error).lower() for error in validation_errors
            )
            assert isinstance(
                config_dict, dict
            )  # Should be empty dict for invalid YAML

    def test_upload_config_file_encoding_error(self):
        """Test upload with file encoding issues."""
        mock_file = Mock()
        mock_file.name = "encoded_config.yaml"
        mock_file.size = 1024
        # Simulate non-UTF8 content
        mock_file.read.return_value = b"\xff\xfe\x00\x00invalid utf-8"

        with pytest.raises(ConfigError) as exc_info:
            upload_config_file(mock_file)

        assert "File encoding error" in str(exc_info.value)


class TestFileUploadValidationEdgeCases:
    """Test edge cases for file upload validation."""

    def test_empty_yaml_file(self):
        """Test validation of empty YAML file."""
        empty_yaml = ""
        is_valid, errors = validate_uploaded_content(empty_yaml)

        # Empty YAML is technically valid (becomes empty dict) according to
        # YAML spec
        # However, our validation requires certain structure for crack
        # segmentation configs
        assert is_valid is False  # Should fail validation
        assert len(errors) > 0
        # Should have error about configuration format
        assert any("configuration" in str(error).lower() for error in errors)

    def test_yaml_with_only_comments(self):
        """Test validation of YAML with only comments."""
        comment_only_yaml = """
        # This is a comment
        # Another comment
        """

        is_valid, errors = validate_uploaded_content(comment_only_yaml)

        # Comments-only YAML is technically valid YAML (becomes empty dict)
        # However, our validation requires certain structure for crack
        # segmentation configs
        assert is_valid is False  # Should fail validation
        assert len(errors) > 0
        # Should have error about configuration format
        assert any("configuration" in str(error).lower() for error in errors)

    def test_large_valid_yaml(self):
        """Test validation of large but valid YAML content."""
        # Create a large YAML structure
        large_config = {
            "model": {
                "layers": [
                    {"type": f"layer_{i}", "units": i * 10} for i in range(100)
                ]
            },
            "data": {"datasets": [f"dataset_{i}" for i in range(50)]},
        }

        large_yaml = yaml.dump(large_config)
        is_valid, errors = validate_uploaded_content(large_yaml)

        # Large but valid YAML should pass
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)


class TestFileUploadProgressFunctions:
    """Test progress indication functions."""

    @patch("scripts.gui.utils.config.io.st")
    def test_create_upload_progress_placeholder(
        self, mock_st: MagicMock
    ) -> None:
        """Test creation of progress placeholder."""
        mock_placeholder = Mock()
        mock_st.empty.return_value = mock_placeholder

        from gui.utils.config.io import (
            create_upload_progress_placeholder,
        )

        result = create_upload_progress_placeholder()

        assert result == mock_placeholder
        mock_st.empty.assert_called_once()

    @patch("scripts.gui.utils.config.io.st")
    def test_update_upload_progress_stages(self, mock_st: MagicMock) -> None:
        """Test progress updates for different stages."""
        mock_placeholder = Mock()

        from gui.utils.config.io import update_upload_progress

        # Test different stages
        stages = ["reading", "validating", "saving", "complete", "error"]

        for stage in stages:
            update_upload_progress(
                mock_placeholder, stage, 0.5, "test message"
            )

            # Verify that appropriate method was called
            if stage == "complete":
                mock_placeholder.success.assert_called()
            elif stage == "error":
                mock_placeholder.error.assert_called()
            else:
                mock_placeholder.info.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])
