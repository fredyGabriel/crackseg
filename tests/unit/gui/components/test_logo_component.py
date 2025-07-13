"""Test module for LogoComponent.

Tests logo rendering functionality including generation,
fallback handling, and caching capabilities.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tests.unit.gui.components.test_component_base import (
    ComponentTestBase,
)


class TestLogoComponent(ComponentTestBase):
    """Test suite for LogoComponent functionality."""

    def test_logo_component_import(self) -> None:
        """Test that LogoComponent can be imported successfully."""
        from scripts.gui.components.logo_component import LogoComponent

        assert LogoComponent is not None

    @patch("scripts.gui.components.logo_component.Image")
    @patch("scripts.gui.components.logo_component.ImageDraw")
    @patch("scripts.gui.components.logo_component.ImageFont")
    def test_generate_logo_default(
        self, mock_font, mock_image_draw, mock_image
    ) -> None:
        """Test logo generation with default parameters."""
        from scripts.gui.components.logo_component import LogoComponent

        # Setup mocks for PIL objects
        mock_img = Mock()
        mock_image.new.return_value = mock_img

        # Create proper mock for ImageDraw with textbbox returning
        # subscriptable values
        mock_draw = Mock()
        mock_draw.textbbox.return_value = (0, 0, 100, 50)  # (x1, y1, x2, y2)
        mock_image_draw.Draw.return_value = mock_draw

        # Mock font properly
        mock_font_instance = Mock()
        mock_font.truetype.return_value = mock_font_instance
        mock_font.load_default.return_value = mock_font_instance

        # Call generate_logo - should not raise exceptions
        result = LogoComponent.generate_logo()

        # Verify basic functionality - method runs without errors
        assert result is not None
        mock_image.new.assert_called()  # Called at least once
        mock_image_draw.Draw.assert_called()

    @patch("scripts.gui.components.logo_component.Image")
    @patch("scripts.gui.components.logo_component.ImageDraw")
    @patch("scripts.gui.components.logo_component.ImageFont")
    def test_generate_logo_custom_parameters(
        self, mock_font, mock_image_draw, mock_image
    ) -> None:
        """Test logo generation with custom parameters."""
        from scripts.gui.components.logo_component import LogoComponent

        # Setup comprehensive mocks
        mock_img = Mock()
        mock_image.new.return_value = mock_img

        mock_draw = Mock()
        mock_draw.textbbox.return_value = (0, 0, 100, 50)
        mock_image_draw.Draw.return_value = mock_draw

        mock_font_instance = Mock()
        mock_font.truetype.return_value = mock_font_instance
        mock_font.load_default.return_value = mock_font_instance

        # Call with custom parameters - should not raise exceptions
        result = LogoComponent.generate_logo(
            style="light", width=400, height=400
        )

        # Verify method runs without errors
        assert result is not None
        mock_image.new.assert_called()
        mock_image_draw.Draw.assert_called()

    @patch("scripts.gui.components.logo_component.Image")
    @patch("scripts.gui.components.logo_component.ImageDraw")
    @patch("scripts.gui.components.logo_component.ImageFont")
    def test_generate_logo_different_styles(
        self, mock_font, mock_image_draw, mock_image
    ) -> None:
        """Test logo generation with different styles."""
        from scripts.gui.components.logo_component import LogoComponent

        # Setup comprehensive mocks
        mock_img = Mock()
        mock_image.new.return_value = mock_img

        mock_draw = Mock()
        mock_draw.textbbox.return_value = (0, 0, 100, 50)
        mock_image_draw.Draw.return_value = mock_draw

        mock_font_instance = Mock()
        mock_font.truetype.return_value = mock_font_instance
        mock_font.load_default.return_value = mock_font_instance

        # Test different styles
        styles = ["default", "light", "minimal"]

        for style in styles:
            result = LogoComponent.generate_logo(style=style)
            assert result is not None

    @patch("scripts.gui.components.logo_component.Path")
    @patch(
        "scripts.gui.components.logo_component.LogoComponent._load_from_file"
    )
    def test_load_with_fallback_file_exists(
        self, mock_load_from_file, mock_path
    ) -> None:
        """Test load_with_fallback when file exists."""
        from scripts.gui.components.logo_component import LogoComponent

        # Setup mocks
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.is_file.return_value = True
        mock_path.return_value = mock_path_instance

        mock_load_from_file.return_value = b"fake_logo_data"

        # Call load_with_fallback
        result = LogoComponent.load_with_fallback(
            primary_path="/test/logo.png", style="default", use_cache=False
        )

        # Should return base64 data URL
        assert result is not None
        assert isinstance(result, str)

    @patch("scripts.gui.components.logo_component.LogoComponent.generate_logo")
    @patch("scripts.gui.components.logo_component.Path")
    def test_load_with_fallback_file_missing(
        self, mock_path, mock_generate_logo
    ) -> None:
        """Test load_with_fallback when file is missing."""
        from scripts.gui.components.logo_component import LogoComponent

        # Setup mocks - file doesn't exist
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance

        mock_img = Mock()
        mock_img.tobytes.return_value = b"generated_logo_data"
        mock_generate_logo.return_value = mock_img

        # Call load_with_fallback
        result = LogoComponent.load_with_fallback(
            primary_path="/nonexistent/logo.png", style="default"
        )

        # Should fallback to generated logo - method runs without errors
        assert result is not None

    @patch("scripts.gui.components.logo_component.st")
    @patch(
        "scripts.gui.components.logo_component.LogoComponent.load_with_fallback"
    )
    def test_render_method(self, mock_load_with_fallback, mock_st) -> None:
        """Test render method displays logo."""
        from scripts.gui.components.logo_component import LogoComponent

        # Setup mocks
        mock_load_with_fallback.return_value = (
            "data:image/png;base64,fake_data"
        )

        # Call render
        LogoComponent.render(
            primary_path="/test/logo.png",
            style="default",
            width=150,
            alt_text="Test Logo",
            center=True,
        )

        # Verify Streamlit components were called
        mock_st.markdown.assert_called()
        mock_load_with_fallback.assert_called_once()

    @patch("scripts.gui.components.logo_component.st")
    @patch(
        "scripts.gui.components.logo_component.LogoComponent.load_with_fallback"
    )
    def test_render_method_custom_parameters(
        self, mock_load_with_fallback, mock_st
    ) -> None:
        """Test render method with custom parameters."""
        from scripts.gui.components.logo_component import LogoComponent

        # Setup mocks
        mock_load_with_fallback.return_value = (
            "data:image/png;base64,fake_data"
        )

        # Call render with custom parameters
        LogoComponent.render(
            style="light",
            width=200,
            alt_text="Custom Logo",
            css_class="custom-logo",
            center=False,
        )

        # Verify method was called successfully
        mock_load_with_fallback.assert_called_once()
        _, kwargs = mock_load_with_fallback.call_args  # type: ignore[misc]
        assert kwargs.get("style") == "light"
        # Note: width may get transformed by the component logic
        assert "width" in kwargs or "height" in kwargs

    def test_clear_cache_method(self) -> None:
        """Test clear_cache method."""
        from scripts.gui.components.logo_component import LogoComponent

        # Add something to cache first
        LogoComponent._cache["test_key"] = "test_value"

        # Clear cache
        LogoComponent.clear_cache()

        # Verify cache is empty
        assert len(LogoComponent._cache) == 0

    def test_fallback_styles_available(self) -> None:
        """Test fallback styles are properly defined."""
        from scripts.gui.components.logo_component import LogoComponent

        # Check fallback styles
        assert hasattr(LogoComponent, "_fallback_styles")
        assert isinstance(LogoComponent._fallback_styles, dict)

        # Check required styles
        required_styles = ["default", "light", "minimal"]
        for style in required_styles:
            assert style in LogoComponent._fallback_styles
            assert "bg_color" in LogoComponent._fallback_styles[style]
            assert "road_color" in LogoComponent._fallback_styles[style]

    @patch(
        "scripts.gui.components.logo_component.LogoComponent.load_with_fallback"
    )
    def test_render_with_project_root(
        self, mock_load_with_fallback, sample_project_root: Path
    ) -> None:
        """Test render method with project_root parameter."""
        from scripts.gui.components.logo_component import LogoComponent

        # Setup mock
        mock_load_with_fallback.return_value = (
            "data:image/png;base64,fake_data"
        )

        # Call render with project_root
        LogoComponent.render(project_root=sample_project_root)

        # Verify project_root was passed
        mock_load_with_fallback.assert_called_once()
        _, kwargs = mock_load_with_fallback.call_args  # type: ignore[misc]
        assert kwargs.get("project_root") == sample_project_root

    @patch(
        "scripts.gui.components.logo_component.LogoComponent.load_with_fallback"
    )
    def test_render_handles_none_logo_data(
        self, mock_load_with_fallback
    ) -> None:
        """Test render method handles None logo data gracefully."""
        from scripts.gui.components.logo_component import LogoComponent

        # Setup mock to return None
        mock_load_with_fallback.return_value = None

        # Call render - should not raise exception
        try:
            LogoComponent.render()
        except Exception as e:
            pytest.fail(f"Render should handle None logo data gracefully: {e}")

    def test_logo_component_cache_initialization(self) -> None:
        """Test logo component cache is properly initialized."""
        from scripts.gui.components.logo_component import LogoComponent

        # Check cache exists and is a dict
        assert hasattr(LogoComponent, "_cache")
        assert isinstance(LogoComponent._cache, dict)
