"""Test module for PageRouter.

Tests page routing functionality including navigation,
validation, and metadata management.
"""

from unittest.mock import Mock, patch

from tests.unit.gui.components.test_component_base import (
    ComponentTestBase,
    MockSessionState,
)


class TestPageRouter(ComponentTestBase):
    """Test suite for PageRouter functionality."""

    def test_page_router_import(self) -> None:
        """Test that PageRouter can be imported successfully."""
        from gui.components.page_router import PageRouter

        assert PageRouter is not None

    @patch("scripts.gui.components.page_router.PAGE_CONFIG")
    def test_get_available_pages(self, mock_page_config) -> None:
        """Test get_available pages with session state."""
        from gui.components.page_router import PageRouter

        # Mock page configuration
        page_config_dict: dict[str, dict[str, list[str]]] = {
            "Home": {"requires": []},
            "Config": {"requires": ["config_loaded"]},
            "Train": {"requires": ["config_loaded", "run_directory"]},
        }  # type: ignore[var-annotated]

        # Test with no requirements met
        state = MockSessionState()
        state.config_loaded = False
        state.run_directory = None

        with patch(
            "scripts.gui.components.page_router.PAGE_CONFIG",
            page_config_dict,  # type: ignore[arg-type]
        ):
            available = PageRouter.get_available_pages(state)  # type: ignore[arg-type]
            assert "Home" in available

    def test_validate_page_transition_success(self) -> None:
        """Test successful page transition validation."""
        from gui.components.page_router import PageRouter

        state = MockSessionState()
        state.config_loaded = True
        state.run_directory = "/test/run"

        with patch(
            "scripts.gui.components.page_router.PAGE_CONFIG",
            {
                "Home": {"requires": []},
                "Config": {"requires": ["config_loaded"]},
            },
        ):
            with patch.object(
                PageRouter,
                "get_available_pages",
                return_value=["Home", "Config"],
            ):
                is_valid, error_msg = PageRouter.validate_page_transition(
                    "Home",
                    "Config",
                    state,  # type: ignore[arg-type]
                )
                assert is_valid is True
                assert error_msg == ""

    def test_validate_page_transition_failure(self) -> None:
        """Test failed page transition validation."""
        from gui.components.page_router import PageRouter

        state = MockSessionState()
        state.config_loaded = False
        state.run_directory = None

        with patch(
            "scripts.gui.components.page_router.PAGE_CONFIG",
            {"Train": {"requires": ["config_loaded", "run_directory"]}},
        ):
            with patch.object(
                PageRouter, "get_available_pages", return_value=["Home"]
            ):
                is_valid, error_msg = PageRouter.validate_page_transition(
                    "Home",
                    "Train",
                    state,  # type: ignore[arg-type]
                )
                assert is_valid is False
                assert "Cannot navigate to Train" in error_msg

    @patch("scripts.gui.components.page_router.st")
    @patch("scripts.gui.components.page_router.SessionStateManager")
    def test_route_to_page_success(
        self, mock_session_manager, mock_st
    ) -> None:
        """Test successful page routing."""
        from gui.components.page_router import PageRouter

        state = MockSessionState()
        state.current_page = "Home"

        page_functions = {"page_config": Mock()}  # type: ignore[var-annotated]

        with patch.object(
            PageRouter, "validate_page_transition", return_value=(True, "")
        ):
            with patch.object(
                PageRouter,
                "_page_metadata",
                {"Config": {"title": "Test Config"}},
            ):
                PageRouter.route_to_page("Config", state, page_functions)  # type: ignore[arg-type]

                # Verify session state update was called
                mock_session_manager.update.assert_called_with(
                    {"current_page": "Config"}
                )

    @patch("scripts.gui.components.page_router.st")
    def test_route_to_page_validation_failure(self, mock_st) -> None:
        """Test page routing with validation failure."""
        from gui.components.page_router import PageRouter

        state = MockSessionState()
        state.current_page = "Home"

        page_functions = {}  # type: ignore[var-annotated]

        with patch.object(
            PageRouter,
            "validate_page_transition",
            return_value=(False, "Access denied"),
        ):
            PageRouter.route_to_page("Config", state, page_functions)  # type: ignore[arg-type]

            # Verify error was shown
            mock_st.error.assert_called_with("Access denied")

    @patch("scripts.gui.components.page_router.SessionStateManager")
    def test_handle_navigation_change_success(
        self, mock_session_manager
    ) -> None:
        """Test successful navigation change handling."""
        from gui.components.page_router import PageRouter

        state = MockSessionState()
        state.current_page = "Home"

        with patch.object(
            PageRouter, "validate_page_transition", return_value=(True, "")
        ):
            result = PageRouter.handle_navigation_change("Config", state)  # type: ignore[arg-type]

            assert result is True
            mock_session_manager.update.assert_called_with(
                {"current_page": "Config"}
            )

    def test_handle_navigation_change_same_page(self) -> None:
        """Test navigation change to the same page."""
        from gui.components.page_router import PageRouter

        state = MockSessionState()
        state.current_page = "Home"

        result = PageRouter.handle_navigation_change("Home", state)  # type: ignore[arg-type]

        assert result is True

    def test_handle_navigation_change_validation_failure(self) -> None:
        """Test navigation change with validation failure."""
        from gui.components.page_router import PageRouter

        state = MockSessionState()
        state.current_page = "Home"

        with patch.object(
            PageRouter,
            "validate_page_transition",
            return_value=(False, "Invalid transition"),
        ):
            result = PageRouter.handle_navigation_change("Config", state)  # type: ignore[arg-type]

            assert result is False

    def test_get_page_breadcrumbs(self) -> None:
        """Test page breadcrumbs generation."""
        from gui.components.page_router import PageRouter

        breadcrumbs = PageRouter.get_page_breadcrumbs("Config")

        assert isinstance(breadcrumbs, str)

    def test_page_components_mapping(self) -> None:
        """Test page components mapping exists."""
        from gui.components.page_router import PageRouter

        assert hasattr(PageRouter, "_page_components")
        assert isinstance(PageRouter._page_components, dict)

    def test_page_metadata_mapping(self) -> None:
        """Test page metadata mapping exists."""
        from gui.components.page_router import PageRouter

        assert hasattr(PageRouter, "_page_metadata")
        assert isinstance(PageRouter._page_metadata, dict)
