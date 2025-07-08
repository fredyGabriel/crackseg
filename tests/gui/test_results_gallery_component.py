"""Tests for Results Gallery Component.

This module tests the main container component for the results gallery.
It validates the real API and behavior of the production component.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.gui.components.results_gallery_component import (
    ResultsGalleryComponent,
)
from scripts.gui.utils.results import (
    ResultTriplet,
    TripletHealth,
    ValidationLevel,
)


class TestResultsGalleryComponent:
    """Test suite for Results Gallery Component."""

    @pytest.fixture
    def component(self) -> ResultsGalleryComponent:
        """Create a component instance for testing."""
        with patch("streamlit.session_state", {}):
            return ResultsGalleryComponent()

    @pytest.fixture
    def sample_triplets(self) -> list[ResultTriplet]:
        """Create sample triplets for testing."""
        triplets = []
        for i in range(3):
            triplet = ResultTriplet(
                id=f"triplet_{i}",
                image_path=Path(f"/test/image_{i}.png"),
                mask_path=Path(f"/test/mask_{i}.png"),
                prediction_path=Path(f"/test/pred_{i}.png"),
                dataset_name=f"dataset_{i}",
                metadata={"test": f"value_{i}"},
                health_status=TripletHealth.BROKEN,
                missing_files=[
                    Path(f"/test/image_{i}.png"),
                    Path(f"/test/mask_{i}.png"),
                    Path(f"/test/pred_{i}.png"),
                ],
            )
            triplets.append(triplet)
        return triplets

    def test_component_initialization(
        self, component: ResultsGalleryComponent
    ) -> None:
        """Test component initializes correctly."""
        assert component is not None
        assert hasattr(component, "state")
        assert hasattr(component, "scanner_service")
        assert hasattr(component, "exporter_service")
        assert hasattr(component, "actions")
        assert hasattr(component, "event_handlers")
        assert hasattr(component, "renderer")

    def test_session_state_initialization(
        self, component: ResultsGalleryComponent
    ) -> None:
        """Test session state is properly initialized."""
        # The component should have its state manager initialized
        assert component.state is not None

        # Test initial state values
        scan_results = component.state.get("scan_results", [])
        assert isinstance(scan_results, list)
        assert len(scan_results) == 0

    def test_event_handler_setup(
        self, component: ResultsGalleryComponent
    ) -> None:
        """Test event handlers are properly set up."""
        assert component.event_handlers is not None
        # The setup_event_handlers should have been called during init
        assert hasattr(component.event_handlers, "setup_event_handlers")

    def test_config_update(self, component: ResultsGalleryComponent) -> None:
        """Test configuration updates work correctly."""
        validation_level = ValidationLevel.THOROUGH
        max_triplets = 100

        # Test the render method with configuration parameters
        component.render(
            validation_level=validation_level,
            max_triplets=max_triplets,
            grid_columns=4,
        )

        # Check if config was stored in state
        config = component.state.get("config", {})
        assert config.get("validation_level") == validation_level
        assert config.get("max_triplets") == max_triplets
        assert config.get("grid_columns") == 4

    def test_render_with_no_directory(
        self, component: ResultsGalleryComponent
    ) -> None:
        """Test rendering without a scan directory."""
        # This should not raise an exception
        try:
            component.render(scan_directory=None)
        except Exception as e:
            pytest.fail(f"Render should handle None directory gracefully: {e}")

    def test_render_with_directory(
        self, component: ResultsGalleryComponent
    ) -> None:
        """Test rendering with a scan directory."""
        test_dir = Path("/test/directory")

        try:
            component.render(scan_directory=test_dir)
        except Exception as e:
            pytest.fail(f"Render should handle valid directory: {e}")

    def test_ui_state_property(
        self,
        component: ResultsGalleryComponent,
        sample_triplets: list[ResultTriplet],
    ) -> None:
        """Test UI state property returns correct information."""
        # Set some test data in state
        component.state.set("scan_results", sample_triplets)
        component.state.set("selected_triplet_ids", {sample_triplets[0].id})
        component.state.set(
            "validation_stats", {"total_triplets": 3, "valid_triplets": 1}
        )

        ui_state = component.ui_state

        assert "total_triplets" in ui_state
        assert "valid_triplets" in ui_state
        assert "selected_triplets" in ui_state
        assert "cache_stats" in ui_state

        assert ui_state["total_triplets"] == 3
        assert ui_state["valid_triplets"] == 1
        assert len(ui_state["selected_triplets"]) == 1

    def test_scanner_service_integration(
        self, component: ResultsGalleryComponent
    ) -> None:
        """Test scanner service is properly integrated."""
        assert component.scanner_service is not None
        assert hasattr(component.scanner_service, "cache")

    def test_exporter_service_integration(
        self, component: ResultsGalleryComponent
    ) -> None:
        """Test exporter service is properly integrated."""
        assert component.exporter_service is not None
        assert component.exporter_service.state == component.state

    def test_actions_integration(
        self, component: ResultsGalleryComponent
    ) -> None:
        """Test actions are properly integrated."""
        assert component.actions is not None
        assert component.actions.state == component.state
        assert component.actions.scanner == component.scanner_service
        assert component.actions.exporter == component.exporter_service

    def test_validation_level_integration(
        self, component: ResultsGalleryComponent
    ) -> None:
        """Test validation level is properly handled."""
        for level in ValidationLevel:
            try:
                component.render(validation_level=level)
                config = component.state.get("config", {})
                assert config.get("validation_level") == level
            except Exception as e:
                pytest.fail(f"Should handle validation level {level}: {e}")

    def test_render_parameters(
        self, component: ResultsGalleryComponent
    ) -> None:
        """Test all render parameters are handled correctly."""
        scan_directory = Path("/test")
        validation_level = ValidationLevel.BASIC
        max_triplets = 25
        grid_columns = 2
        show_validation_panel = False
        show_export_panel = False
        enable_real_time_scanning = False

        try:
            component.render(
                scan_directory=scan_directory,
                validation_level=validation_level,
                max_triplets=max_triplets,
                grid_columns=grid_columns,
                show_validation_panel=show_validation_panel,
                show_export_panel=show_export_panel,
                enable_real_time_scanning=enable_real_time_scanning,
            )
            config = component.state.get("config", {})

            # Verify each parameter is stored correctly
            assert config.get("validation_level") == validation_level
            assert config.get("max_triplets") == max_triplets
            assert config.get("grid_columns") == grid_columns
            assert config.get("show_validation_panel") == show_validation_panel
            assert config.get("show_export_panel") == show_export_panel
            assert config.get("enable_real_time") == enable_real_time_scanning

        except Exception as e:
            pytest.fail(f"Render should handle all parameters: {e}")

    def test_state_manager_functionality(
        self, component: ResultsGalleryComponent
    ) -> None:
        """Test state manager works correctly."""
        # Use a valid state key that the state manager recognizes
        test_key = "config"
        test_value = {"test": "data"}

        component.state.set(test_key, test_value)
        retrieved_value = component.state.get(test_key)

        assert retrieved_value == test_value

    def test_event_progress_handler(
        self, component: ResultsGalleryComponent
    ) -> None:
        """Test progress event handling."""
        # Verify event handlers are set up
        assert component.event_handlers is not None
        # The actual event handling is tested in the event_handlers module
        # Here we just verify the component has the handler setup

    def test_event_triplet_found_handler(
        self, component: ResultsGalleryComponent
    ) -> None:
        """Test triplet found event handling."""
        # Verify event handlers are set up
        assert component.event_handlers is not None
        # The actual event handling is tested in the event_handlers module

    def test_backward_compatibility_attributes(
        self, component: ResultsGalleryComponent
    ) -> None:
        """Test backward compatibility attributes exist."""
        # These attributes are provided for test compatibility
        assert hasattr(component, "event_manager")
        assert hasattr(component, "cache")
        assert hasattr(component, "_state_keys")

        # Test state keys mapping
        assert isinstance(component._state_keys, dict)
        expected_keys = [
            "scan_active",
            "scan_results",
            "selected_triplet_ids",
            "validation_stats",
            "scan_progress",
            "export_data",
            "scan_directory",
        ]
        for key in expected_keys:
            assert key in component._state_keys

    def test_render_delegation(
        self, component: ResultsGalleryComponent
    ) -> None:
        """Test that render delegates to the renderer properly."""
        assert component.renderer is not None

        # Mock the renderer to verify delegation
        with patch.object(component.renderer, "render_all") as mock_render:
            test_directory = Path("/test/path")
            component.render(scan_directory=test_directory)

            # Verify renderer.render_all was called with the directory
            mock_render.assert_called_once_with(test_directory)

    def test_component_modular_architecture(
        self, component: ResultsGalleryComponent
    ) -> None:
        """Test the modular architecture is properly set up."""
        # Verify all components are different instances
        assert component.state is not component.scanner_service
        assert component.state is not component.exporter_service
        assert component.actions is not component.event_handlers
        assert component.renderer is not component.actions

        # Verify proper dependency injection
        assert component.actions.state is component.state
        assert component.renderer.state is component.state
        assert component.renderer.actions is component.actions
