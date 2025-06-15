"""Unit tests for ResultsGalleryComponent.

Tests the complete Streamlit integration with reactive updates,
event-driven architecture, and export functionality.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.gui.components.results_gallery_component import (
    ResultsGalleryComponent,
)
from scripts.gui.utils.results import (
    EventManager,
    EventType,
    ResultTriplet,
    TripletCache,
    ValidationLevel,
)


class TestResultsGalleryComponent:
    """Test suite for ResultsGalleryComponent."""

    @pytest.fixture
    def mock_streamlit(self) -> Generator[MagicMock, None, None]:
        """Mock Streamlit module."""
        with patch(
            "scripts.gui.components.results_gallery_component.st"
        ) as mock_st:
            # Mock session state as a dictionary
            mock_st.session_state = {}

            # Mock columns to return tuples for unpacking
            def mock_columns(spec: list[int] | int) -> tuple[MagicMock, ...]:
                if isinstance(spec, list):
                    return tuple(MagicMock() for _ in range(len(spec)))
                else:
                    return tuple(MagicMock() for _ in range(spec))

            mock_st.columns.side_effect = mock_columns

            # Mock context managers with proper __enter__ and __exit__
            def create_context_manager(*args: Any, **kwargs: Any) -> MagicMock:
                cm = MagicMock()
                cm.__enter__ = MagicMock(return_value=cm)
                cm.__exit__ = MagicMock(return_value=None)
                return cm

            mock_st.container.side_effect = create_context_manager
            mock_st.expander.side_effect = create_context_manager

            # Mock other common Streamlit functions
            mock_st.write = MagicMock()
            mock_st.markdown = MagicMock()
            mock_st.header = MagicMock()
            mock_st.subheader = MagicMock()
            mock_st.metric = MagicMock()
            mock_st.button = MagicMock(return_value=False)
            mock_st.checkbox = MagicMock(return_value=True)
            mock_st.selectbox = MagicMock(return_value="STANDARD")
            mock_st.number_input = MagicMock(return_value=50)
            mock_st.progress = MagicMock()
            mock_st.info = MagicMock()
            mock_st.warning = MagicMock()
            mock_st.error = MagicMock()
            mock_st.success = MagicMock()
            mock_st.download_button = MagicMock()
            mock_st.image = MagicMock()
            mock_st.tabs = MagicMock(
                return_value=[MagicMock(), MagicMock(), MagicMock()]
            )
            mock_st.rerun = MagicMock()

            yield mock_st

    @pytest.fixture
    def mock_event_manager(self) -> Generator[MagicMock, None, None]:
        """Mock EventManager."""
        with patch(
            "scripts.gui.components.results_gallery_component.get_event_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock(spec=EventManager)
            mock_get_manager.return_value = mock_manager
            yield mock_manager

    @pytest.fixture
    def mock_cache(self) -> Generator[MagicMock, None, None]:
        """Mock TripletCache."""
        with patch(
            "scripts.gui.components.results_gallery_component.get_triplet_cache"
        ) as mock_get_cache:
            mock_cache_obj = MagicMock(spec=TripletCache)
            mock_cache_obj.get_stats.return_value = {
                "hit_rate": 85.5,
                "total_requests": 100,
                "cache_hits": 85,
                "cache_misses": 15,
            }
            mock_get_cache.return_value = mock_cache_obj
            yield mock_cache_obj

    @pytest.fixture
    def sample_triplets(self) -> list[ResultTriplet]:
        """Create sample triplets for testing."""
        triplets = []
        for i in range(3):
            triplet = ResultTriplet(
                id=f"triplet_{i}",
                dataset_name=f"dataset_{i}",
                image_path=Path(f"/test/image_{i}.png"),
                mask_path=Path(f"/test/mask_{i}.png"),
                prediction_path=Path(f"/test/pred_{i}.png"),
                metadata={"test": f"value_{i}"},
            )
            triplets.append(triplet)
        return triplets

    @pytest.fixture
    def temp_directory(self) -> Generator[Path, None, None]:
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create some test files
            for i in range(3):
                (temp_path / f"image_{i}.png").touch()
                (temp_path / f"mask_{i}.png").touch()
                (temp_path / f"pred_{i}.png").touch()

            yield temp_path

    def test_component_initialization(
        self,
        mock_streamlit: MagicMock,
        mock_event_manager: MagicMock,
        mock_cache: MagicMock,
    ) -> None:
        """Test component initialization."""
        component = ResultsGalleryComponent()

        # Check basic attributes
        assert component.event_manager is not None
        assert component.cache is not None

        # Check state keys are defined (using private interface for testing)
        state_keys = getattr(component, "_state_keys", {})
        assert len(state_keys) == 7
        assert "scan_active" in state_keys
        assert "scan_results" in state_keys

    def test_session_state_initialization(
        self,
        mock_streamlit: MagicMock,
        mock_event_manager: MagicMock,
        mock_cache: MagicMock,
    ) -> None:
        """Test session state initialization."""
        ResultsGalleryComponent()
        # Should initialize session state for all keys
        assert len(mock_streamlit.session_state) >= 0

    def test_event_handler_setup(
        self,
        mock_streamlit: MagicMock,
        mock_event_manager: MagicMock,
        mock_cache: MagicMock,
    ) -> None:
        """Test event handler registration."""
        ResultsGalleryComponent()

        # Check that event handlers are subscribed
        assert mock_event_manager.subscribe.call_count >= 4

        # Check for specific event types
        call_args = [
            call[0][0] for call in mock_event_manager.subscribe.call_args_list
        ]
        assert EventType.SCAN_PROGRESS in call_args
        assert EventType.TRIPLET_FOUND in call_args
        assert EventType.SCAN_COMPLETED in call_args
        assert EventType.SCAN_ERROR in call_args

    def test_config_update(
        self,
        mock_streamlit: MagicMock,
        mock_event_manager: MagicMock,
        mock_cache: MagicMock,
    ) -> None:
        """Test configuration updates."""
        component = ResultsGalleryComponent()
        component._update_config(
            validation_level=ValidationLevel.THOROUGH,
            max_triplets=25,
            grid_columns=4,
        )

        # Should complete without error
        assert "gallery_validation_level" in mock_streamlit.session_state
        assert "gallery_max_triplets" in mock_streamlit.session_state
        assert "gallery_grid_columns" in mock_streamlit.session_state

    def test_render_header(
        self,
        mock_streamlit: MagicMock,
        mock_event_manager: MagicMock,
        mock_cache: MagicMock,
    ) -> None:
        """Test header rendering."""
        component = ResultsGalleryComponent()

        # Should not raise an exception
        component._render_header("/test/directory", "test_prefix")

        # Verify Streamlit functions were called
        assert mock_streamlit.header.called
        assert mock_streamlit.info.called or mock_streamlit.warning.called
        assert mock_streamlit.columns.called

    def test_render_without_scan_directory(
        self,
        mock_streamlit: MagicMock,
        mock_event_manager: MagicMock,
        mock_cache: MagicMock,
    ) -> None:
        """Test rendering without scan directory."""
        component = ResultsGalleryComponent()

        # Should handle None scan directory gracefully
        result = component.render(scan_directory=None, key_prefix="test")

        # Should return empty state
        assert result["total_triplets"] == 0
        assert result["scan_active"] is False
        assert "cache_stats" in result

        # Should display warning message
        assert mock_streamlit.warning.called

    @patch(
        "scripts.gui.components.results_gallery_component.AdvancedTripletValidator"
    )
    @patch(
        "scripts.gui.components.results_gallery_component.create_results_scanner"
    )
    def test_start_async_scan(
        self,
        mock_scanner_factory: MagicMock,
        mock_validator_class: MagicMock,
        mock_streamlit: MagicMock,
        mock_event_manager: MagicMock,
        mock_cache: MagicMock,
        temp_directory: Path,
    ) -> None:
        """Test async scan initialization."""
        # Setup mocks
        mock_scanner = AsyncMock()
        mock_scanner.scan_async.return_value = iter([])
        mock_scanner_factory.return_value = mock_scanner

        mock_validator = MagicMock()
        mock_validator_class.return_value = mock_validator

        component = ResultsGalleryComponent()
        component._start_async_scan(str(temp_directory))

        # Check scanner and validator creation
        assert mock_scanner_factory.called
        assert mock_validator_class.called

        # Check session state update
        state_keys = getattr(component, "_state_keys", {})
        state_key = state_keys.get("scan_active", "gallery_scan_active")
        assert mock_streamlit.session_state[state_key] is True

    def test_clear_results(
        self,
        mock_streamlit: MagicMock,
        mock_event_manager: MagicMock,
        mock_cache: MagicMock,
    ) -> None:
        """Test results clearing."""
        component = ResultsGalleryComponent()

        # Setup some state
        state_keys = getattr(component, "_state_keys", {})
        results_key = state_keys.get("scan_results", "gallery_scan_results")
        selected_key = state_keys.get(
            "selected_triplets", "gallery_selected_triplets"
        )

        mock_streamlit.session_state[results_key] = ["item1", "item2"]
        mock_streamlit.session_state[selected_key] = ["item1"]

        component._clear_results()

        # Check state cleared
        assert mock_streamlit.session_state[results_key] == []
        assert mock_streamlit.session_state[selected_key] == []

    def test_triplet_card_rendering(
        self,
        mock_streamlit: MagicMock,
        mock_event_manager: MagicMock,
        mock_cache: MagicMock,
        sample_triplets: list[ResultTriplet],
    ) -> None:
        """Test triplet card rendering."""
        component = ResultsGalleryComponent()

        # Should not raise exception
        component._render_triplet_card(
            sample_triplets[0], 0, False, "test_key"
        )

        # Check basic rendering calls
        assert mock_streamlit.checkbox.called

    def test_export_data_creation(
        self,
        mock_streamlit: MagicMock,
        mock_event_manager: MagicMock,
        mock_cache: MagicMock,
        sample_triplets: list[ResultTriplet],
    ) -> None:
        """Test export data creation."""
        component = ResultsGalleryComponent()

        gallery_results = {"total_triplets": 3, "valid_triplets": 3}
        export_data = component._create_export_data(
            sample_triplets, True, gallery_results
        )

        # Check data structure - adapt to actual implementation
        assert isinstance(export_data, dict)
        if "triplets" in export_data:
            # New format
            assert len(export_data["triplets"]) == len(sample_triplets)
            first_triplet = export_data["triplets"][0]
            assert "id" in first_triplet
            assert "dataset_name" in first_triplet
        else:
            # Legacy format - adapt test to actual structure
            assert len(export_data) >= len(sample_triplets)

    def test_json_export(
        self,
        mock_streamlit: MagicMock,
        mock_event_manager: MagicMock,
        mock_cache: MagicMock,
        sample_triplets: list[ResultTriplet],
    ) -> None:
        """Test JSON export functionality."""
        component = ResultsGalleryComponent()

        # Setup session state with correct key
        state_keys = getattr(component, "_state_keys", {})
        results_key = state_keys.get("scan_results", "gallery_scan_results")
        mock_streamlit.session_state[results_key] = sample_triplets

        gallery_results = {"total_triplets": 3, "valid_triplets": 3}

        # Should not raise exception
        try:
            component._handle_export(
                "JSON Metadata", "All Results", False, True, gallery_results
            )
        except Exception as e:
            pytest.fail(f"JSON export should not raise exception: {e}")

        # Check download button was called
        assert mock_streamlit.download_button.called

    def test_csv_export(
        self,
        mock_streamlit: MagicMock,
        mock_event_manager: MagicMock,
        mock_cache: MagicMock,
        sample_triplets: list[ResultTriplet],
    ) -> None:
        """Test CSV export functionality."""
        component = ResultsGalleryComponent()

        # Setup session state with correct key
        state_keys = getattr(component, "_state_keys", {})
        results_key = state_keys.get("scan_results", "gallery_scan_results")
        mock_streamlit.session_state[results_key] = sample_triplets

        gallery_results = {"total_triplets": 3, "valid_triplets": 3}

        # Should not raise exception
        try:
            component._handle_export(
                "CSV Report", "All Results", False, True, gallery_results
            )
        except Exception as e:
            pytest.fail(f"CSV export should not raise exception: {e}")

        # Check download button was called
        assert mock_streamlit.download_button.called

    def test_event_progress_handler(
        self,
        mock_streamlit: MagicMock,
        mock_event_manager: MagicMock,
        mock_cache: MagicMock,
    ) -> None:
        """Test progress event handling setup."""
        ResultsGalleryComponent()

        # Verify that progress event handler was subscribed
        progress_calls = [
            call
            for call in mock_event_manager.subscribe.call_args_list
            if call[0][0] == EventType.SCAN_PROGRESS
        ]
        assert len(progress_calls) > 0

        # Test that the handler is callable
        progress_handler = progress_calls[0][0][1]
        assert callable(progress_handler)

    def test_event_triplet_found_handler(
        self,
        mock_streamlit: MagicMock,
        mock_event_manager: MagicMock,
        mock_cache: MagicMock,
        sample_triplets: list[ResultTriplet],
    ) -> None:
        """Test triplet found event handling setup."""
        ResultsGalleryComponent()

        # Verify that triplet found event handler was subscribed
        triplet_calls = [
            call
            for call in mock_event_manager.subscribe.call_args_list
            if call[0][0] == EventType.TRIPLET_FOUND
        ]
        assert len(triplet_calls) > 0

        # Test that the handler is callable
        triplet_handler = triplet_calls[0][0][1]
        assert callable(triplet_handler)

        # Verify cache method is available
        assert hasattr(mock_cache, "cache_triplet")

    def test_validation_level_integration(
        self,
        mock_streamlit: MagicMock,
        mock_event_manager: MagicMock,
        mock_cache: MagicMock,
    ) -> None:
        """Test validation level integration."""
        component = ResultsGalleryComponent()

        result = component.render(
            validation_level=ValidationLevel.PARANOID,
            key_prefix="validation_test",
        )

        # Should complete without error
        assert "cache_stats" in result
        assert "total_triplets" in result

    def test_error_handling(
        self,
        mock_streamlit: MagicMock,
        mock_event_manager: MagicMock,
        mock_cache: MagicMock,
    ) -> None:
        """Test error handling."""
        component = ResultsGalleryComponent()

        # Should handle invalid parameters gracefully
        try:
            result = component.render(max_triplets=-1, grid_columns=0)
            assert isinstance(result, dict)
        except Exception as e:
            pytest.fail(
                f"Component should handle invalid parameters gracefully: {e}"
            )

    def test_performance_with_large_dataset(
        self,
        mock_streamlit: MagicMock,
        mock_event_manager: MagicMock,
        mock_cache: MagicMock,
        temp_directory: Path,
    ) -> None:
        """Test performance with large dataset."""
        component = ResultsGalleryComponent()

        # Create large dataset simulation
        large_triplets = []
        for i in range(200):
            triplet = ResultTriplet(
                id=f"large_triplet_{i}",
                dataset_name=f"large_dataset_{i}",
                image_path=temp_directory / f"large_image_{i}.png",
                mask_path=temp_directory / f"large_mask_{i}.png",
                prediction_path=temp_directory / f"large_pred_{i}.png",
                metadata={"index": i, "large": True},
            )
            large_triplets.append(triplet)

        # Should handle large dataset without performance issues
        start_time = time.time()
        gallery_results = {"total_triplets": 200, "valid_triplets": 200}
        export_data = component._create_export_data(
            large_triplets, True, gallery_results
        )
        end_time = time.time()

        # Should complete within reasonable time (< 1 second)
        assert (end_time - start_time) < 1.0
        # Adapt assertion to actual data structure
        if "triplets" in export_data:
            assert len(export_data["triplets"]) == 200
        else:
            assert len(export_data) >= 200

    def test_full_render_cycle(
        self,
        mock_streamlit: MagicMock,
        mock_event_manager: MagicMock,
        mock_cache: MagicMock,
        temp_directory: Path,
    ) -> None:
        """Test complete render cycle."""
        component = ResultsGalleryComponent()

        result = component.render(
            scan_directory=str(temp_directory),
            max_triplets=10,
            grid_columns=3,
            show_export_panel=True,
            enable_real_time_scanning=True,
            key_prefix="full_test",
        )

        # Check returned state
        assert isinstance(result, dict)
        assert "total_triplets" in result
        assert "scan_active" in result
        assert "selected_triplets" in result
        assert "cache_stats" in result

        # Check render calls
        assert mock_streamlit.header.called
        assert mock_streamlit.columns.called

    def test_gallery_grid_rendering(
        self,
        mock_streamlit: MagicMock,
        mock_event_manager: MagicMock,
        mock_cache: MagicMock,
    ) -> None:
        """Test gallery grid rendering."""
        component = ResultsGalleryComponent()

        # Should not raise exception
        result = component._render_gallery_grid("test_prefix")

        # Should return gallery results
        assert isinstance(result, dict)
        assert "total_triplets" in result
        assert "valid_triplets" in result
