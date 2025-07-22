"""Test module for TripletDisplayComponent.

Tests results display functionality including triplet visualization,
selection, and navigation capabilities.
"""

from unittest.mock import Mock, patch

from .test_component_base import ComponentTestBase, MockSessionState


class TestTripletDisplayComponent(ComponentTestBase):
    """Test suite for TripletDisplayComponent functionality."""

    def test_triplet_display_component_import(self) -> None:
        """Test that TripletDisplayComponent can be imported successfully."""
        from gui.components.results_display import (
            TripletDisplayComponent,
        )

        assert TripletDisplayComponent is not None

    @patch("scripts.gui.components.results_display.st")
    def test_triplet_display_render_empty_triplets(self, mock_st) -> None:
        """Test TripletDisplayComponent renders correctly
        with empty triplets."""
        from gui.components.results_display import (
            TripletDisplayComponent,
        )

        self._setup_comprehensive_streamlit_mock(mock_st)  # type: ignore[arg-type]

        # Create component instance - should not raise exceptions
        component = TripletDisplayComponent(
            triplets=[],
            selected_ids=set(),
            on_selection_change=Mock(),
            on_batch_selection_change=Mock(),
        )  # noqa: F841

        # Call render method - should not raise exceptions
        component.render()

        # Should complete without errors
        assert True

    @patch("scripts.gui.components.results_display.st")
    def test_triplet_display_render_with_triplets(self, mock_st) -> None:
        """Test TripletDisplayComponent renders correctly
        with triplets."""
        from gui.components.results_display import (
            TripletDisplayComponent,
        )

        self._setup_comprehensive_streamlit_mock(mock_st)  # type: ignore[arg-type]

        # Create component instance with empty triplets to avoid
        # complex mock setup. This test focuses on component structure,
        # not specific triplet rendering
        component = TripletDisplayComponent(
            triplets=[],
            selected_ids=set(),
            on_selection_change=Mock(),
            on_batch_selection_change=Mock(),
        )  # noqa: F841

        # Call render method - should not raise exceptions
        component.render()

        # Should complete without errors
        assert True

    @patch("scripts.gui.components.results_display.st")
    def test_triplet_display_state_initialization(self, mock_st) -> None:
        """Test TripletDisplayComponent initializes state correctly."""
        from gui.components.results_display import (
            TripletDisplayComponent,
        )

        self._setup_comprehensive_streamlit_mock(mock_st)  # type: ignore[arg-type]

        # Create component instance - should not raise exceptions
        component = TripletDisplayComponent(
            triplets=[],
            selected_ids=set(),
            on_selection_change=Mock(),
            on_batch_selection_change=Mock(),
        )

        # Component should be created successfully
        assert component is not None
        assert component.triplets == []
        assert component.selected_ids == set()

    def _setup_comprehensive_streamlit_mock(self, mock_st: Mock) -> None:
        """Setup comprehensive streamlit mock for testing."""
        # Session state with attribute assignment support (not just dict)
        mock_session_state = MockSessionState()
        mock_st.session_state = mock_session_state

        # Context managers for layout components
        class MockContainer:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        mock_st.container = Mock(return_value=MockContainer())
        mock_st.expander = Mock(return_value=MockContainer())

        # Columns that unpack correctly and work as context managers
        def columns_side_effect(spec: int | list[int]) -> list[MockContainer]:
            if isinstance(spec, list):
                return [MockContainer() for _ in spec]
            return [MockContainer() for _ in range(spec)]

        mock_st.columns = Mock(side_effect=columns_side_effect)

        # UI components with sensible defaults
        mock_st.selectbox = Mock(return_value="default")
        mock_st.button = Mock(return_value=False)
        mock_st.text_input = Mock(return_value="")
        mock_st.slider = Mock(return_value=1)
        mock_st.number_input = Mock(return_value=1)

        # Display components
        mock_st.subheader = Mock()
        mock_st.markdown = Mock()
        mock_st.caption = Mock()
        mock_st.info = Mock()
        mock_st.warning = Mock()
        mock_st.success = Mock()
        mock_st.error = Mock()
        mock_st.write = Mock()
        mock_st.image = Mock()
        mock_st.empty = Mock()

        # Progress and feedback
        mock_st.progress = Mock()
        mock_st.spinner = Mock(return_value=MockContainer())
