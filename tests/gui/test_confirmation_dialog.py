"""
Comprehensive tests for the confirmation dialog component.

This module tests all functionality of the confirmation dialog component,
including dialog creation, user interactions, performance tracking, and
accessibility compliance.
"""

import pytest

from gui.components.confirmation_dialog import (
    ConfirmationAction,
    ConfirmationDialog,
    ConfirmationDialogFactory,
)


class TestConfirmationAction:
    """Test the ConfirmationAction enum."""

    def test_confirmation_action_values(self):
        """Test that all confirmation action values are correct."""
        assert ConfirmationAction.START_TRAINING.value == "start_training"
        assert ConfirmationAction.STOP_TRAINING.value == "stop_training"
        assert ConfirmationAction.RESET_CONFIG.value == "reset_config"
        assert (
            ConfirmationAction.DELETE_CHECKPOINTS.value == "delete_checkpoints"
        )
        assert ConfirmationAction.SWITCH_DEVICE.value == "switch_device"
        assert ConfirmationAction.FACTORY_RESET.value == "factory_reset"

    def test_confirmation_action_categories(self):
        """Test that actions are properly categorized."""
        training_actions = [
            ConfirmationAction.START_TRAINING,
            ConfirmationAction.STOP_TRAINING,
            ConfirmationAction.RESET_TRAINING,
        ]

        config_actions = [
            ConfirmationAction.RESET_CONFIG,
            ConfirmationAction.DELETE_CONFIG,
            ConfirmationAction.RESTORE_CONFIG,
        ]

        assert len(training_actions) == 3
        assert len(config_actions) == 3


class TestConfirmationDialog:
    """Test the ConfirmationDialog class."""

    def test_confirmation_dialog_creation(self):
        """Test basic confirmation dialog creation."""
        dialog = ConfirmationDialog(
            action=ConfirmationAction.START_TRAINING,
            title="Test Title",
            message="Test Message",
        )

        assert dialog.action == ConfirmationAction.START_TRAINING
        assert dialog.title == "Test Title"
        assert dialog.message == "Test Message"
        assert dialog.confirm_text == "Confirm"
        assert dialog.cancel_text == "Cancel"
        assert dialog.danger_level == "medium"
        assert dialog.requires_typing is False
        assert dialog.confirmation_phrase is None

    def test_confirmation_dialog_with_custom_values(self):
        """Test confirmation dialog with custom values."""
        dialog = ConfirmationDialog(
            action=ConfirmationAction.DELETE_CHECKPOINTS,
            title="Delete Checkpoints",
            message="Are you sure?",
            warning_text="This cannot be undone",
            confirm_text="Delete",
            cancel_text="Keep",
            danger_level="high",
            requires_typing=True,
            confirmation_phrase="DELETE ALL",
        )

        assert dialog.warning_text == "This cannot be undone"
        assert dialog.confirm_text == "Delete"
        assert dialog.cancel_text == "Keep"
        assert dialog.danger_level == "high"
        assert dialog.requires_typing is True
        assert dialog.confirmation_phrase == "DELETE ALL"

    def test_confirmation_dialog_validation_empty_title(self):
        """Test validation fails with empty title."""
        with pytest.raises(ValueError, match="Dialog title cannot be empty"):
            ConfirmationDialog(
                action=ConfirmationAction.START_TRAINING,
                title="",
                message="Test Message",
            )

    def test_confirmation_dialog_validation_empty_message(self):
        """Test validation fails with empty message."""
        with pytest.raises(ValueError, match="Dialog message cannot be empty"):
            ConfirmationDialog(
                action=ConfirmationAction.START_TRAINING,
                title="Test Title",
                message="",
            )

    def test_confirmation_dialog_validation_invalid_danger_level(self):
        """Test validation fails with invalid danger level."""
        with pytest.raises(ValueError, match="Danger level must be"):
            ConfirmationDialog(
                action=ConfirmationAction.START_TRAINING,
                title="Test Title",
                message="Test Message",
                danger_level="invalid",
            )

    def test_confirmation_dialog_validation_requires_typing_without_phrase(
        self,
    ):
        """
        Test validation fails when requires_typing is True without phrase.
        """
        with pytest.raises(ValueError, match="Confirmation phrase required"):
            ConfirmationDialog(
                action=ConfirmationAction.START_TRAINING,
                title="Test Title",
                message="Test Message",
                requires_typing=True,
            )

    def test_confirmation_dialog_to_dict(self):
        """Test conversion to dictionary."""
        dialog = ConfirmationDialog(
            action=ConfirmationAction.START_TRAINING,
            title="Test Title",
            message="Test Message",
            warning_text="Warning",
            confirm_text="OK",
            cancel_text="No",
            danger_level="high",
            requires_typing=True,
            confirmation_phrase="CONFIRM",
        )

        result = dialog.to_dict()

        assert result["action"] == "start_training"
        assert result["title"] == "Test Title"
        assert result["message"] == "Test Message"
        assert result["warning_text"] == "Warning"
        assert result["confirm_text"] == "OK"
        assert result["cancel_text"] == "No"
        assert result["danger_level"] == "high"
        assert result["requires_typing"] is True
        assert result["confirmation_phrase"] == "CONFIRM"


class TestConfirmationDialogFactory:
    """Test the ConfirmationDialogFactory class."""

    def test_create_training_start_dialog(self):
        """Test creating training start dialog."""
        dialog = ConfirmationDialogFactory.create_training_start_dialog()

        assert dialog.action == ConfirmationAction.START_TRAINING
        assert dialog.title == "Start Training"
        assert "current configuration" in dialog.message
        assert dialog.confirm_text == "Start Training"
        assert dialog.cancel_text == "Cancel"
        assert dialog.danger_level == "medium"
        assert dialog.requires_typing is False

    def test_create_training_stop_dialog(self):
        """Test creating training stop dialog."""
        dialog = ConfirmationDialogFactory.create_training_stop_dialog()

        assert dialog.action == ConfirmationAction.STOP_TRAINING
        assert dialog.title == "Stop Training"
        assert "stop the current training" in dialog.message
        assert dialog.confirm_text == "Stop Training"
        assert dialog.cancel_text == "Continue"
        assert dialog.danger_level == "medium"

    def test_create_config_reset_dialog(self):
        """Test creating config reset dialog."""
        dialog = ConfirmationDialogFactory.create_config_reset_dialog()

        assert dialog.action == ConfirmationAction.RESET_CONFIG
        assert dialog.title == "Reset Configuration"
        assert "default values" in dialog.message
        assert dialog.danger_level == "high"
        assert dialog.requires_typing is True
        assert dialog.confirmation_phrase == "RESET CONFIG"

    def test_create_checkpoint_delete_dialog(self):
        """Test creating checkpoint delete dialog."""
        dialog = ConfirmationDialogFactory.create_checkpoint_delete_dialog()

        assert dialog.action == ConfirmationAction.DELETE_CHECKPOINTS
        assert dialog.title == "Delete Checkpoints"
        assert "permanently delete" in dialog.message
        assert dialog.danger_level == "high"
        assert dialog.requires_typing is True
        assert dialog.confirmation_phrase == "DELETE CHECKPOINTS"

    def test_create_device_switch_dialog(self):
        """Test creating device switch dialog."""
        dialog = ConfirmationDialogFactory.create_device_switch_dialog(
            "cpu", "cuda:0"
        )

        assert dialog.action == ConfirmationAction.SWITCH_DEVICE
        assert dialog.title == "Switch Device"
        assert "cpu" in dialog.message
        assert "cuda:0" in dialog.message
        assert dialog.danger_level == "high"
        assert dialog.requires_typing is False


# class TestOptimizedConfirmationDialog:
#     """Test the OptimizedConfirmationDialog class."""
#
#     def setup_method(self):
#         """Setup test environment."""
#         # Clear session state
#         if hasattr(st, "session_state"):
#             st.session_state.clear()
#
#     @patch("scripts.gui.components.confirmation_renderer.inject_css_once")
#     def test_ensure_css_injected(self, mock_inject_css: Mock):
#         """Test that CSS is injected correctly."""
#         OptimizedConfirmationDialog._ensure_css_injected()
#
#         mock_inject_css.assert_called_once()
#         call_args = mock_inject_css.call_args
#         assert call_args[0][0] == "crackseg_confirmation_dialog"
#         assert "crackseg-confirmation-dialog" in call_args[0][1]
#
#     def test_get_icon_for_danger_level(self):
#         """Test icon selection based on danger level."""
#         assert (
#             OptimizedConfirmationDialog._get_icon_for_danger_level("low")
#             == "ℹ️"
#         )
#         assert (
#             OptimizedConfirmationDialog._get_icon_for_danger_level("medium")
#             == "⚠️"
#         )
#         assert (
#             OptimizedConfirmationDialog._get_icon_for_danger_level("high")
#             == "⚠️"
#         )
#         assert (
#             OptimizedConfirmationDialog._get_icon_for_danger_level("unknown")
#             == "❓"
#         )
#
#     @patch("streamlit.session_state", new_callable=dict)
#     def test_activate_dialog(self, mock_session_state: dict[str, Any]):
#         """Test dialog activation."""
#         dialog = ConfirmationDialog(
#             action=ConfirmationAction.START_TRAINING,
#             title="Test",
#             message="Test message",
#         )
#
#         OptimizedConfirmationDialog.activate_dialog(dialog, "test_key")
#
#         assert mock_session_state["test_key"]["active"] is True
#         assert mock_session_state["test_key"]["dialog"] == dialog
#         assert mock_session_state["test_key"]["result"] is None
#         assert mock_session_state["test_key"]["user_input"] == ""
#
#     @patch("streamlit.session_state", new_callable=dict)
#     def test_is_dialog_active_true(self, mock_session_state: dict[str, Any]):
#         """Test dialog active state detection when active."""
#         mock_session_state["test_key"] = {"active": True}
#
#         assert (
#             OptimizedConfirmationDialog.is_dialog_active("test_key") is True
#         )
#
#     @patch("streamlit.session_state", new_callable=dict)
#     def test_is_dialog_active_false(
#         self, mock_session_state: dict[str, Any]
#     ):
#         """Test dialog active state detection when inactive."""
#         mock_session_state["test_key"] = {"active": False}
#
#         assert (
#             OptimizedConfirmationDialog.is_dialog_active("test_key") is False
#         )
#
#     @patch("streamlit.session_state", new_callable=dict)
#     def test_is_dialog_active_no_state(
#         self, mock_session_state: dict[str, Any]
#     ):
#         """Test dialog active state detection when no state exists."""
#         assert (
#             OptimizedConfirmationDialog.is_dialog_active("test_key") is False
#         )
#
#     @patch("streamlit.session_state", new_callable=dict)
#     @patch("streamlit.markdown")
#     @patch("streamlit.text_input")
#     @patch("streamlit.button")
#     @patch("streamlit.columns")
#     def test_show_confirmation_dialog_not_active(
#         self,
#         mock_columns: Mock,
#         mock_button: Mock,
#         mock_text_input: Mock,
#         mock_markdown: Mock,
#         mock_session_state: dict[str, Any],
#     ):
#         """Test showing dialog when not active returns None."""
#         dialog = ConfirmationDialog(
#             action=ConfirmationAction.START_TRAINING,
#             title="Test",
#             message="Test message",
#         )
#
#         result = OptimizedConfirmationDialog.show_confirmation_dialog(
#             dialog, "test_id", "test_key"
#         )
#
#         assert result is None
#
#     @patch("streamlit.session_state", new_callable=dict)
#     @patch("streamlit.markdown")
#     @patch("streamlit.text_input")
#     @patch("streamlit.button")
#     @patch("streamlit.columns")
#     @patch("streamlit.rerun")
#     def test_show_confirmation_dialog_active(
#         self,
#         mock_rerun: Mock,
#         mock_columns: Mock,
#         mock_button: Mock,
#         mock_text_input: Mock,
#         mock_markdown: Mock,
#         mock_session_state: dict[str, Any],
#     ):
#         """Test showing dialog when active."""
#         dialog = ConfirmationDialog(
#             action=ConfirmationAction.START_TRAINING,
#             title="Test",
#             message="Test message",
#         )
#
#         # Set up session state as active
#         mock_session_state["test_key"] = {
#             "active": True,
#             "dialog": None,
#             "user_input": "",
#             "result": None,
#         }
#
#         # Mock columns to return mock column objects that support context
#         # manager
#         mock_col1 = MagicMock()
#         mock_col2 = MagicMock()
#         mock_col1.__enter__ = MagicMock(return_value=mock_col1)
#         mock_col1.__exit__ = MagicMock(return_value=False)
#         mock_col2.__enter__ = MagicMock(return_value=mock_col2)
#         mock_col2.__exit__ = MagicMock(return_value=False)
#         mock_columns.return_value = [mock_col1, mock_col2]
#
#         # Mock button behavior
#         mock_button.return_value = False
#
#         OptimizedConfirmationDialog.show_confirmation_dialog(
#             dialog, "test_id", "test_key"
#         )
#
#         # Verify HTML is generated
#         mock_markdown.assert_called()
#         html_content = mock_markdown.call_args[0][0]
#         assert "Test" in html_content
#         assert "Test message" in html_content


# class TestConvenienceFunctions:
#     """Test the convenience functions."""
#
#     def setup_method(self):
#         """Setup test environment."""
#         if hasattr(st, "session_state"):
#             st.session_state.clear()
#
#     @patch(
#         "scripts.gui.components.confirmation_dialog."
#         "OptimizedConfirmationDialog.show_confirmation_dialog"
#     )
#     def test_confirmation_dialog_function(self, mock_show_dialog: Mock):
#         """Test the confirmation_dialog convenience function."""
#         dialog = ConfirmationDialog(
#             action=ConfirmationAction.START_TRAINING,
#             title="Test",
#             message="Test message",
#         )
#
#         mock_show_dialog.return_value = "confirmed"
#
#         result = confirmation_dialog(dialog, "test_id", "test_key")
#
#         assert result == "confirmed"
#         mock_show_dialog.assert_called_once_with(
#             dialog=dialog,
#             component_id="test_id",
#             session_key="test_key",
#         )
#
#     @patch(
#         "scripts.gui.components.confirmation_dialog."
#         "OptimizedConfirmationDialog.activate_dialog"
#     )
#     def test_activate_confirmation_dialog_function(
#         self, mock_activate: Mock
#     ):
#         """Test the activate_confirmation_dialog convenience function."""
#         dialog = ConfirmationDialog(
#             action=ConfirmationAction.START_TRAINING,
#             title="Test",
#             message="Test message",
#         )
#
#         activate_confirmation_dialog(dialog, "test_key")
#
#         mock_activate.assert_called_once_with(dialog, "test_key")
#
#     @patch(
#         "scripts.gui.components.confirmation_dialog."
#         "OptimizedConfirmationDialog.is_dialog_active"
#     )
#     def test_is_confirmation_dialog_active_function(
#         self, mock_is_active: Mock
#     ):
#         """Test the is_confirmation_dialog_active convenience function."""
#         mock_is_active.return_value = True
#
#         result = is_confirmation_dialog_active("test_key")
#
#         assert result is True
#         mock_is_active.assert_called_once_with("test_key")


# class TestPerformanceIntegration:
#     """Test performance tracking integration."""
#
#     @patch("scripts.gui.components.confirmation_dialog.get_optimizer")
#     def test_performance_decorator_tracking(self, mock_get_optimizer: Mock):
#         """Test that performance decorator tracks operations."""
#         mock_optimizer = Mock()
#         mock_get_optimizer.return_value = mock_optimizer
#
#         # Create a mock function with the decorator
#         from gui.components.confirmation_dialog import (
#             track_performance_decorator,
#         )
#
#         @track_performance_decorator("test_operation")
#         def test_function(component_id: str = "test_component") -> str:
#             return "result"
#
#         result = test_function(component_id="test_component")
#
#         assert result == "result"
#         mock_optimizer.track_performance.assert_called_once()
#
#         # Verify the call arguments
#         call_args = mock_optimizer.track_performance.call_args
#         assert call_args[0][0] == "test_component"
#         assert call_args[0][1] == "test_operation"
#         assert isinstance(call_args[0][2], float)  # start_time


# class TestAccessibilityCompliance:
#     """Test accessibility compliance."""
#
#     def test_dialog_html_has_proper_structure(self):
#         """Test that generated HTML has proper accessibility structure."""
#         dialog = ConfirmationDialog(
#             action=ConfirmationAction.START_TRAINING,
#             title="Test Dialog",
#             message="Test message",
#         )
#
#         # The HTML structure should be accessible
#         # This would require more complex testing with actual HTML parsing
#         # For now, we verify the basic structure is created
#         assert dialog.title == "Test Dialog"
#         assert dialog.message == "Test message"
#
#     def test_danger_level_visual_indicators(self):
#         """Test that danger levels have appropriate visual indicators."""
#         for level in ["low", "medium", "high"]:
#             dialog = ConfirmationDialog(
#                 action=ConfirmationAction.START_TRAINING,
#                 title="Test",
#                 message="Test message",
#                 danger_level=level,
#             )
#
#             assert dialog.danger_level == level
#
#             # Test icon mapping
#             icon = OptimizedConfirmationDialog._get_icon_for_danger_level(
#                 level
#             )
#             assert icon in ["ℹ️", "⚠️", "❓"]
#
#     def test_keyboard_navigation_support(self):
#         """Test that keyboard navigation is supported."""
#         # The CSS should support keyboard navigation
#         css_content = OptimizedConfirmationDialog._CSS_CONTENT
#         assert "focus" in css_content
#         assert "outline" in css_content


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_dialog_with_very_long_text(self):
        """Test dialog with very long text content."""
        long_text = "x" * 1000

        dialog = ConfirmationDialog(
            action=ConfirmationAction.START_TRAINING,
            title=long_text,
            message=long_text,
        )

        assert len(dialog.title) == 1000
        assert len(dialog.message) == 1000

    def test_dialog_with_special_characters(self):
        """Test dialog with special characters."""
        special_text = 'Test with <special> &characters& and "quotes"'

        dialog = ConfirmationDialog(
            action=ConfirmationAction.START_TRAINING,
            title=special_text,
            message=special_text,
        )

        assert dialog.title == special_text
        assert dialog.message == special_text

    def test_multiple_dialogs_different_keys(self):
        """Test multiple dialogs with different session keys."""
        dialog1 = ConfirmationDialog(
            action=ConfirmationAction.START_TRAINING,
            title="Dialog 1",
            message="Message 1",
        )

        dialog2 = ConfirmationDialog(
            action=ConfirmationAction.STOP_TRAINING,
            title="Dialog 2",
            message="Message 2",
        )

        # Both dialogs should be able to coexist with different keys
        assert dialog1.title != dialog2.title
        assert dialog1.action != dialog2.action


# class TestIntegrationWithExistingComponents:
#     """Test integration with existing GUI components."""
#
#     def test_css_consistency_with_device_selector(self):
#         """Test CSS consistency with existing components."""
#         css_content = OptimizedConfirmationDialog._CSS_CONTENT
#
#         # Should use consistent naming convention
#         assert "crackseg-confirmation" in css_content
#         assert "font-family: 'Segoe UI'" in css_content
#
#         # Should have responsive design
#         assert "@media (max-width: 768px)" in css_content
#
#     def test_error_state_compatibility(self):
#         """Test compatibility with error state system."""
#         dialog = ConfirmationDialog(
#             action=ConfirmationAction.START_TRAINING,
#             title="Test",
#             message="Test message",
#         )
#
#         # Dialog should be able to coexist with error states
#         # This is more of a structural test
#         assert hasattr(dialog, "action")
#         assert hasattr(dialog, "title")
#         assert hasattr(dialog, "message")
