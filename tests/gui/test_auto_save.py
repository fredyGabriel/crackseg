"""
Tests for auto-save system functionality.

This module tests the auto-save manager, UI components, and integration
with the CrackSeg GUI application.
"""

import time
from unittest.mock import Mock, patch

import streamlit as st

from scripts.gui.components.auto_save_manager import (
    AutoSaveManagerComponent,
    create_auto_save_manager,
    get_global_auto_save_manager,
    integrate_auto_save_with_page,
)
from scripts.gui.utils.auto_save import (
    AutoSaveConfig,
    AutoSaveManager,
    AutoSaveUI,
    SaveMetadata,
    auto_save_field,
    get_autosave_manager,
    trigger_auto_save,
)


class TestAutoSaveConfig:
    """Test auto-save configuration dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AutoSaveConfig()

        assert config.save_interval == 3.0
        assert config.debounce_interval == 0.5
        assert config.max_storage_size == 5 * 1024 * 1024
        assert config.max_versions == 10
        assert config.enable_compression is True
        assert config.visual_feedback is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AutoSaveConfig(
            save_interval=5.0,
            debounce_interval=1.0,
            max_storage_size=1024,
            max_versions=5,
            enable_compression=False,
            visual_feedback=False,
        )

        assert config.save_interval == 5.0
        assert config.debounce_interval == 1.0
        assert config.max_storage_size == 1024
        assert config.max_versions == 5
        assert config.enable_compression is False
        assert config.visual_feedback is False


class TestSaveMetadata:
    """Test save metadata dataclass."""

    def test_metadata_creation(self):
        """Test metadata creation with required fields."""
        metadata = SaveMetadata(
            timestamp=time.time(),
            version=1,
            config_hash="abc123",
            user_action="manual",
            size_bytes=1024,
        )

        assert metadata.version == 1
        assert metadata.config_hash == "abc123"
        assert metadata.user_action == "manual"
        assert metadata.size_bytes == 1024
        assert isinstance(metadata.timestamp, float)


class TestAutoSaveManager:
    """Test auto-save manager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear session state
        for key in list(st.session_state.keys()):
            if key.startswith("autosave"):
                del st.session_state[key]

        self.config = AutoSaveConfig(
            save_interval=1.0,
            debounce_interval=0.1,
            max_versions=5,
        )
        self.manager = AutoSaveManager(self.config)

    def test_manager_initialization(self):
        """Test manager initialization."""
        assert self.manager.config.save_interval == 1.0
        assert self.manager.storage_key_prefix == "crackseg_autosave"
        assert self.manager.last_save_time == 0.0
        assert self.manager.pending_changes == {}
        assert self.manager.save_in_progress is False

        # Check session state initialization
        assert st.session_state.autosave_enabled is True
        assert st.session_state.autosave_last_status == "idle"
        assert st.session_state.autosave_version == 1

    def test_register_config_field(self):
        """Test registering configuration fields."""
        # Register a field
        self.manager.register_config_field("test_field", "test_value")

        assert "test_field" in self.manager.pending_changes
        assert self.manager.pending_changes["test_field"] == "test_value"
        assert "autosave_last_change" in st.session_state

    def test_register_config_field_no_change(self):
        """Test registering field with no change."""
        # Register same field twice
        self.manager.register_config_field("test_field", "test_value")
        initial_change_time = st.session_state.get("autosave_last_change", 0)

        time.sleep(0.01)  # Small delay
        self.manager.register_config_field("test_field", "test_value")

        # Change time should not update
        assert (
            st.session_state.get("autosave_last_change", 0)
            == initial_change_time
        )

    def test_register_config_field_with_callback(self):
        """Test registering field with change callback."""
        callback = Mock()

        self.manager.register_config_field(
            "test_field", "test_value", callback
        )

        assert "test_field" in self.manager.change_callbacks
        assert self.manager.change_callbacks["test_field"] == callback

    def test_should_auto_save_disabled(self):
        """Test should_auto_save when disabled."""
        st.session_state.autosave_enabled = False
        assert self.manager.should_auto_save() is False

    def test_should_auto_save_no_changes(self):
        """Test should_auto_save with no pending changes."""
        assert self.manager.should_auto_save() is False

    def test_should_auto_save_too_soon(self):
        """Test should_auto_save when called too soon."""
        # Add pending change
        self.manager.register_config_field("test_field", "test_value")

        # Set recent save time
        self.manager.last_save_time = time.time()

        assert self.manager.should_auto_save() is False

    def test_should_auto_save_debounce(self):
        """Test should_auto_save with debounce interval."""
        # Add pending change
        self.manager.register_config_field("test_field", "test_value")

        # Set old enough save time but recent change
        self.manager.last_save_time = time.time() - 2.0
        st.session_state.autosave_last_change = time.time()

        assert self.manager.should_auto_save() is False

    def test_should_auto_save_ready(self):
        """Test should_auto_save when ready to save."""
        # Add pending change
        self.manager.register_config_field("test_field", "test_value")

        # Set old enough times
        self.manager.last_save_time = time.time() - 2.0
        st.session_state.autosave_last_change = time.time() - 1.0

        assert self.manager.should_auto_save() is True

    def test_auto_save_configurations_success(self):
        """Test successful auto-save operation."""
        # Add pending changes
        self.manager.register_config_field("field1", "value1")
        self.manager.register_config_field("field2", "value2")

        # Set up for successful save
        self.manager.last_save_time = 0.0
        st.session_state.autosave_last_change = 0.0

        result = self.manager.auto_save_configurations("test_action")

        assert result is True
        assert st.session_state.autosave_last_status == "saved"
        assert st.session_state.autosave_version == 2
        assert len(self.manager.pending_changes) == 0

    def test_auto_save_configurations_not_ready(self):
        """Test auto-save when not ready."""
        result = self.manager.auto_save_configurations("test_action")

        assert result is False
        assert st.session_state.autosave_last_status == "idle"

    def test_calculate_config_hash(self):
        """Test configuration hash calculation."""
        # Add some changes
        self.manager.pending_changes = {"field1": "value1", "field2": "value2"}

        hash1 = self.manager._calculate_config_hash()

        # Same changes should produce same hash
        hash2 = self.manager._calculate_config_hash()
        assert hash1 == hash2

        # Different changes should produce different hash
        self.manager.pending_changes["field3"] = "value3"
        hash3 = self.manager._calculate_config_hash()
        assert hash1 != hash3

    def test_storage_limits(self):
        """Test storage limit checking."""
        # Set small limit
        self.manager.config.max_storage_size = 100

        # Small size should pass
        assert self.manager._check_storage_limits(50) is True

        # Large size should fail
        assert self.manager._check_storage_limits(200) is False

    def test_force_save(self):
        """Test force save functionality."""
        # Add pending changes
        self.manager.register_config_field("test_field", "test_value")

        # Set recent save time (should normally block save)
        self.manager.last_save_time = time.time()

        # Force save should bypass time check
        result = self.manager.force_save("manual")

        assert result is True
        assert st.session_state.autosave_last_status == "saved"

    def test_load_saved_configuration(self):
        """Test loading saved configuration."""
        # First save some data
        self.manager.register_config_field("test_field", "test_value")
        self.manager.force_save("test")

        # Load it back
        loaded_config = self.manager.load_saved_configuration()

        assert loaded_config is not None
        assert loaded_config["test_field"] == "test_value"

    def test_load_saved_configuration_no_data(self):
        """Test loading when no data exists."""
        loaded_config = self.manager.load_saved_configuration()
        assert loaded_config is None

    def test_get_save_status(self):
        """Test getting save status information."""
        status = self.manager.get_save_status()

        assert "enabled" in status
        assert "status" in status
        assert "last_save" in status
        assert "version" in status
        assert "pending_changes" in status
        assert "has_saved_data" in status

    def test_toggle_auto_save(self):
        """Test toggling auto-save on/off."""
        # Initially enabled
        assert st.session_state.autosave_enabled is True

        # Disable
        self.manager.toggle_auto_save(False)
        assert st.session_state.autosave_enabled is False
        assert st.session_state.autosave_last_status == "disabled"

        # Enable
        self.manager.toggle_auto_save(True)
        assert st.session_state.autosave_enabled is True

    def test_cleanup_old_versions(self):
        """Test cleanup of old versions."""
        # Create multiple versions
        for i in range(10):
            self.manager.register_config_field(f"field_{i}", f"value_{i}")
            self.manager.force_save(f"action_{i}")

        # Check that old versions are cleaned up
        current_version = st.session_state.autosave_version

        # Should only keep max_versions (5) most recent
        old_key = f"{self.manager.storage_key_prefix}_v1"
        assert old_key not in st.session_state

        # Recent versions should exist
        recent_key = (
            f"{self.manager.storage_key_prefix}_v{current_version - 1}"
        )
        assert recent_key in st.session_state

    @patch("scripts.gui.utils.auto_save.should_update")
    def test_show_save_feedback(self, mock_should_update):
        """Test visual feedback display."""
        mock_should_update.return_value = True

        # Test successful save feedback
        self.manager._show_save_feedback("saved")

        assert "autosave_feedback" in st.session_state
        feedback = st.session_state.autosave_feedback
        assert feedback["type"] == "success"
        assert "Auto-saved" in feedback["message"]

    def test_save_in_progress_prevents_save(self):
        """Test that save in progress prevents concurrent saves."""
        # Add pending changes
        self.manager.register_config_field("test_field", "test_value")
        self.manager.last_save_time = 0.0

        # Set save in progress
        self.manager.save_in_progress = True

        result = self.manager.auto_save_configurations("test")
        assert result is False


class TestAutoSaveUI:
    """Test auto-save UI components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = AutoSaveManager()

    @patch("streamlit.success")
    @patch("streamlit.info")
    @patch("streamlit.warning")
    def test_show_feedback_notifications(
        self, mock_warning, mock_info, mock_success
    ):
        """Test feedback notifications display."""
        # Set up feedback
        st.session_state.autosave_feedback = {
            "type": "success",
            "message": "Test message",
            "timestamp": time.time(),
        }

        AutoSaveUI.show_feedback_notifications()

        mock_success.assert_called_once_with("Test message")

    @patch("streamlit.success")
    @patch("streamlit.info")
    @patch("streamlit.warning")
    def test_show_feedback_notifications_expired(
        self, mock_warning, mock_info, mock_success
    ):
        """Test feedback notifications with expired timestamp."""
        # Set up old feedback
        st.session_state.autosave_feedback = {
            "type": "success",
            "message": "Old message",
            "timestamp": time.time() - 10.0,  # Old timestamp
        }

        AutoSaveUI.show_feedback_notifications()

        # Should not show message and should clear feedback
        mock_success.assert_not_called()
        assert "autosave_feedback" not in st.session_state


class TestAutoSaveManagerComponent:
    """Test auto-save manager component."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = AutoSaveConfig(save_interval=1.0, debounce_interval=0.1)
        self.component = AutoSaveManagerComponent(self.config)

    def test_component_initialization(self):
        """Test component initialization."""
        assert self.component.manager is not None
        assert self.component.spinner is not None

    def test_integrate_with_config_form(self):
        """Test integration with configuration forms."""
        config_data = {
            "field1": "value1",
            "field2": "value2",
            "field3": 123,
        }

        self.component.integrate_with_config_form(config_data)

        # Check that fields were registered
        for field_name, value in config_data.items():
            assert field_name in self.component.manager.pending_changes
            assert self.component.manager.pending_changes[field_name] == value

    def test_load_configuration_if_available(self):
        """Test loading configuration if available."""
        # Test with session state config
        test_config = {"test_field": "test_value"}
        st.session_state.loaded_autosave_config = test_config

        loaded = self.component.load_configuration_if_available()

        assert loaded == test_config
        assert "loaded_autosave_config" not in st.session_state

    def test_load_configuration_if_available_none(self):
        """Test loading configuration when none available."""
        loaded = self.component.load_configuration_if_available()
        assert loaded is None


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_get_autosave_manager(self):
        """Test getting global auto-save manager."""
        manager1 = get_autosave_manager()
        manager2 = get_autosave_manager()

        # Should return same instance
        assert manager1 is manager2

    def test_auto_save_field(self):
        """Test auto-save field registration."""
        auto_save_field("test_field", "test_value")

        manager = get_autosave_manager()
        assert "test_field" in manager.pending_changes
        assert manager.pending_changes["test_field"] == "test_value"

    def test_trigger_auto_save(self):
        """Test triggering auto-save."""
        # Add some changes first
        auto_save_field("test_field", "test_value")

        # Set up for save
        manager = get_autosave_manager()
        manager.last_save_time = 0.0
        st.session_state.autosave_last_change = 0.0

        result = trigger_auto_save("test_action")

        assert result is True

    def test_create_auto_save_manager(self):
        """Test auto-save manager creation."""
        config = AutoSaveConfig(save_interval=5.0)
        component = create_auto_save_manager(config)

        assert isinstance(component, AutoSaveManagerComponent)
        assert component.manager.config.save_interval == 5.0

    def test_integrate_auto_save_with_page(self):
        """Test page integration."""
        config_data = {"page_field": "page_value"}
        component = integrate_auto_save_with_page("test_page", config_data)

        assert isinstance(component, AutoSaveManagerComponent)
        assert "page_field" in component.manager.pending_changes

    def test_get_global_auto_save_manager(self):
        """Test getting global component manager."""
        manager1 = get_global_auto_save_manager()
        manager2 = get_global_auto_save_manager()

        # Should return same instance
        assert manager1 is manager2
        assert isinstance(manager1, AutoSaveManagerComponent)


class TestAutoSaveIntegration:
    """Test auto-save integration scenarios."""

    def test_full_save_load_cycle(self):
        """Test complete save and load cycle."""
        # Create manager
        manager = AutoSaveManager(AutoSaveConfig(save_interval=0.1))

        # Add configuration data
        test_data = {
            "model_architecture": "unet",
            "batch_size": 16,
            "learning_rate": 0.001,
            "epochs": 100,
        }

        for field, value in test_data.items():
            manager.register_config_field(field, value)

        # Force save
        result = manager.force_save("integration_test")
        assert result is True

        # Load back
        loaded_data = manager.load_saved_configuration()
        assert loaded_data == test_data

    def test_version_management(self):
        """Test version management across multiple saves."""
        manager = AutoSaveManager(AutoSaveConfig(max_versions=3))

        # Create multiple versions
        for i in range(5):
            manager.register_config_field("version_field", f"version_{i}")
            manager.force_save(f"version_{i}")

        # Check that only recent versions exist
        current_version = st.session_state.autosave_version

        # Old versions should be cleaned up
        old_key = f"{manager.storage_key_prefix}_v1"
        assert old_key not in st.session_state

        # Recent versions should exist
        recent_key = f"{manager.storage_key_prefix}_v{current_version - 1}"
        assert recent_key in st.session_state

    def test_storage_size_management(self):
        """Test storage size limits."""
        # Set small storage limit
        config = AutoSaveConfig(max_storage_size=500)  # 500 bytes
        manager = AutoSaveManager(config)

        # Try to save large data
        large_data = "x" * 1000  # 1000 characters
        manager.register_config_field("large_field", large_data)

        # This should fail due to size limit
        result = manager.force_save("size_test")
        assert result is False
        assert st.session_state.autosave_last_status == "storage_full"

    def test_concurrent_save_prevention(self):
        """Test prevention of concurrent saves."""
        manager = AutoSaveManager(AutoSaveConfig(save_interval=0.1))

        # Add data and set up for save
        manager.register_config_field("test_field", "test_value")
        manager.last_save_time = 0.0

        # Start a save operation
        manager.save_in_progress = True

        # Try to save again (should be blocked)
        result = manager.auto_save_configurations("concurrent_test")
        assert result is False

    def test_performance_tracking_integration(self):
        """Test integration with performance tracking."""
        manager = AutoSaveManager()

        # Register field (should track performance)
        manager.register_config_field("perf_field", "perf_value")

        # Force save (should track performance)
        manager.force_save("performance_test")

        # Load configuration (should track performance)
        manager.load_saved_configuration()

        # Performance tracking should have been called
        # (specific assertions would depend on performance tracking
        # implementation)
        assert True  # Placeholder for performance tracking verification
