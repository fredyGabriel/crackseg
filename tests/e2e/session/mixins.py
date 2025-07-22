"""
Session management mixins for BaseE2ETest integration. This module
provides mixins that integrate session state management capabilities
with the BaseE2ETest class, following the established mixin pattern
used throughout the E2E testing framework.
"""

import logging
from typing import Any

from selenium.webdriver.remote.webdriver import WebDriver

from .cookie_manager import CookieData, CookieManager
from .state_manager import SessionSnapshot, StateManager, StateValidationError
from .storage_manager import StorageManager, StorageType
from .streamlit_session import StreamlitSessionManager

logger = logging.getLogger(__name__)


class SessionManagementMixin:
    """
    Main session management mixin for BaseE2ETest integration. Provides
    comprehensive session state management including cookies, storage, and
    state validation for E2E tests.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize session management mixin."""
        super().__init__(*args, **kwargs)
        self._session_manager = StateManager()
        self._cookie_manager = CookieManager()
        self._storage_manager = StorageManager()

        if hasattr(self, "log_test_step"):
            self.log_test_step(  # type: ignore[attr-defined]
                "Session management initialized"
            )

    # Cookie management methods
    def set_cookie(
        self,
        driver: WebDriver,
        name: str,
        value: str,
        **kwargs: Any,
    ) -> bool:
        """
        Set a cookie with comprehensive options. Args: driver: WebDriver
        instance name: Cookie name value: Cookie value **kwargs: Additional
        cookie options Returns: True if cookie was set successfully
        """
        success = self._cookie_manager.set_cookie(
            driver, name, value, **kwargs
        )

        if hasattr(self, "log_test_step"):
            status = "succeeded" if success else "failed"
            self.log_test_step(  # type: ignore[attr-defined]
                f"Set cookie {name}", f"Operation {status}"
            )

        return success

    def get_cookie(self, driver: WebDriver, name: str) -> CookieData | None:
        """
        Get a specific cookie by name. Args: driver: WebDriver instance name:
        Cookie name to retrieve Returns: CookieData object if found, None
        otherwise
        """
        cookie = self._cookie_manager.get_cookie(driver, name)

        if hasattr(self, "log_test_step"):
            status = "found" if cookie else "not found"
            self.log_test_step(  # type: ignore[attr-defined]
                f"Get cookie {name}", f"Cookie {status}"
            )

        return cookie

    def delete_cookie(self, driver: WebDriver, name: str) -> bool:
        """
        Delete a specific cookie. Args: driver: WebDriver instance name:
        Cookie name to delete Returns: True if cookie was deleted successfully
        """
        success = self._cookie_manager.delete_cookie(driver, name)

        if hasattr(self, "log_test_step"):
            status = "succeeded" if success else "failed"
            self.log_test_step(  # type: ignore[attr-defined]
                f"Delete cookie {name}", f"Operation {status}"
            )

        return success

    def backup_cookies(self, driver: WebDriver, key: str = "default") -> bool:
        """
        Backup current cookies for later restoration. Args: driver: WebDriver
        instance key: Backup identifier key Returns: True if backup was
        successful
        """
        success = self._cookie_manager.backup_cookies(driver, key)

        if hasattr(self, "log_test_step"):
            status = "succeeded" if success else "failed"
            self.log_test_step(  # type: ignore[attr-defined]
                f"Backup cookies with key {key}", f"Operation {status}"
            )

        return success

    def restore_cookies(self, driver: WebDriver, key: str = "default") -> bool:
        """
        Restore previously backed up cookies. Args: driver: WebDriver instance
        key: Backup identifier key Returns: True if restoration was successful
        """
        success = self._cookie_manager.restore_cookies(driver, key)

        if hasattr(self, "log_test_step"):
            status = "succeeded" if success else "failed"
            self.log_test_step(  # type: ignore[attr-defined]
                f"Restore cookies with key {key}", f"Operation {status}"
            )

        return success

    # Storage management methods
    def set_storage_item(
        self,
        driver: WebDriver,
        key: str,
        value: Any,
        storage_type: StorageType = StorageType.LOCAL,
    ) -> bool:
        """
        Set an item in the specified storage. Args: driver: WebDriver instance
        key: Storage key value: Value to store storage_type: Type of storage
        Returns: True if item was set successfully
        """
        success = self._storage_manager.set_item(
            driver, key, value, storage_type
        )

        if hasattr(self, "log_test_step"):
            status = "succeeded" if success else "failed"
            self.log_test_step(  # type: ignore[attr-defined]
                f"Set {storage_type.value} item {key}", f"Operation {status}"
            )

        return success

    def get_storage_item(
        self,
        driver: WebDriver,
        key: str,
        storage_type: StorageType = StorageType.LOCAL,
        default: Any = None,
    ) -> Any:
        """
        Get an item from the specified storage. Args: driver: WebDriver
        instance key: Storage key storage_type: Type of storage default:
        Default value if key not found Returns: Deserialized value or default
        if not found
        """
        value = self._storage_manager.get_item(
            driver, key, storage_type, default
        )

        if hasattr(self, "log_test_step"):
            status = "found" if value != default else "not found"
            self.log_test_step(  # type: ignore[attr-defined]
                f"Get {storage_type.value} item {key}", f"Item {status}"
            )

        return value

    def remove_storage_item(
        self,
        driver: WebDriver,
        key: str,
        storage_type: StorageType = StorageType.LOCAL,
    ) -> bool:
        """
        Remove an item from the specified storage. Args: driver: WebDriver
        instance key: Storage key to remove storage_type: Type of storage
        Returns: True if item was removed successfully
        """
        success = self._storage_manager.remove_item(driver, key, storage_type)

        if hasattr(self, "log_test_step"):
            status = "succeeded" if success else "failed"
            self.log_test_step(  # type: ignore[attr-defined]
                f"Remove {storage_type.value} item {key}",
                f"Operation {status}",
            )

        return success

    # Session state management methods
    def capture_session_snapshot(
        self,
        driver: WebDriver,
        include_metadata: bool = True,
    ) -> SessionSnapshot:
        """
        Capture complete session state snapshot. Args: driver: WebDriver
        instance include_metadata: Whether to include additional metadata
        Returns: SessionSnapshot containing complete state
        """
        snapshot = self._session_manager.capture_session_snapshot(
            driver, include_metadata
        )

        if hasattr(self, "log_test_step"):
            self.log_test_step(  # type: ignore[attr-defined]
                "Captured session snapshot", f"Timestamp: {snapshot.timestamp}"
            )

        return snapshot

    def validate_session_state(
        self,
        driver: WebDriver,
        expected_state: dict[str, Any],
        strict: bool = True,
        timeout: float = 5.0,
    ) -> bool:
        """
        Validate current session state against expected values. Args: driver:
        WebDriver instance expected_state: Expected state values strict:
        Whether to require exact matches timeout: Timeout for state
        stabilization Returns: True if validation passes
        """
        try:
            success = self._session_manager.validate_session_state(
                driver, expected_state, strict, timeout
            )

            if hasattr(self, "log_assertion"):
                self.log_assertion(  # type: ignore[attr-defined]
                    "Session state validation",
                    success,
                    f"Strict mode: {strict}",
                )

            return success

        except StateValidationError as e:
            if hasattr(self, "log_assertion"):
                self.log_assertion(  # type: ignore[attr-defined]
                    "Session state validation", False, f"Validation error: {e}"
                )

            if strict:
                raise
            return False

    def restore_session_state(
        self,
        driver: WebDriver,
        snapshot: SessionSnapshot,
        navigate_to_url: bool = True,
    ) -> bool:
        """
        Restore session state from snapshot. Args: driver: WebDriver instance
        snapshot: Session snapshot to restore navigate_to_url: Whether to
        navigate to snapshot URL Returns: True if restoration was successful
        """
        success = self._session_manager.restore_session_state(
            driver, snapshot, navigate_to_url
        )

        if hasattr(self, "log_test_step"):
            status = "succeeded" if success else "failed"
            self.log_test_step(  # type: ignore[attr-defined]
                "Restore session state",
                f"Operation {status}, URL: {snapshot.url}",
            )

        return success


class MultiTabSessionMixin:
    """
    Multi-tab session management mixin. Provides utilities for managing
    session state across multiple browser tabs and windows, including tab
    coordination and state synchronization.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize multi-tab session mixin."""
        super().__init__(*args, **kwargs)
        self._tab_handles: dict[str, str] = {}
        self._tab_snapshots: dict[str, SessionSnapshot] = {}

        if hasattr(self, "log_test_step"):
            self.log_test_step(  # type: ignore[attr-defined]
                "Multi-tab session management initialized"
            )

    def open_new_tab(
        self,
        driver: WebDriver,
        url: str,
        tab_name: str | None = None,
    ) -> str:
        """
        Open a new tab and navigate to URL. Args: driver: WebDriver instance
        url: URL to navigate to in new tab tab_name: Optional name for the tab
        Returns: Tab handle for the new tab
        """
        # Store current tab handle
        original_handle = driver.current_window_handle

        # Open new tab
        driver.execute_script("window.open('');")

        # Switch to new tab
        new_handles = driver.window_handles
        new_handle = [h for h in new_handles if h != original_handle][-1]
        driver.switch_to.window(new_handle)

        # Navigate to URL
        driver.get(url)

        # Store tab handle with name
        if tab_name:
            self._tab_handles[tab_name] = new_handle

        if hasattr(self, "log_test_step"):
            self.log_test_step(  # type: ignore[attr-defined]
                f"Opened new tab: {tab_name or 'unnamed'}", f"URL: {url}"
            )

        return new_handle

    def switch_to_tab(
        self,
        driver: WebDriver,
        tab_identifier: str,
    ) -> bool:
        """
        Switch to a specific tab. Args: driver: WebDriver instance
        tab_identifier: Tab name or handle Returns: True if switch was
        successful
        """
        try:
            # Try as tab name first
            if tab_identifier in self._tab_handles:
                handle = self._tab_handles[tab_identifier]
            else:
                # Use as handle directly
                handle = tab_identifier

            driver.switch_to.window(handle)

            if hasattr(self, "log_test_step"):
                self.log_test_step(  # type: ignore[attr-defined]
                    f"Switched to tab: {tab_identifier}"
                )

            return True

        except Exception as e:
            if hasattr(self, "log_test_step"):
                self.log_test_step(  # type: ignore[attr-defined]
                    f"Failed to switch to tab: {tab_identifier}", f"Error: {e}"
                )
            return False

    def close_tab(
        self,
        driver: WebDriver,
        tab_identifier: str,
        switch_to_remaining: bool = True,
    ) -> bool:
        """
        Close a specific tab. Args: driver: WebDriver instance tab_identifier:
        Tab name or handle switch_to_remaining: Whether to switch to remaining
        tab Returns: True if close was successful
        """
        try:
            # Get handle
            if tab_identifier in self._tab_handles:
                handle = self._tab_handles[tab_identifier]
                del self._tab_handles[tab_identifier]
            else:
                handle = tab_identifier

            # Switch to tab and close it
            driver.switch_to.window(handle)
            driver.close()

            # Switch to remaining tab if requested
            if switch_to_remaining and driver.window_handles:
                driver.switch_to.window(driver.window_handles[0])

            if hasattr(self, "log_test_step"):
                self.log_test_step(  # type: ignore[attr-defined]
                    f"Closed tab: {tab_identifier}"
                )

            return True

        except Exception as e:
            if hasattr(self, "log_test_step"):
                self.log_test_step(  # type: ignore[attr-defined]
                    f"Failed to close tab: {tab_identifier}", f"Error: {e}"
                )
            return False


class StreamlitSessionMixin:
    """
    Streamlit-specific session management mixin. Provides specialized
    utilities for managing Streamlit application session state, including
    CrackSeg-specific workflows and state validation.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize Streamlit session mixin."""
        super().__init__(*args, **kwargs)
        self._streamlit_manager = StreamlitSessionManager()

        if hasattr(self, "log_test_step"):
            self.log_test_step(  # type: ignore[attr-defined]
                "Streamlit session management initialized"
            )

    def get_streamlit_session_state(
        self,
        driver: WebDriver,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """
        Get Streamlit session_state. Args: driver: WebDriver instance timeout:
        Timeout for session state retrieval Returns: Dictionary containing
        session state
        """
        state = self._streamlit_manager.get_streamlit_session_state(
            driver, timeout
        )

        if hasattr(self, "log_test_step"):
            self.log_test_step(  # type: ignore[attr-defined]
                "Retrieved Streamlit session state",
                f"Found {len(state)} items",
            )

        return state

    def set_streamlit_session_value(
        self,
        driver: WebDriver,
        key: str,
        value: Any,
        wait_for_rerun: bool = True,
    ) -> bool:
        """
        Set a value in Streamlit session_state. Args: driver: WebDriver
        instance key: Session state key value: Value to set wait_for_rerun:
        Whether to wait for Streamlit rerun Returns: True if value was set
        successfully
        """
        success = self._streamlit_manager.set_streamlit_session_value(
            driver, key, value, wait_for_rerun
        )

        if hasattr(self, "log_test_step"):
            status = "succeeded" if success else "failed"
            self.log_test_step(  # type: ignore[attr-defined]
                f"Set Streamlit session value {key}", f"Operation {status}"
            )

        return success

    def setup_crackseg_test_state(
        self,
        driver: WebDriver,
        config_overrides: dict[str, Any] | None = None,
        model_settings: dict[str, Any] | None = None,
        training_settings: dict[str, Any] | None = None,
    ) -> bool:
        """
        Setup CrackSeg application test state. Args: driver: WebDriver
        instance config_overrides: Configuration values to set model_settings:
        Model settings to configure training_settings: Training settings to
        configure Returns: True if setup was successful
        """
        success = self._streamlit_manager.setup_crackseg_test_state(
            driver, config_overrides, model_settings, training_settings
        )

        if hasattr(self, "log_test_step"):
            status = "succeeded" if success else "failed"
            self.log_test_step(  # type: ignore[attr-defined]
                "Setup CrackSeg test state", f"Operation {status}"
            )

        return success

    def validate_crackseg_app_state(
        self,
        driver: WebDriver,
        expected_config: dict[str, Any] | None = None,
        expected_model_state: dict[str, Any] | None = None,
        expected_training_state: dict[str, Any] | None = None,
    ) -> bool:
        """
        Validate CrackSeg application-specific state. Args: driver: WebDriver
        instance expected_config: Expected configuration state
        expected_model_state: Expected model state expected_training_state:
        Expected training state Returns: True if validation passes
        """
        success = self._streamlit_manager.validate_crackseg_app_state(
            driver,
            expected_config,
            expected_model_state,
            expected_training_state,
        )

        if hasattr(self, "log_assertion"):
            self.log_assertion(  # type: ignore[attr-defined]
                "CrackSeg app state validation", success
            )

        return success
