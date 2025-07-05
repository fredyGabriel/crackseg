"""Session state validation and persistence management.

This module provides comprehensive session state management capabilities
including state validation, comparison, persistence testing, and restoration
mechanisms for E2E testing scenarios.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from selenium.common.exceptions import WebDriverException
from selenium.webdriver.remote.webdriver import WebDriver

from .cookie_manager import CookieManager
from .storage_manager import StorageManager, StorageType

logger = logging.getLogger(__name__)


class StateValidationError(Exception):
    """Exception raised when session state validation fails."""

    def __init__(
        self, message: str, details: dict[str, Any] | None = None
    ) -> None:
        """Initialize state validation error.

        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message)
        self.details = details or {}


@dataclass
class SessionSnapshot:
    """Complete session state snapshot.

    Attributes:
        timestamp: When snapshot was taken
        cookies: All cookies at snapshot time
        local_storage: All localStorage items
        session_storage: All sessionStorage items
        url: Current page URL
        title: Current page title
        metadata: Additional snapshot metadata
    """

    timestamp: float
    cookies: dict[str, Any]
    local_storage: dict[str, Any]
    session_storage: dict[str, Any]
    url: str
    title: str
    metadata: dict[str, Any] = field(default_factory=dict)


class HasSessionState(Protocol):
    """Protocol for classes that support session state management."""

    def capture_session_snapshot(self, driver: WebDriver) -> SessionSnapshot:
        """Capture complete session state snapshot."""
        ...

    def validate_session_state(
        self,
        driver: WebDriver,
        expected_state: dict[str, Any],
        strict: bool = True,
    ) -> bool:
        """Validate current session state against expected values."""
        ...

    def restore_session_state(
        self,
        driver: WebDriver,
        snapshot: SessionSnapshot,
    ) -> bool:
        """Restore session state from snapshot."""
        ...


class StateManager:
    """Session state validation and persistence management.

    Provides comprehensive session state operations including validation,
    comparison, persistence testing, and restoration capabilities.
    """

    def __init__(self) -> None:
        """Initialize state manager."""
        self.cookie_manager = CookieManager()
        self.storage_manager = StorageManager()
        self._snapshots: dict[str, SessionSnapshot] = {}
        logger.debug("StateManager initialized")

    def capture_session_snapshot(
        self,
        driver: WebDriver,
        include_metadata: bool = True,
    ) -> SessionSnapshot:
        """Capture complete session state snapshot.

        Args:
            driver: WebDriver instance
            include_metadata: Whether to include additional metadata

        Returns:
            SessionSnapshot containing complete state
        """
        try:
            # Capture cookies
            cookies = {}
            for cookie in self.cookie_manager.get_all_cookies(driver):
                cookies[cookie.name] = {
                    "value": cookie.value,
                    "domain": cookie.domain,
                    "path": cookie.path,
                    "secure": cookie.secure,
                    "http_only": cookie.http_only,
                    "same_site": cookie.same_site,
                    "expiry": cookie.expiry,
                    "session": cookie.session,
                }

            # Capture storage
            local_storage = self.storage_manager.get_all_items(
                driver, StorageType.LOCAL
            )
            session_storage = self.storage_manager.get_all_items(
                driver, StorageType.SESSION
            )

            # Capture page state
            url = driver.current_url
            title = driver.title

            # Additional metadata
            metadata = {}
            if include_metadata:
                try:
                    metadata.update(
                        {
                            "window_size": driver.get_window_size(),
                            "page_source_hash": hash(driver.page_source),
                            "cookies_count": len(cookies),
                            "local_storage_count": len(local_storage),
                            "session_storage_count": len(session_storage),
                        }
                    )
                except WebDriverException:
                    logger.warning("Could not capture complete metadata")

            snapshot = SessionSnapshot(
                timestamp=time.time(),
                cookies=cookies,
                local_storage=local_storage,
                session_storage=session_storage,
                url=url,
                title=title,
                metadata=metadata,
            )

            logger.debug(
                f"Captured session snapshot with {len(cookies)} cookies"
            )
            return snapshot

        except WebDriverException as e:
            logger.error(f"Failed to capture session snapshot: {e}")
            raise StateValidationError(
                "Failed to capture session snapshot", {"error": str(e)}
            ) from e

    def validate_session_state(
        self,
        driver: WebDriver,
        expected_state: dict[str, Any],
        strict: bool = True,
        timeout: float = 5.0,
    ) -> bool:
        """Validate current session state against expected values.

        Args:
            driver: WebDriver instance
            expected_state: Expected state values
            strict: Whether to require exact matches
            timeout: Timeout for state stabilization

        Returns:
            True if validation passes

        Raises:
            StateValidationError: If validation fails
        """
        try:
            # Wait for state to stabilize
            time.sleep(timeout)

            current_snapshot = self.capture_session_snapshot(driver)
            validation_errors = []

            # Validate cookies
            if "cookies" in expected_state:
                for cookie_name, expected_value in expected_state[
                    "cookies"
                ].items():
                    if cookie_name not in current_snapshot.cookies:
                        validation_errors.append(
                            f"Missing cookie: {cookie_name}"
                        )
                    elif (
                        current_snapshot.cookies[cookie_name]["value"]
                        != expected_value
                    ):
                        validation_errors.append(
                            f"Cookie {cookie_name} value mismatch: "
                            f"expected {expected_value}, "
                            f"got "
                            f"{current_snapshot.cookies[cookie_name]['value']}"
                        )

            # Validate localStorage
            if "local_storage" in expected_state:
                for key, expected_value in expected_state[
                    "local_storage"
                ].items():
                    if key not in current_snapshot.local_storage:
                        validation_errors.append(
                            f"Missing localStorage key: {key}"
                        )
                    elif current_snapshot.local_storage[key] != expected_value:
                        validation_errors.append(
                            f"localStorage {key} value mismatch: "
                            f"expected {expected_value}, "
                            f"got {current_snapshot.local_storage[key]}"
                        )

            # Validate sessionStorage
            if "session_storage" in expected_state:
                for key, expected_value in expected_state[
                    "session_storage"
                ].items():
                    if key not in current_snapshot.session_storage:
                        validation_errors.append(
                            f"Missing sessionStorage key: {key}"
                        )
                    elif (
                        current_snapshot.session_storage[key] != expected_value
                    ):
                        validation_errors.append(
                            f"sessionStorage {key} value mismatch: "
                            f"expected {expected_value}, "
                            f"got {current_snapshot.session_storage[key]}"
                        )

            # Validate URL
            if "url" in expected_state:
                expected_url = expected_state["url"]
                if current_snapshot.url != expected_url:
                    if strict or not current_snapshot.url.endswith(
                        expected_url
                    ):
                        validation_errors.append(
                            f"URL mismatch: expected {expected_url}, "
                            f"got {current_snapshot.url}"
                        )

            # Check for validation errors
            if validation_errors:
                error_message = "Session state validation failed"
                details = {
                    "errors": validation_errors,
                    "current_state": {
                        "cookies": current_snapshot.cookies,
                        "local_storage": current_snapshot.local_storage,
                        "session_storage": current_snapshot.session_storage,
                        "url": current_snapshot.url,
                    },
                    "expected_state": expected_state,
                }

                if strict:
                    raise StateValidationError(error_message, details)
                else:
                    logger.warning(f"{error_message}: {validation_errors}")
                    return False

            logger.debug("Session state validation passed")
            return True

        except WebDriverException as e:
            logger.error(f"Failed to validate session state: {e}")
            raise StateValidationError(
                "Failed to validate session state", {"error": str(e)}
            ) from e

    def restore_session_state(
        self,
        driver: WebDriver,
        snapshot: SessionSnapshot,
        navigate_to_url: bool = True,
    ) -> bool:
        """Restore session state from snapshot.

        Args:
            driver: WebDriver instance
            snapshot: Session snapshot to restore
            navigate_to_url: Whether to navigate to snapshot URL

        Returns:
            True if restoration was successful
        """
        try:
            # Navigate to URL if requested
            if navigate_to_url and driver.current_url != snapshot.url:
                driver.get(snapshot.url)
                time.sleep(1)  # Allow page to load

            # Restore cookies
            self.cookie_manager.delete_all_cookies(driver)
            for cookie_name, cookie_data in snapshot.cookies.items():
                self.cookie_manager.set_cookie(
                    driver,
                    cookie_name,
                    cookie_data["value"],
                    domain=cookie_data.get("domain"),
                    path=cookie_data.get("path", "/"),
                    secure=cookie_data.get("secure", False),
                    http_only=cookie_data.get("http_only", False),
                    same_site=cookie_data.get("same_site"),
                    session=cookie_data.get("session", True),
                )

            # Restore localStorage
            self.storage_manager.clear_storage(driver, StorageType.LOCAL)
            for key, value in snapshot.local_storage.items():
                self.storage_manager.set_item(
                    driver, key, value, StorageType.LOCAL
                )

            # Restore sessionStorage
            self.storage_manager.clear_storage(driver, StorageType.SESSION)
            for key, value in snapshot.session_storage.items():
                self.storage_manager.set_item(
                    driver, key, value, StorageType.SESSION
                )

            logger.debug("Session state restored successfully")
            return True

        except WebDriverException as e:
            logger.error(f"Failed to restore session state: {e}")
            return False

    def compare_snapshots(
        self,
        snapshot1: SessionSnapshot,
        snapshot2: SessionSnapshot,
        ignore_timestamp: bool = True,
    ) -> dict[str, Any]:
        """Compare two session snapshots.

        Args:
            snapshot1: First snapshot
            snapshot2: Second snapshot
            ignore_timestamp: Whether to ignore timestamp differences

        Returns:
            Dictionary with comparison results
        """
        comparison: dict[str, Any] = {
            "identical": True,
            "differences": {},
        }

        # Compare cookies
        if snapshot1.cookies != snapshot2.cookies:
            comparison["identical"] = False
            comparison["differences"]["cookies"] = {
                "snapshot1": snapshot1.cookies,
                "snapshot2": snapshot2.cookies,
            }

        # Compare localStorage
        if snapshot1.local_storage != snapshot2.local_storage:
            comparison["identical"] = False
            comparison["differences"]["local_storage"] = {
                "snapshot1": snapshot1.local_storage,
                "snapshot2": snapshot2.local_storage,
            }

        # Compare sessionStorage
        if snapshot1.session_storage != snapshot2.session_storage:
            comparison["identical"] = False
            comparison["differences"]["session_storage"] = {
                "snapshot1": snapshot1.session_storage,
                "snapshot2": snapshot2.session_storage,
            }

        # Compare URL
        if snapshot1.url != snapshot2.url:
            comparison["identical"] = False
            comparison["differences"]["url"] = {
                "snapshot1": snapshot1.url,
                "snapshot2": snapshot2.url,
            }

        # Compare title
        if snapshot1.title != snapshot2.title:
            comparison["identical"] = False
            comparison["differences"]["title"] = {
                "snapshot1": snapshot1.title,
                "snapshot2": snapshot2.title,
            }

        # Compare timestamp (if not ignored)
        if not ignore_timestamp and snapshot1.timestamp != snapshot2.timestamp:
            comparison["differences"]["timestamp"] = {
                "snapshot1": snapshot1.timestamp,
                "snapshot2": snapshot2.timestamp,
            }

        return comparison

    def save_snapshot(self, snapshot: SessionSnapshot, key: str) -> None:
        """Save a snapshot for later use.

        Args:
            snapshot: Snapshot to save
            key: Key to save snapshot under
        """
        self._snapshots[key] = snapshot
        logger.debug(f"Saved snapshot with key: {key}")

    def load_snapshot(self, key: str) -> SessionSnapshot | None:
        """Load a previously saved snapshot.

        Args:
            key: Key of snapshot to load

        Returns:
            Snapshot if found, None otherwise
        """
        snapshot = self._snapshots.get(key)
        if snapshot:
            logger.debug(f"Loaded snapshot with key: {key}")
        else:
            logger.warning(f"No snapshot found with key: {key}")
        return snapshot

    def clear_snapshots(self, key: str | None = None) -> None:
        """Clear saved snapshots.

        Args:
            key: Specific key to clear (all if None)
        """
        if key is None:
            self._snapshots.clear()
            logger.debug("Cleared all snapshots")
        elif key in self._snapshots:
            del self._snapshots[key]
            logger.debug(f"Cleared snapshot: {key}")

    def test_session_persistence(
        self,
        driver: WebDriver,
        navigation_url: str | None = None,
        refresh_page: bool = False,
    ) -> dict[str, Any]:
        """Test session state persistence across navigation.

        Args:
            driver: WebDriver instance
            navigation_url: URL to navigate to
            refresh_page: Whether to refresh current page

        Returns:
            Dictionary with persistence test results
        """
        try:
            # Capture initial state
            initial_snapshot = self.capture_session_snapshot(driver)

            # Perform navigation
            if navigation_url:
                driver.get(navigation_url)
            elif refresh_page:
                driver.refresh()

            # Wait for navigation to complete
            time.sleep(2)

            # Capture final state
            final_snapshot = self.capture_session_snapshot(driver)

            # Compare states
            comparison = self.compare_snapshots(
                initial_snapshot, final_snapshot
            )

            # Build test results
            results = {
                "persistence_test_passed": True,
                "cookies_persisted": initial_snapshot.cookies
                == final_snapshot.cookies,
                "local_storage_persisted": (
                    initial_snapshot.local_storage
                    == final_snapshot.local_storage
                ),
                "session_storage_persisted": (
                    initial_snapshot.session_storage
                    == final_snapshot.session_storage
                ),
                "comparison": comparison,
            }

            # Check overall persistence
            if not comparison["identical"]:
                results["persistence_test_passed"] = False
                logger.warning("Session persistence test failed")
            else:
                logger.debug("Session persistence test passed")

            return results

        except Exception as e:
            logger.error(f"Session persistence test failed: {e}")
            return {
                "persistence_test_passed": False,
                "error": str(e),
            }
