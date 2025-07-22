"""
Streamlit-specific session state management utilities. This module
provides specialized session state management for Streamlit
applications, including integration with Streamlit's session_state
mechanism, app-specific state validation, and CrackSeg application
utilities.
"""

import json
import logging
import time
from typing import Any

from selenium.common.exceptions import (
    JavascriptException,
    TimeoutException,
)
from selenium.webdriver.remote.webdriver import WebDriver

from ..utils.streamlit import (
    get_streamlit_session_state,
    wait_for_streamlit_ready,
    wait_for_streamlit_rerun,
)
from .state_manager import StateManager

logger = logging.getLogger(__name__)


class StreamlitSessionManager:
    """
    Streamlit-specific session state management. Provides specialized
    utilities for managing Streamlit application session state including
    integration with st.session_state, page-specific state validation, and
    CrackSeg application workflows.
    """

    def __init__(self) -> None:
        """Initialize Streamlit session manager."""
        self.state_manager = StateManager()
        self._app_state_cache: dict[str, dict[str, Any]] = {}
        logger.debug("StreamlitSessionManager initialized")

    def get_streamlit_session_state(
        self,
        driver: WebDriver,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """
        Get Streamlit session_state using JavaScript. Args: driver: WebDriver
        instance timeout: Timeout for session state retrieval Returns:
        Dictionary containing session state
        """
        try:
            # Wait for Streamlit to be ready
            wait_for_streamlit_ready(driver, timeout=int(timeout))

            # Use existing utility function
            return get_streamlit_session_state(driver, timeout=int(timeout))

        except (TimeoutException, JavascriptException) as e:
            logger.error(f"Failed to get Streamlit session state: {e}")
            return {}

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
        try:
            # Serialize value to JSON for JavaScript
            json_value = json.dumps(value)

            # Set value using JavaScript
            script = """
if (window.parent && window.parent.sessionStorage) { var
streamlitSessionState = JSON.parse(
window.parent.sessionStorage.getItem( 'streamlit_session_state' ) ||
'{}' ); streamlitSessionState[arguments[0]] =
JSON.parse(arguments[1]); window.parent.sessionStorage.setItem(
'streamlit_session_state', JSON.stringify(streamlitSessionState) );
return true; } return false;
"""

            result = driver.execute_script(script, key, json_value)

            if result and wait_for_rerun:
                wait_for_streamlit_rerun(driver)

            logger.debug(f"Set Streamlit session state: {key}")
            return bool(result)

        except JavascriptException as e:
            logger.error(f"Failed to set Streamlit session state {key}: {e}")
            return False

    def remove_streamlit_session_value(
        self,
        driver: WebDriver,
        key: str,
        wait_for_rerun: bool = True,
    ) -> bool:
        """
        Remove a value from Streamlit session_state. Args: driver: WebDriver
        instance key: Session state key to remove wait_for_rerun: Whether to
        wait for Streamlit rerun Returns: True if value was removed
        successfully
        """
        try:
            script = """
if (window.parent && window.parent.sessionStorage) { var
streamlitSessionState = JSON.parse(
window.parent.sessionStorage.getItem( 'streamlit_session_state' ) ||
'{}' ); if (arguments[0] in streamlitSessionState) { delete
streamlitSessionState[arguments[0]];
window.parent.sessionStorage.setItem( 'streamlit_session_state',
JSON.stringify(streamlitSessionState) ); return true; } } return
false;
"""

            result = driver.execute_script(script, key)

            if result and wait_for_rerun:
                wait_for_streamlit_rerun(driver)

            logger.debug(f"Removed Streamlit session state: {key}")
            return bool(result)

        except JavascriptException as e:
            logger.error(
                f"Failed to remove Streamlit session state {key}: {e}"
            )
            return False

    def clear_streamlit_session_state(
        self,
        driver: WebDriver,
        wait_for_rerun: bool = True,
    ) -> bool:
        """
        Clear all Streamlit session_state. Args: driver: WebDriver instance
        wait_for_rerun: Whether to wait for Streamlit rerun Returns: True if
        session state was cleared successfully
        """
        try:
            script = """
if (window.parent && window.parent.sessionStorage) {
window.parent.sessionStorage.removeItem('streamlit_session_state');
return true; } return false;
"""

            result = driver.execute_script(script)

            if result and wait_for_rerun:
                wait_for_streamlit_rerun(driver)

            logger.debug("Cleared Streamlit session state")
            return bool(result)

        except JavascriptException as e:
            logger.error(f"Failed to clear Streamlit session state: {e}")
            return False

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
        try:
            session_state = self.get_streamlit_session_state(driver)
            validation_errors = []

            # Validate configuration state
            if expected_config:
                for key, expected_value in expected_config.items():
                    config_key = f"config_{key}"
                    if config_key not in session_state:
                        validation_errors.append(f"Missing config key: {key}")
                    elif session_state[config_key] != expected_value:
                        validation_errors.append(
                            f"Config {key} mismatch: "
                            f"expected {expected_value}, "
                            f"got {session_state[config_key]}"
                        )

            # Validate model state
            if expected_model_state:
                for key, expected_value in expected_model_state.items():
                    model_key = f"model_{key}"
                    if model_key not in session_state:
                        validation_errors.append(f"Missing model key: {key}")
                    elif session_state[model_key] != expected_value:
                        validation_errors.append(
                            f"Model {key} mismatch: "
                            f"expected {expected_value}, "
                            f"got {session_state[model_key]}"
                        )

            # Validate training state
            if expected_training_state:
                for key, expected_value in expected_training_state.items():
                    training_key = f"training_{key}"
                    if training_key not in session_state:
                        validation_errors.append(
                            f"Missing training key: {key}"
                        )
                    elif session_state[training_key] != expected_value:
                        validation_errors.append(
                            f"Training {key} mismatch: "
                            f"expected {expected_value}, "
                            f"got {session_state[training_key]}"
                        )

            if validation_errors:
                logger.warning(
                    f"CrackSeg app state validation failed: "
                    f"{validation_errors}"
                )
                return False

            logger.debug("CrackSeg app state validation passed")
            return True

        except Exception as e:
            logger.error(f"Failed to validate CrackSeg app state: {e}")
            return False

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
        try:
            # Set configuration overrides
            if config_overrides:
                for key, value in config_overrides.items():
                    self.set_streamlit_session_value(
                        driver, f"config_{key}", value, wait_for_rerun=False
                    )

            # Set model settings
            if model_settings:
                for key, value in model_settings.items():
                    self.set_streamlit_session_value(
                        driver, f"model_{key}", value, wait_for_rerun=False
                    )

            # Set training settings
            if training_settings:
                for key, value in training_settings.items():
                    self.set_streamlit_session_value(
                        driver, f"training_{key}", value, wait_for_rerun=False
                    )

            # Trigger Streamlit rerun to apply all changes
            wait_for_streamlit_rerun(driver)

            logger.debug("CrackSeg test state setup completed")
            return True

        except Exception as e:
            logger.error(f"Failed to setup CrackSeg test state: {e}")
            return False

    def capture_crackseg_app_snapshot(
        self,
        driver: WebDriver,
        include_config: bool = True,
        include_model: bool = True,
        include_training: bool = True,
    ) -> dict[str, Any]:
        """
        Capture CrackSeg application state snapshot. Args: driver: WebDriver
        instance include_config: Whether to include configuration state
        include_model: Whether to include model state include_training:
        Whether to include training state Returns: Dictionary containing
        application state snapshot
        """
        try:
            session_state = self.get_streamlit_session_state(driver)
            snapshot: dict[str, Any] = {
                "timestamp": time.time(),
                "url": driver.current_url,
                "title": driver.title,
            }

            # Capture configuration state
            if include_config:
                config_state = {}
                for key, value in session_state.items():
                    if key.startswith("config_"):
                        config_key = key[7:]  # Remove "config_" prefix
                        config_state[config_key] = value
                snapshot["config"] = config_state

            # Capture model state
            if include_model:
                model_state = {}
                for key, value in session_state.items():
                    if key.startswith("model_"):
                        model_key = key[6:]  # Remove "model_" prefix
                        model_state[model_key] = value
                snapshot["model"] = model_state

            # Capture training state
            if include_training:
                training_state = {}
                for key, value in session_state.items():
                    if key.startswith("training_"):
                        training_key = key[9:]  # Remove "training_" prefix
                        training_state[training_key] = value
                snapshot["training"] = training_state

            # Include raw session state
            snapshot["raw_session_state"] = session_state

            logger.debug("CrackSeg app snapshot captured")
            return snapshot

        except Exception as e:
            logger.error(f"Failed to capture CrackSeg app snapshot: {e}")
            return {}

    def wait_for_streamlit_state_change(
        self,
        driver: WebDriver,
        key: str,
        expected_value: Any | None = None,
        timeout: float = 30.0,
        poll_interval: float = 1.0,
    ) -> bool:
        """
        Wait for a Streamlit session state change. Args: driver: WebDriver
        instance key: Session state key to monitor expected_value: Expected
        value (any change if None) timeout: Maximum time to wait
        poll_interval: How often to check state Returns: True if state change
        detected
        """
        try:
            start_time = time.time()
            initial_state = self.get_streamlit_session_state(driver)
            initial_value = initial_state.get(key)

            while time.time() - start_time < timeout:
                current_state = self.get_streamlit_session_state(driver)
                current_value = current_state.get(key)

                # Check for expected change
                if expected_value is not None:
                    if current_value == expected_value:
                        logger.debug(
                            f"Streamlit state {key} changed to expected value"
                        )
                        return True
                else:
                    if current_value != initial_value:
                        logger.debug(f"Streamlit state {key} changed")
                        return True

                time.sleep(poll_interval)

            logger.warning(
                f"Timeout waiting for Streamlit state change: {key}"
            )
            return False

        except Exception as e:
            logger.error(f"Failed to wait for Streamlit state change: {e}")
            return False

    def reset_crackseg_app_state(
        self,
        driver: WebDriver,
        preserve_config: bool = False,
        preserve_model: bool = False,
    ) -> bool:
        """
        Reset CrackSeg application state to defaults. Args: driver: WebDriver
        instance preserve_config: Whether to preserve configuration state
        preserve_model: Whether to preserve model state Returns: True if reset
        was successful
        """
        try:
            session_state = self.get_streamlit_session_state(driver)

            # Identify keys to clear
            keys_to_clear = []
            for key in session_state.keys():
                if key.startswith("training_"):
                    keys_to_clear.append(key)
                elif key.startswith("config_") and not preserve_config:
                    keys_to_clear.append(key)
                elif key.startswith("model_") and not preserve_model:
                    keys_to_clear.append(key)

            # Clear identified keys
            for key in keys_to_clear:
                self.remove_streamlit_session_value(
                    driver, key, wait_for_rerun=False
                )

            # Trigger rerun to apply changes
            if keys_to_clear:
                wait_for_streamlit_rerun(driver)

            logger.debug(
                f"Reset CrackSeg app state, cleared {len(keys_to_clear)} keys"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to reset CrackSeg app state: {e}")
            return False

    def cache_app_state(self, snapshot: dict[str, Any], key: str) -> None:
        """
        Cache an application state snapshot. Args: snapshot: State snapshot to
        cache key: Cache key
        """
        self._app_state_cache[key] = snapshot
        logger.debug(f"Cached app state with key: {key}")

    def get_cached_app_state(self, key: str) -> dict[str, Any] | None:
        """
        Get a cached application state snapshot. Args: key: Cache key Returns:
        Cached snapshot if found, None otherwise
        """
        snapshot = self._app_state_cache.get(key)
        if snapshot:
            logger.debug(f"Retrieved cached app state: {key}")
        else:
            logger.warning(f"No cached app state found: {key}")
        return snapshot

    def clear_app_state_cache(self, key: str | None = None) -> None:
        """
        Clear cached application state. Args: key: Specific key to clear (all
        if None)
        """
        if key is None:
            self._app_state_cache.clear()
            logger.debug("Cleared all cached app states")
        elif key in self._app_state_cache:
            del self._app_state_cache[key]
            logger.debug(f"Cleared cached app state: {key}")
