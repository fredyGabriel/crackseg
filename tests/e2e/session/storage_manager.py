"""Storage management utilities for session state testing.

This module provides comprehensive localStorage and sessionStorage management
capabilities for E2E testing including CRUD operations, validation, and
cross-browser compatibility.
"""

import json
import logging
from enum import Enum
from typing import Any

from selenium.common.exceptions import JavascriptException
from selenium.webdriver.remote.webdriver import WebDriver

logger = logging.getLogger(__name__)


class StorageType(Enum):
    """Storage types for web storage operations."""

    LOCAL = "localStorage"
    SESSION = "sessionStorage"


class StorageManager:
    """Storage management for localStorage and sessionStorage.

    Provides comprehensive storage operations including creation, retrieval,
    modification, deletion, and validation with cross-browser compatibility.
    """

    def __init__(self) -> None:
        """Initialize storage manager."""
        self._storage_backup: dict[str, dict[str, dict[str, Any]]] = {}
        logger.debug("StorageManager initialized")

    def set_item(
        self,
        driver: WebDriver,
        key: str,
        value: Any,
        storage_type: StorageType = StorageType.LOCAL,
    ) -> bool:
        """Set an item in the specified storage.

        Args:
            driver: WebDriver instance
            key: Storage key
            value: Value to store (will be JSON serialized)
            storage_type: Type of storage (localStorage or sessionStorage)

        Returns:
            True if item was set successfully
        """
        try:
            # Serialize value to JSON
            json_value = json.dumps(value)

            # Execute JavaScript to set storage item
            script = (
                f"{storage_type.value}.setItem(arguments[0], arguments[1]);"
            )
            driver.execute_script(script, key, json_value)

            logger.debug(f"Set {storage_type.value} item: {key}")
            return True

        except JavascriptException as e:
            logger.error(f"Failed to set {storage_type.value} item {key}: {e}")
            return False

    def get_item(
        self,
        driver: WebDriver,
        key: str,
        storage_type: StorageType = StorageType.LOCAL,
        default: Any = None,
    ) -> Any:
        """Get an item from the specified storage.

        Args:
            driver: WebDriver instance
            key: Storage key
            storage_type: Type of storage (localStorage or sessionStorage)
            default: Default value if key not found

        Returns:
            Deserialized value or default if not found
        """
        try:
            script = f"return {storage_type.value}.getItem(arguments[0]);"
            result = driver.execute_script(script, key)

            if result is None:
                return default

            # Deserialize from JSON
            return json.loads(result)

        except (JavascriptException, json.JSONDecodeError) as e:
            logger.error(f"Failed to get {storage_type.value} item {key}: {e}")
            return default

    def remove_item(
        self,
        driver: WebDriver,
        key: str,
        storage_type: StorageType = StorageType.LOCAL,
    ) -> bool:
        """Remove an item from the specified storage.

        Args:
            driver: WebDriver instance
            key: Storage key to remove
            storage_type: Type of storage (localStorage or sessionStorage)

        Returns:
            True if item was removed successfully
        """
        try:
            script = f"{storage_type.value}.removeItem(arguments[0]);"
            driver.execute_script(script, key)

            logger.debug(f"Removed {storage_type.value} item: {key}")
            return True

        except JavascriptException as e:
            logger.error(
                f"Failed to remove {storage_type.value} item {key}: {e}"
            )
            return False

    def clear_storage(
        self,
        driver: WebDriver,
        storage_type: StorageType = StorageType.LOCAL,
    ) -> bool:
        """Clear all items from the specified storage.

        Args:
            driver: WebDriver instance
            storage_type: Type of storage (localStorage or sessionStorage)

        Returns:
            True if storage was cleared successfully
        """
        try:
            script = f"{storage_type.value}.clear();"
            driver.execute_script(script)

            logger.debug(f"Cleared {storage_type.value}")
            return True

        except JavascriptException as e:
            logger.error(f"Failed to clear {storage_type.value}: {e}")
            return False

    def get_all_items(
        self,
        driver: WebDriver,
        storage_type: StorageType = StorageType.LOCAL,
    ) -> dict[str, Any]:
        """Get all items from the specified storage.

        Args:
            driver: WebDriver instance
            storage_type: Type of storage (localStorage or sessionStorage)

        Returns:
            Dictionary of all storage items
        """
        try:
            script = f"""
            var items = {{}};
            for (var i = 0; i < {storage_type.value}.length; i++) {{
                var key = {storage_type.value}.key(i);
                var value = {storage_type.value}.getItem(key);
                items[key] = value;
            }}
            return items;
            """

            raw_items = driver.execute_script(script)

            # Deserialize values from JSON
            items = {}
            for key, value in raw_items.items():
                try:
                    items[key] = json.loads(value)
                except json.JSONDecodeError:
                    # Keep as string if not valid JSON
                    items[key] = value

            logger.debug(f"Retrieved {len(items)} {storage_type.value} items")
            return items

        except JavascriptException as e:
            logger.error(f"Failed to get all {storage_type.value} items: {e}")
            return {}

    def item_exists(
        self,
        driver: WebDriver,
        key: str,
        storage_type: StorageType = StorageType.LOCAL,
    ) -> bool:
        """Check if an item exists in the specified storage.

        Args:
            driver: WebDriver instance
            key: Storage key to check
            storage_type: Type of storage (localStorage or sessionStorage)

        Returns:
            True if item exists
        """
        try:
            script = (
                f"return {storage_type.value}.getItem(arguments[0]) !== null;"
            )
            return driver.execute_script(script, key)

        except JavascriptException as e:
            logger.error(
                f"Failed to check {storage_type.value} item {key}: {e}"
            )
            return False

    def get_storage_size(
        self,
        driver: WebDriver,
        storage_type: StorageType = StorageType.LOCAL,
    ) -> int:
        """Get the number of items in the specified storage.

        Args:
            driver: WebDriver instance
            storage_type: Type of storage (localStorage or sessionStorage)

        Returns:
            Number of items in storage
        """
        try:
            script = f"return {storage_type.value}.length;"
            return driver.execute_script(script)

        except JavascriptException as e:
            logger.error(f"Failed to get {storage_type.value} size: {e}")
            return 0

    def backup_storage(
        self,
        driver: WebDriver,
        key: str = "default",
        storage_type: StorageType | None = None,
    ) -> bool:
        """Backup storage for later restoration.

        Args:
            driver: WebDriver instance
            key: Backup identifier key
            storage_type: Type of storage (None for both)

        Returns:
            True if backup was successful
        """
        try:
            backup_data = {}

            storage_types = (
                [storage_type] if storage_type else list(StorageType)
            )

            for stype in storage_types:
                backup_data[stype.value] = self.get_all_items(driver, stype)

            self._storage_backup[key] = backup_data

            total_items = sum(len(data) for data in backup_data.values())
            logger.debug(
                f"Backed up {total_items} storage items with key: {key}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to backup storage: {e}")
            return False

    def restore_storage(
        self,
        driver: WebDriver,
        key: str = "default",
        storage_type: StorageType | None = None,
    ) -> bool:
        """Restore previously backed up storage.

        Args:
            driver: WebDriver instance
            key: Backup identifier key
            storage_type: Type of storage (None for both)

        Returns:
            True if restoration was successful
        """
        if key not in self._storage_backup:
            logger.warning(f"No storage backup found for key: {key}")
            return False

        try:
            backup_data = self._storage_backup[key]
            storage_types = (
                [storage_type] if storage_type else list(StorageType)
            )

            for stype in storage_types:
                if stype.value not in backup_data:
                    continue

                # Clear current storage
                self.clear_storage(driver, stype)

                # Restore backed up items
                for item_key, item_value in backup_data[stype.value].items():
                    self.set_item(driver, item_key, item_value, stype)

            total_items = sum(
                len(backup_data.get(stype.value, {}))
                for stype in storage_types
            )
            logger.debug(f"Restored {total_items} storage items")
            return True

        except Exception as e:
            logger.error(f"Failed to restore storage: {e}")
            return False

    def validate_storage_persistence(
        self,
        driver: WebDriver,
        key: str,
        expected_value: Any,
        storage_type: StorageType = StorageType.LOCAL,
        navigation_url: str | None = None,
    ) -> bool:
        """Validate that storage persists across page navigation.

        Args:
            driver: WebDriver instance
            key: Storage key to validate
            expected_value: Expected storage value
            storage_type: Type of storage (localStorage or sessionStorage)
            navigation_url: URL to navigate to (refreshes current page if None)

        Returns:
            True if storage persisted with expected value
        """
        try:
            # Navigate or refresh
            if navigation_url:
                driver.get(navigation_url)
            else:
                driver.refresh()

            # Check storage persistence
            actual_value = self.get_item(driver, key, storage_type)

            if actual_value != expected_value:
                logger.warning(
                    f"{storage_type.value} item {key} value mismatch: "
                    f"expected {expected_value}, got {actual_value}"
                )
                return False

            logger.debug(
                f"{storage_type.value} item {key} persisted correctly"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to validate storage persistence: {e}")
            return False

    def compare_storage(
        self,
        driver: WebDriver,
        expected_items: dict[str, Any],
        storage_type: StorageType = StorageType.LOCAL,
    ) -> dict[str, Any]:
        """Compare current storage with expected items.

        Args:
            driver: WebDriver instance
            expected_items: Expected storage items
            storage_type: Type of storage (localStorage or sessionStorage)

        Returns:
            Dictionary with comparison results
        """
        try:
            actual_items = self.get_all_items(driver, storage_type)

            comparison: dict[str, Any] = {
                "matches": True,
                "missing_keys": [],
                "extra_keys": [],
                "value_mismatches": {},
            }

            # Check for missing and mismatched keys
            for key, expected_value in expected_items.items():
                if key not in actual_items:
                    comparison["missing_keys"].append(key)
                    comparison["matches"] = False
                elif actual_items[key] != expected_value:
                    comparison["value_mismatches"][key] = {
                        "expected": expected_value,
                        "actual": actual_items[key],
                    }
                    comparison["matches"] = False

            # Check for extra keys
            for key in actual_items:
                if key not in expected_items:
                    comparison["extra_keys"].append(key)
                    comparison["matches"] = False

            return comparison

        except Exception as e:
            logger.error(f"Failed to compare storage: {e}")
            return {"matches": False, "error": str(e)}

    def clear_backup(self, key: str = "default") -> None:
        """Clear storage backup.

        Args:
            key: Backup key to clear (all if "all")
        """
        if key == "all":
            self._storage_backup.clear()
            logger.debug("Cleared all storage backups")
        elif key in self._storage_backup:
            del self._storage_backup[key]
            logger.debug(f"Cleared storage backup: {key}")
