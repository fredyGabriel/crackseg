"""Cookie management utilities for session state testing.

This module provides comprehensive cookie management capabilities for E2E
testing including CRUD operations, validation, and cross-browser compatibility.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from selenium.common.exceptions import WebDriverException
from selenium.webdriver.remote.webdriver import WebDriver

logger = logging.getLogger(__name__)


@dataclass
class CookieData:
    """Data structure for cookie information.

    Attributes:
        name: Cookie name
        value: Cookie value
        domain: Cookie domain
        path: Cookie path
        secure: Whether cookie requires HTTPS
        http_only: Whether cookie is HTTP only
        same_site: SameSite attribute value
        expiry: Expiration timestamp
        session: Whether cookie is session-only
    """

    name: str
    value: str
    domain: str | None = None
    path: str = "/"
    secure: bool = False
    http_only: bool = False
    same_site: str | None = None
    expiry: int | None = None
    session: bool = True


class CookieManager:
    """Cookie management for E2E testing.

    Provides comprehensive cookie operations including creation, retrieval,
    modification, deletion, and validation with cross-browser compatibility.
    """

    def __init__(self) -> None:
        """Initialize cookie manager."""
        self._cookie_backup: dict[str, list[dict[str, Any]]] = {}
        logger.debug("CookieManager initialized")

    def set_cookie(
        self,
        driver: WebDriver,
        name: str,
        value: str,
        domain: str | None = None,
        path: str = "/",
        secure: bool = False,
        http_only: bool = False,
        same_site: str | None = None,
        expiry: datetime | None = None,
        session: bool = True,
    ) -> bool:
        """Set a cookie with comprehensive options.

        Args:
            driver: WebDriver instance
            name: Cookie name
            value: Cookie value
            domain: Cookie domain (None for current domain)
            path: Cookie path
            secure: Whether cookie requires HTTPS
            http_only: Whether cookie is HTTP only
            same_site: SameSite attribute ("Strict", "Lax", "None")
            expiry: Expiration datetime
            session: Whether cookie is session-only

        Returns:
            True if cookie was set successfully

        Raises:
            WebDriverException: If cookie operation fails
        """
        try:
            cookie_dict: dict[str, Any] = {
                "name": name,
                "value": value,
                "path": path,
            }

            # Add optional attributes
            if domain:
                cookie_dict["domain"] = domain
            if secure:
                cookie_dict["secure"] = secure
            if http_only:
                cookie_dict["httpOnly"] = http_only
            if same_site:
                cookie_dict["sameSite"] = same_site
            if expiry and not session:
                cookie_dict["expiry"] = int(expiry.timestamp())

            driver.add_cookie(cookie_dict)

            logger.debug(f"Set cookie: {name}={value}")
            return True

        except WebDriverException as e:
            logger.error(f"Failed to set cookie {name}: {e}")
            return False

    def get_cookie(self, driver: WebDriver, name: str) -> CookieData | None:
        """Get a specific cookie by name.

        Args:
            driver: WebDriver instance
            name: Cookie name to retrieve

        Returns:
            CookieData object if found, None otherwise
        """
        try:
            cookie = driver.get_cookie(name)
            if not cookie:
                return None

            return CookieData(
                name=cookie["name"],
                value=cookie["value"],
                domain=cookie.get("domain"),
                path=cookie.get("path", "/"),
                secure=cookie.get("secure", False),
                http_only=cookie.get("httpOnly", False),
                same_site=cookie.get("sameSite"),
                expiry=cookie.get("expiry"),
                session=cookie.get("expiry") is None,
            )

        except WebDriverException as e:
            logger.error(f"Failed to get cookie {name}: {e}")
            return None

    def get_all_cookies(self, driver: WebDriver) -> list[CookieData]:
        """Get all cookies for the current domain.

        Args:
            driver: WebDriver instance

        Returns:
            List of CookieData objects
        """
        try:
            cookies = driver.get_cookies()
            return [
                CookieData(
                    name=cookie["name"],
                    value=cookie["value"],
                    domain=cookie.get("domain"),
                    path=cookie.get("path", "/"),
                    secure=cookie.get("secure", False),
                    http_only=cookie.get("httpOnly", False),
                    same_site=cookie.get("sameSite"),
                    expiry=cookie.get("expiry"),
                    session=cookie.get("expiry") is None,
                )
                for cookie in cookies
            ]

        except WebDriverException as e:
            logger.error(f"Failed to get cookies: {e}")
            return []

    def delete_cookie(self, driver: WebDriver, name: str) -> bool:
        """Delete a specific cookie.

        Args:
            driver: WebDriver instance
            name: Cookie name to delete

        Returns:
            True if cookie was deleted successfully
        """
        try:
            driver.delete_cookie(name)
            logger.debug(f"Deleted cookie: {name}")
            return True

        except WebDriverException as e:
            logger.error(f"Failed to delete cookie {name}: {e}")
            return False

    def delete_all_cookies(self, driver: WebDriver) -> bool:
        """Delete all cookies for the current domain.

        Args:
            driver: WebDriver instance

        Returns:
            True if all cookies were deleted successfully
        """
        try:
            driver.delete_all_cookies()
            logger.debug("Deleted all cookies")
            return True

        except WebDriverException as e:
            logger.error(f"Failed to delete all cookies: {e}")
            return False

    def cookie_exists(self, driver: WebDriver, name: str) -> bool:
        """Check if a cookie exists.

        Args:
            driver: WebDriver instance
            name: Cookie name to check

        Returns:
            True if cookie exists
        """
        return self.get_cookie(driver, name) is not None

    def backup_cookies(self, driver: WebDriver, key: str = "default") -> bool:
        """Backup current cookies for later restoration.

        Args:
            driver: WebDriver instance
            key: Backup identifier key

        Returns:
            True if backup was successful
        """
        try:
            cookies = driver.get_cookies()
            self._cookie_backup[key] = cookies
            logger.debug(f"Backed up {len(cookies)} cookies with key: {key}")
            return True

        except WebDriverException as e:
            logger.error(f"Failed to backup cookies: {e}")
            return False

    def restore_cookies(self, driver: WebDriver, key: str = "default") -> bool:
        """Restore previously backed up cookies.

        Args:
            driver: WebDriver instance
            key: Backup identifier key

        Returns:
            True if restoration was successful
        """
        if key not in self._cookie_backup:
            logger.warning(f"No cookie backup found for key: {key}")
            return False

        try:
            # Clear current cookies
            driver.delete_all_cookies()

            # Restore backed up cookies
            for cookie in self._cookie_backup[key]:
                driver.add_cookie(cookie)

            logger.debug(f"Restored {len(self._cookie_backup[key])} cookies")
            return True

        except WebDriverException as e:
            logger.error(f"Failed to restore cookies: {e}")
            return False

    def export_cookies(self, driver: WebDriver) -> str:
        """Export cookies to JSON string.

        Args:
            driver: WebDriver instance

        Returns:
            JSON string representation of cookies
        """
        try:
            cookies = driver.get_cookies()
            return json.dumps(cookies, indent=2)

        except WebDriverException as e:
            logger.error(f"Failed to export cookies: {e}")
            return "[]"

    def import_cookies(self, driver: WebDriver, cookies_json: str) -> bool:
        """Import cookies from JSON string.

        Args:
            driver: WebDriver instance
            cookies_json: JSON string of cookies

        Returns:
            True if import was successful
        """
        try:
            cookies = json.loads(cookies_json)

            # Clear current cookies
            driver.delete_all_cookies()

            # Add imported cookies
            for cookie in cookies:
                driver.add_cookie(cookie)

            logger.debug(f"Imported {len(cookies)} cookies")
            return True

        except (json.JSONDecodeError, WebDriverException) as e:
            logger.error(f"Failed to import cookies: {e}")
            return False

    def validate_cookie_persistence(
        self,
        driver: WebDriver,
        cookie_name: str,
        expected_value: str,
        navigation_url: str | None = None,
    ) -> bool:
        """Validate that a cookie persists across page navigation.

        Args:
            driver: WebDriver instance
            cookie_name: Name of cookie to validate
            expected_value: Expected cookie value
            navigation_url: URL to navigate to (refreshes current page if None)

        Returns:
            True if cookie persisted with expected value
        """
        try:
            # Navigate or refresh
            if navigation_url:
                driver.get(navigation_url)
            else:
                driver.refresh()

            # Check cookie persistence
            cookie = self.get_cookie(driver, cookie_name)

            if not cookie:
                logger.warning(
                    f"Cookie {cookie_name} not found after navigation"
                )
                return False

            if cookie.value != expected_value:
                logger.warning(
                    f"Cookie {cookie_name} value mismatch: "
                    f"expected {expected_value}, got {cookie.value}"
                )
                return False

            logger.debug(f"Cookie {cookie_name} persisted correctly")
            return True

        except WebDriverException as e:
            logger.error(f"Failed to validate cookie persistence: {e}")
            return False

    def clear_backup(self, key: str = "default") -> None:
        """Clear cookie backup.

        Args:
            key: Backup key to clear (all if "all")
        """
        if key == "all":
            self._cookie_backup.clear()
            logger.debug("Cleared all cookie backups")
        elif key in self._cookie_backup:
            del self._cookie_backup[key]
            logger.debug(f"Cleared cookie backup: {key}")
