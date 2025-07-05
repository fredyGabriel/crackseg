"""Session state management for E2E testing.

This module provides comprehensive session state management capabilities
including cookie handling, local storage management, session persistence,
and Streamlit-specific session state utilities.

Key components:
- CookieManager: Cookie CRUD operations and validation
- StorageManager: Local/session storage management
- StateManager: Session state validation and persistence
- StreamlitSessionManager: Streamlit-specific session state integration
- SessionManagementMixin: Main mixin for BaseE2ETest integration
- MultiTabSessionMixin: Multi-tab session coordination
- StreamlitSessionMixin: Streamlit-specific session handling

Examples:
    Basic usage with mixin pattern:
    >>> class MyTest(BaseE2ETest, SessionManagementMixin):
    ...     def test_session_persistence(self, driver):
    ...         self.set_cookie(driver, "user_id", "123")
    ...         self.navigate_and_verify(driver, "/login")
    ...         assert self.get_cookie(driver, "user_id") == "123"

    Multi-tab session testing:
    >>> class MultiTabTest(BaseE2ETest, MultiTabSessionMixin):
    ...     def test_cross_tab_state(self, driver):
    ...         tab1 = self.open_new_tab(driver, "/app")
    ...         self.set_session_state(driver, "key", "value")
    ...         tab2 = self.open_new_tab(driver, "/app")
    ...         assert self.get_session_state(tab2, "key") == "value"

    Streamlit session state:
    >>> class StreamlitTest(BaseE2ETest, StreamlitSessionMixin):
    ...     def test_streamlit_state(self, driver):
    ...         self.set_streamlit_session_value(
    ...             driver, "model_config", config
    ...         )
    ...         self.navigate_to_page(driver, "Training")
    ...         state = self.get_streamlit_session_state(driver)
    ...         assert state["model_config"] == config
"""

from .cookie_manager import CookieManager

# Mixin imports for integration with BaseE2ETest
from .mixins import (
    MultiTabSessionMixin,
    SessionManagementMixin,
    StreamlitSessionMixin,
)

# Protocol imports
from .state_manager import HasSessionState, StateManager, StateValidationError
from .storage_manager import StorageManager, StorageType
from .streamlit_session import StreamlitSessionManager

__all__ = [
    # Core managers
    "CookieManager",
    "StateManager",
    "StorageManager",
    "StreamlitSessionManager",
    # Protocols
    "HasSessionState",
    # Mixins for BaseE2ETest integration
    "SessionManagementMixin",
    "MultiTabSessionMixin",
    "StreamlitSessionMixin",
    # Enums and exceptions
    "StorageType",
    "StateValidationError",
]
