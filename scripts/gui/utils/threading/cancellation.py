"""Cancellation mechanisms for UI responsive threading.

This module provides thread-safe cancellation capabilities for long-running
background operations, allowing UI components to cleanly interrupt tasks.
"""

import threading
from typing import Any


class CancellationToken:
    """Thread-safe cancellation token for background operations.

    Allows UI to request cancellation of long-running operations
    in a thread-safe manner. Provides mechanisms for checking cancellation
    status, waiting for cancellation, and resetting for reuse.

    Features:
    - Thread-safe cancellation requests
    - Optional cancellation reason tracking
    - Timeout support for cancellation waiting
    - Reusable token design

    Example:
        >>> token = CancellationToken()
        >>> # In background thread:
        >>> for i in range(1000):
        ...     if token.is_cancelled:
        ...         return "Cancelled"
        ...     # Do work...
        >>> # In UI thread:
        >>> token.cancel("User requested stop")
    """

    def __init__(self) -> None:
        """Initialize cancellation token."""
        self._cancelled = threading.Event()
        self._reason: str | None = None
        self._lock = threading.Lock()

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested.

        This property is thread-safe and can be checked frequently
        in background operations without performance concerns.

        Returns:
            True if cancellation has been requested
        """
        return self._cancelled.is_set()

    @property
    def cancellation_reason(self) -> str | None:
        """Get the reason for cancellation.

        Returns:
            Human-readable reason for cancellation, or None if not cancelled
        """
        with self._lock:
            return self._reason

    def cancel(self, reason: str = "User requested cancellation") -> None:
        """Request cancellation of the operation.

        This method is thread-safe and can be called from any thread.
        Once called, the cancellation cannot be undone (use reset() instead).

        Args:
            reason: Human-readable reason for cancellation
        """
        with self._lock:
            self._reason = reason
        self._cancelled.set()

    def reset(self) -> None:
        """Reset the cancellation token for reuse.

        Clears both the cancellation flag and reason, allowing the token
        to be reused for subsequent operations.
        """
        with self._lock:
            self._reason = None
        self._cancelled.clear()

    def wait_for_cancellation(self, timeout: float | None = None) -> bool:
        """Wait for cancellation to be requested.

        This method blocks the current thread until cancellation is requested
        or the timeout expires. Useful for implementing cancellation-aware
        waiting in background threads.

        Args:
            timeout: Maximum time to wait in seconds (None for infinite)

        Returns:
            True if cancellation was requested, False if timeout occurred
        """
        return self._cancelled.wait(timeout=timeout)

    def check_cancellation(self) -> None:
        """Check for cancellation and raise exception if cancelled.

        Convenience method for operations that want to use exception-based
        cancellation handling rather than checking is_cancelled.

        Raises:
            CancellationError: If cancellation has been requested
        """
        if self.is_cancelled:
            raise CancellationError(
                self.cancellation_reason or "Operation cancelled"
            )

    def __bool__(self) -> bool:
        """Support boolean evaluation of cancellation status.

        Returns:
            True if NOT cancelled (opposite of is_cancelled for intuitive use)
        """
        return not self.is_cancelled

    def __enter__(self) -> "CancellationToken":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - automatically reset token."""
        self.reset()


class CancellationError(Exception):
    """Exception raised when an operation is cancelled.

    This exception is raised by check_cancellation() when a cancellation
    token indicates that the operation should be terminated.
    """

    def __init__(self, reason: str = "Operation was cancelled") -> None:
        """Initialize cancellation error.

        Args:
            reason: Reason for cancellation
        """
        super().__init__(reason)
        self.reason = reason


class CancellationManager:
    """Manager for multiple cancellation tokens.

    Provides centralized management of cancellation tokens for operations
    that need to coordinate cancellation across multiple tasks.

    Example:
        >>> manager = CancellationManager()
        >>> token1 = manager.create_token("task1")
        >>> token2 = manager.create_token("task2")
        >>> manager.cancel_all("Shutdown requested")
    """

    def __init__(self) -> None:
        """Initialize cancellation manager."""
        self._tokens: dict[str, CancellationToken] = {}
        self._lock = threading.Lock()

    def create_token(self, name: str) -> CancellationToken:
        """Create a new named cancellation token.

        Args:
            name: Unique name for the token

        Returns:
            New cancellation token

        Raises:
            ValueError: If token name already exists
        """
        with self._lock:
            if name in self._tokens:
                raise ValueError(f"Token '{name}' already exists")

            token = CancellationToken()
            self._tokens[name] = token
            return token

    def get_token(self, name: str) -> CancellationToken | None:
        """Get existing token by name.

        Args:
            name: Token name

        Returns:
            Cancellation token or None if not found
        """
        with self._lock:
            return self._tokens.get(name)

    def cancel_token(self, name: str, reason: str = "Cancelled") -> bool:
        """Cancel a specific token by name.

        Args:
            name: Token name
            reason: Cancellation reason

        Returns:
            True if token was found and cancelled
        """
        with self._lock:
            token = self._tokens.get(name)
            if token:
                token.cancel(reason)
                return True
            return False

    def cancel_all(self, reason: str = "All operations cancelled") -> int:
        """Cancel all managed tokens.

        Args:
            reason: Cancellation reason

        Returns:
            Number of tokens cancelled
        """
        with self._lock:
            count = 0
            for token in self._tokens.values():
                if not token.is_cancelled:
                    token.cancel(reason)
                    count += 1
            return count

    def remove_token(self, name: str) -> bool:
        """Remove a token from management.

        Args:
            name: Token name

        Returns:
            True if token was found and removed
        """
        with self._lock:
            return self._tokens.pop(name, None) is not None

    def clear_all(self) -> None:
        """Remove all tokens from management."""
        with self._lock:
            self._tokens.clear()

    @property
    def active_tokens(self) -> list[str]:
        """Get list of active (non-cancelled) token names."""
        with self._lock:
            return [
                name
                for name, token in self._tokens.items()
                if not token.is_cancelled
            ]
