"""Event management system for reactive results gallery updates.

This module implements a pub-sub event system that allows the
AsyncResultsScanner to notify UI components and other listeners about
scan progress and triplet discoveries in real-time.

Phase 2: Observer pattern + LRU cache for reactive gallery updates.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any
from weakref import WeakSet

from .core import ResultTriplet, ScanProgress

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events emitted by the results scanner."""

    SCAN_STARTED = auto()
    SCAN_PROGRESS = auto()
    TRIPLET_FOUND = auto()
    SCAN_COMPLETED = auto()
    SCAN_ERROR = auto()
    SCAN_CANCELLED = auto()


@dataclass
class ScanEvent:
    """Event data structure for scanner notifications."""

    event_type: EventType
    timestamp: float
    data: dict[str, Any]

    @classmethod
    def scan_started(cls, base_path: str, pattern: str) -> ScanEvent:
        """Create a scan started event."""
        import time

        return cls(
            event_type=EventType.SCAN_STARTED,
            timestamp=time.time(),
            data={"base_path": base_path, "pattern": pattern},
        )

    @classmethod
    def scan_progress(cls, progress: ScanProgress) -> ScanEvent:
        """Create a scan progress event."""
        import time

        return cls(
            event_type=EventType.SCAN_PROGRESS,
            timestamp=time.time(),
            data={
                "progress": progress,
                "percent": progress.progress_percent,
                "found_triplets": progress.found_triplets,
                "errors": progress.errors,
            },
        )

    @classmethod
    def triplet_found(cls, triplet: ResultTriplet) -> ScanEvent:
        """Create a triplet found event."""
        import time

        return cls(
            event_type=EventType.TRIPLET_FOUND,
            timestamp=time.time(),
            data={
                "triplet": triplet,
                "triplet_id": triplet.id,
                "dataset": triplet.dataset_name,
                "is_complete": triplet.is_complete,
            },
        )

    @classmethod
    def scan_completed(
        cls, total_triplets: int, total_errors: int
    ) -> ScanEvent:
        """Create a scan completed event."""
        import time

        return cls(
            event_type=EventType.SCAN_COMPLETED,
            timestamp=time.time(),
            data={
                "total_triplets": total_triplets,
                "total_errors": total_errors,
                "success": total_errors == 0,
            },
        )

    @classmethod
    def scan_error(cls, error: Exception, context: str = "") -> ScanEvent:
        """Create a scan error event."""
        import time

        return cls(
            event_type=EventType.SCAN_ERROR,
            timestamp=time.time(),
            data={
                "error": str(error),
                "error_type": type(error).__name__,
                "context": context,
            },
        )

    @classmethod
    def scan_cancelled(cls, reason: str = "User requested") -> ScanEvent:
        """Create a scan cancelled event."""
        import time

        return cls(
            event_type=EventType.SCAN_CANCELLED,
            timestamp=time.time(),
            data={"reason": reason},
        )


# Type aliases for event handlers
SyncEventHandler = Callable[[ScanEvent], None]
AsyncEventHandler = Callable[[ScanEvent], Awaitable[None]]
EventHandler = SyncEventHandler | AsyncEventHandler


class EventManager:
    """Pub-sub event manager for results scanner notifications.

    Supports both synchronous and asynchronous event handlers with
    automatic cleanup of weak references to prevent memory leaks.

    Example:
        >>> event_manager = EventManager()
        >>>
        >>> async def on_triplet_found(event: ScanEvent) -> None:
        ...     triplet = event.data["triplet"]
        ...     print(f"New triplet: {triplet.id}")
        >>>
        >>> event_manager.subscribe(EventType.TRIPLET_FOUND, on_triplet_found)
        >>> await event_manager.emit_async(ScanEvent.triplet_found(triplet))
    """

    def __init__(self) -> None:
        """Initialize the event manager."""
        self._handlers: dict[EventType, WeakSet[EventHandler]] = {}
        self._emit_lock = asyncio.Lock()

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Subscribe a handler to an event type.

        Args:
            event_type: Type of event to listen for
            handler: Sync or async function to handle the event

        Example:
            >>> def handle_progress(event: ScanEvent) -> None:
            ...     progress = event.data["progress"]
            ...     print(f"Progress: {progress.progress_percent:.1f}%")
            >>>
            >>> event_manager.subscribe(
            ...     EventType.SCAN_PROGRESS, handle_progress
            ... )
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = WeakSet()

        self._handlers[event_type].add(handler)
        logger.debug(f"Subscribed handler to {event_type.name}")

    def unsubscribe(
        self, event_type: EventType, handler: EventHandler
    ) -> bool:
        """Unsubscribe a handler from an event type.

        Args:
            event_type: Type of event to stop listening for
            handler: Handler function to remove

        Returns:
            True if handler was found and removed, False otherwise
        """
        if event_type not in self._handlers:
            return False

        try:
            self._handlers[event_type].discard(handler)
            logger.debug(f"Unsubscribed handler from {event_type.name}")
            return True
        except KeyError:
            return False

    async def emit_async(self, event: ScanEvent) -> None:
        """Emit an event to all subscribed handlers asynchronously.

        Args:
            event: Event to emit to handlers

        Note:
            Synchronous handlers are executed in a thread pool to avoid
            blocking the event loop. Errors in handlers are logged but
            don't stop other handlers from executing.
        """
        async with self._emit_lock:
            handlers = self._handlers.get(event.event_type, WeakSet())

            if not handlers:
                logger.debug(f"No handlers for {event.event_type.name}")
                return

            # Execute all handlers concurrently
            tasks = []
            for handler in list(
                handlers
            ):  # Copy to avoid modification during iteration
                try:
                    if asyncio.iscoroutinefunction(handler):
                        # Async handler - execute directly
                        task = asyncio.create_task(handler(event))
                        tasks.append(task)
                    else:
                        # Sync handler - execute in thread pool
                        loop = asyncio.get_event_loop()
                        task = loop.run_in_executor(None, handler, event)
                        tasks.append(task)
                except Exception as e:
                    logger.error(f"Error creating task for handler: {e}")

            # Wait for all handlers to complete
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Log any handler errors
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Handler {i} failed: {result}")

    def emit_sync(self, event: ScanEvent) -> None:
        """Emit an event synchronously (for non-async contexts).

        Args:
            event: Event to emit to handlers

        Note:
            Only synchronous handlers will be called. Async handlers
            are skipped with a warning logged.
        """
        handlers = self._handlers.get(event.event_type, WeakSet())

        if not handlers:
            logger.debug(f"No handlers for {event.event_type.name}")
            return

        for handler in list(handlers):
            try:
                if asyncio.iscoroutinefunction(handler):
                    logger.warning(
                        f"Skipping async handler in sync emit for "
                        f"{event.event_type.name}"
                    )
                    continue

                handler(event)
            except Exception as e:
                logger.error(f"Handler error in sync emit: {e}")

    def get_handler_count(self, event_type: EventType) -> int:
        """Get the number of handlers for an event type.

        Args:
            event_type: Event type to check

        Returns:
            Number of active handlers
        """
        handlers = self._handlers.get(event_type, WeakSet())
        return len(handlers)

    def clear_handlers(self, event_type: EventType | None = None) -> None:
        """Clear handlers for a specific event type or all events.

        Args:
            event_type: Specific event type to clear, or None for all
        """
        if event_type is None:
            self._handlers.clear()
            logger.info("Cleared all event handlers")
        else:
            self._handlers.pop(event_type, None)
            logger.info(f"Cleared handlers for {event_type.name}")


# Global event manager instance for convenience
_global_event_manager: EventManager | None = None


def get_event_manager() -> EventManager:
    """Get the global event manager instance.

    Returns:
        Global EventManager instance (created if needed)
    """
    global _global_event_manager
    if _global_event_manager is None:
        _global_event_manager = EventManager()
    return _global_event_manager


def reset_event_manager() -> None:
    """Reset the global event manager (useful for testing)."""
    global _global_event_manager
    if _global_event_manager:
        _global_event_manager.clear_handlers()
    _global_event_manager = None
