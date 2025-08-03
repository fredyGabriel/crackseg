"""
Performance optimization utilities for GUI components.

This module provides comprehensive performance optimization including CSS
caching,
update debouncing, memory management, and performance monitoring for better
user experience and resource efficiency.
"""

import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Optional
from weakref import WeakSet

import streamlit as st


class PerformanceOptimizer:
    """Central performance optimization manager for GUI components."""

    _instance: Optional["PerformanceOptimizer"] = None
    _css_cache: set[str] = set()
    _update_timestamps: dict[str, float] = {}
    _placeholders: WeakSet[Any] = WeakSet()
    _performance_metrics: dict[str, dict[str, Any]] = defaultdict(dict)

    def __new__(cls) -> "PerformanceOptimizer":
        """Singleton pattern for global performance management."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "PerformanceOptimizer":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def inject_css_once(self, css_key: str, css_content: str) -> None:
        """
        Inject CSS only once per session to avoid repeated injection.

        Args:
            css_key: Unique identifier for the CSS block
            css_content: CSS content to inject
        """
        if css_key not in self._css_cache:
            st.markdown(css_content, unsafe_allow_html=True)
            self._css_cache.add(css_key)

    def should_update(
        self, component_id: str, min_interval: float = 0.1
    ) -> bool:
        """
        Debounce updates to prevent excessive rendering.

        Args:
            component_id: Unique identifier for the component
            min_interval: Minimum time interval between updates in seconds

        Returns:
            True if update should proceed, False otherwise
        """
        current_time = time.time()
        last_update = self._update_timestamps.get(component_id, 0)

        if current_time - last_update >= min_interval:
            self._update_timestamps[component_id] = current_time
            return True
        return False

    def register_placeholder(self, placeholder: Any) -> None:
        """
        Register a placeholder for proper cleanup.

        Args:
            placeholder: Streamlit placeholder object
        """
        self._placeholders.add(placeholder)

    def cleanup_placeholders(self) -> None:
        """Clean up all registered placeholders."""
        for placeholder in self._placeholders:
            try:
                placeholder.empty()
            except Exception:
                # Placeholder might already be cleaned up
                pass

    def track_performance(
        self, component_id: str, operation: str, start_time: float
    ) -> None:
        """
        Track performance metrics for components.

        Args:
            component_id: Unique identifier for the component
            operation: Operation being tracked
            start_time: Start time of the operation
        """
        duration = time.time() - start_time

        if component_id not in self._performance_metrics:
            self._performance_metrics[component_id] = {
                "operations": [],
                "total_time": 0,
                "count": 0,
                "average_time": 0,
            }

        metrics = self._performance_metrics[component_id]
        metrics["operations"].append(
            {
                "operation": operation,
                "duration": duration,
                "timestamp": time.time(),
            }
        )
        metrics["total_time"] += duration
        metrics["count"] += 1
        metrics["average_time"] = metrics["total_time"] / metrics["count"]

    def get_performance_report(self) -> dict[str, Any]:
        """
        Get comprehensive performance report.

        Returns:
            Performance metrics for all tracked components
        """
        return dict(self._performance_metrics)

    def reset_performance_metrics(self) -> None:
        """Reset all performance metrics."""
        self._performance_metrics.clear()
        self._update_timestamps.clear()


class OptimizedHTMLBuilder:
    """Optimized HTML builder with template caching and efficient string
    ops."""

    _template_cache: dict[str, str] = {}

    @staticmethod
    def build_progress_html(
        title: str,
        progress: float,
        step_info: str,
        elapsed_str: str,
        remaining_str: str,
        description: str = "",
        state: str = "normal",
    ) -> str:
        """
        Build optimized progress HTML using cached templates.

        Args:
            title: Progress title
            progress: Progress value (0.0 to 1.0)
            step_info: Step information string
            elapsed_str: Formatted elapsed time
            remaining_str: Formatted remaining time
            description: Optional description
            state: Progress state ('normal', 'success', 'warning')

        Returns:
            Optimized HTML string
        """
        template_key = f"progress_{state}"

        if template_key not in OptimizedHTMLBuilder._template_cache:
            template = """
            <div class="crackseg-progress-container crackseg-progress-{state}">
                <div class="crackseg-progress-header">
                    <h4 class="crackseg-progress-title">{title}</h4>
                    <span class="crackseg-progress-percentage">
                        {percentage}</span>
                </div>
                <div class="crackseg-progress-bar-container">
                    <div class="crackseg-progress-bar-fill"
                         style="width: {width};"></div>
                </div>
                <div class="crackseg-progress-info">
                    <span class="crackseg-progress-step">{step_info}</span>
                    <span class="crackseg-progress-time">
                        Elapsed: {elapsed} | Remaining: ~{remaining}
                    </span>
                </div>
                {description_html}
            </div>
            """
            OptimizedHTMLBuilder._template_cache[template_key] = template

        template = OptimizedHTMLBuilder._template_cache[template_key]

        # Optimize string formatting
        percentage_value = progress * 100
        if percentage_value == int(percentage_value):
            percentage = f"{int(percentage_value)}%"
        else:
            percentage = f"{percentage_value:.1f}%"
        width = f"{progress * 100}%"
        description_html = (
            f'<div class="crackseg-progress-description">{description}</div>'
            if description
            else ""
        )

        return template.format(
            state=state,
            title=title,
            percentage=percentage,
            width=width,
            step_info=step_info,
            elapsed=elapsed_str,
            remaining=remaining_str,
            description_html=description_html,
        )

    @staticmethod
    def build_spinner_html(
        message: str,
        subtext: str = "",
        spinner_type: str = "default",
    ) -> str:
        """
        Build optimized spinner HTML using cached templates.

        Args:
            message: Spinner message
            subtext: Optional subtext
            spinner_type: Type of spinner animation

        Returns:
            Optimized HTML string
        """
        template_key = f"spinner_{spinner_type}"

        if template_key not in OptimizedHTMLBuilder._template_cache:
            template = """
            <div class="crackseg-spinner-container">
                <div class="crackseg-spinner-icon"></div>
                <div class="crackseg-spinner-text">{message}</div>
                {subtext_html}
            </div>
            """
            OptimizedHTMLBuilder._template_cache[template_key] = template

        template = OptimizedHTMLBuilder._template_cache[template_key]

        subtext_html = (
            f'<div class="crackseg-spinner-subtext">{subtext}</div>'
            if subtext
            else ""
        )

        return template.format(
            message=message,
            subtext_html=subtext_html,
        )


class AsyncOperationManager:
    """Manage async operations without blocking the UI thread."""

    _active_operations: dict[str, dict[str, Any]] = {}

    @staticmethod
    def start_operation(
        operation_id: str,
        title: str,
        estimated_duration: float = 0,
    ) -> None:
        """
        Start tracking an async operation.

        Args:
            operation_id: Unique identifier for the operation
            title: Human-readable title
            estimated_duration: Estimated duration in seconds
        """
        AsyncOperationManager._active_operations[operation_id] = {
            "title": title,
            "start_time": time.time(),
            "estimated_duration": estimated_duration,
            "status": "running",
        }

    @staticmethod
    def update_operation(
        operation_id: str,
        progress: float,
        status: str = "running",
    ) -> None:
        """
        Update operation progress.

        Args:
            operation_id: Operation identifier
            progress: Progress value (0.0 to 1.0)
            status: Operation status
        """
        if operation_id in AsyncOperationManager._active_operations:
            AsyncOperationManager._active_operations[operation_id].update(
                {
                    "progress": progress,
                    "status": status,
                    "last_update": time.time(),
                }
            )

    @staticmethod
    def finish_operation(operation_id: str, success: bool = True) -> None:
        """
        Mark operation as finished.

        Args:
            operation_id: Operation identifier
            success: Whether operation was successful
        """
        if operation_id in AsyncOperationManager._active_operations:
            AsyncOperationManager._active_operations[operation_id].update(
                {
                    "status": "completed" if success else "failed",
                    "end_time": time.time(),
                }
            )

    @staticmethod
    def get_operation_status(operation_id: str) -> dict[str, Any] | None:
        """
        Get operation status.

        Args:
            operation_id: Operation identifier

        Returns:
            Operation status dictionary or None
        """
        return AsyncOperationManager._active_operations.get(operation_id)

    @staticmethod
    def cleanup_completed_operations() -> None:
        """Clean up completed operations older than 5 minutes."""
        current_time = time.time()
        to_remove = []

        for op_id, op_data in AsyncOperationManager._active_operations.items():
            if (
                op_data.get("status") in ("completed", "failed")
                and current_time - op_data.get("end_time", 0) > 300
            ):
                to_remove.append(op_id)

        for op_id in to_remove:
            del AsyncOperationManager._active_operations[op_id]


class MemoryManager:
    """Manage memory usage and cleanup for GUI components."""

    _memory_usage: dict[str, dict[str, Any]] = {}
    _cleanup_callbacks: dict[str, list[Callable[[], None]]] = defaultdict(list)

    @staticmethod
    def track_memory_usage(
        component_id: str,
        operation: str,
        memory_delta: float,
    ) -> None:
        """
        Track memory usage for components.

        Args:
            component_id: Component identifier
            operation: Operation being tracked
            memory_delta: Memory change in MB
        """
        if component_id not in MemoryManager._memory_usage:
            MemoryManager._memory_usage[component_id] = {
                "total_allocated": 0,
                "peak_usage": 0,
                "operations": [],
            }

        data = MemoryManager._memory_usage[component_id]
        data["total_allocated"] += memory_delta
        data["peak_usage"] = max(data["peak_usage"], data["total_allocated"])
        data["operations"].append(
            {
                "operation": operation,
                "delta": memory_delta,
                "timestamp": time.time(),
            }
        )

    @staticmethod
    def register_cleanup_callback(
        component_id: str,
        callback: Callable[[], None],
    ) -> None:
        """
        Register cleanup callback for component.

        Args:
            component_id: Component identifier
            callback: Cleanup function to call
        """
        MemoryManager._cleanup_callbacks[component_id].append(callback)

    @staticmethod
    def cleanup_component(component_id: str) -> None:
        """
        Clean up all resources for a component.

        Args:
            component_id: Component identifier
        """
        # Execute cleanup callbacks
        for callback in MemoryManager._cleanup_callbacks.get(component_id, []):
            try:
                callback()
            except Exception:
                # Ignore cleanup errors
                pass

        # Clear tracking data
        MemoryManager._memory_usage.pop(component_id, None)
        MemoryManager._cleanup_callbacks.pop(component_id, None)

    @staticmethod
    def get_memory_report() -> dict[str, Any]:
        """
        Get comprehensive memory usage report.

        Returns:
            Memory usage statistics
        """
        return dict(MemoryManager._memory_usage)


# Global optimizer instance
_optimizer = PerformanceOptimizer()


def get_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer instance."""
    return _optimizer


def inject_css_once(css_key: str, css_content: str) -> None:
    """Convenience function for CSS injection."""
    _optimizer.inject_css_once(css_key, css_content)


def should_update(component_id: str, min_interval: float = 0.1) -> bool:
    """Convenience function for update debouncing."""
    return _optimizer.should_update(component_id, min_interval)


def track_performance(
    component_id: str, operation: str, start_time: float
) -> None:
    """Convenience function for performance tracking."""
    _optimizer.track_performance(component_id, operation, start_time)
