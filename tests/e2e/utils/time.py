"""Time and duration utilities for E2E testing.

This module provides utilities for time-based operations, timeout management,
and performance measurement commonly used in E2E testing scenarios.
"""

import time
from collections.abc import Callable
from datetime import datetime
from typing import Any


def get_current_timestamp() -> float:
    """Get current timestamp in seconds since epoch.

    Returns:
        Current timestamp as float

    Example:
        >>> timestamp = get_current_timestamp()
        >>> isinstance(timestamp, float)
        True
    """
    return time.time()


def get_current_timestamp_ms() -> int:
    """Get current timestamp in milliseconds since epoch.

    Returns:
        Current timestamp in milliseconds

    Example:
        >>> timestamp_ms = get_current_timestamp_ms()
        >>> timestamp_ms > 0
        True
    """
    return int(time.time() * 1000)


def format_timestamp(
    timestamp: float, format_string: str = "%Y-%m-%d %H:%M:%S"
) -> str:
    """Format timestamp to human-readable string.

    Args:
        timestamp: Timestamp in seconds since epoch
        format_string: Format string for datetime formatting

    Returns:
        Formatted timestamp string

    Example:
        >>> formatted = format_timestamp(1609459200.0)
        >>> "2021" in formatted
        True
    """
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime(format_string)


def is_timestamp_recent(
    timestamp: float,
    max_age_seconds: float = 300.0,
) -> bool:
    """Check if timestamp is within specified age limit.

    Args:
        timestamp: Timestamp to check
        max_age_seconds: Maximum age in seconds

    Returns:
        True if timestamp is recent, False otherwise

    Example:
        >>> current = get_current_timestamp()
        >>> is_timestamp_recent(current, max_age_seconds=60)
        True
    """
    current_time = get_current_timestamp()
    age = current_time - timestamp
    return 0 <= age <= max_age_seconds


def wait_with_timeout(
    condition: Callable[[], bool],
    timeout_seconds: float = 30.0,
    poll_interval: float = 0.5,
    timeout_message: str = "Timeout waiting for condition",
) -> bool:
    """Wait for condition to become true with timeout.

    Args:
        condition: Function that returns boolean condition
        timeout_seconds: Maximum time to wait
        poll_interval: Time between condition checks
        timeout_message: Message for timeout exception

    Returns:
        True if condition met, False if timeout

    Example:
        >>> result = wait_with_timeout(lambda: True, timeout_seconds=1.0)
        >>> result
        True
    """
    start_time = get_current_timestamp()
    end_time = start_time + timeout_seconds

    while get_current_timestamp() < end_time:
        try:
            if condition():
                return True
        except Exception:
            # Ignore exceptions in condition function
            pass

        time.sleep(poll_interval)

    return False


def measure_execution_time(func: Callable[[], Any]) -> tuple[Any, float]:
    """Measure execution time of a function.

    Args:
        func: Function to measure

    Returns:
        Tuple of (function_result, execution_time_seconds)

    Example:
        >>> result, duration = measure_execution_time(lambda: time.sleep(0.1))
        >>> duration >= 0.1
        True
    """
    start_time = get_current_timestamp()
    result = func()
    end_time = get_current_timestamp()

    duration = end_time - start_time
    return result, duration


def format_duration(duration_seconds: float) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        duration_seconds: Duration in seconds

    Returns:
        Formatted duration string

    Example:
        >>> format_duration(125.5)
        '2m 5.5s'
        >>> format_duration(3661)
        '1h 1m 1s'
    """
    if duration_seconds < 0:
        return "0s"

    hours = int(duration_seconds // 3600)
    minutes = int((duration_seconds % 3600) // 60)
    seconds = duration_seconds % 60

    parts = []

    if hours > 0:
        parts.append(f"{hours}h")

    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")

    if seconds > 0 or not parts:
        if abs(seconds - int(seconds)) < 0.01:  # Effectively an integer
            parts.append(f"{int(seconds)}s")
        else:
            parts.append(f"{seconds:.1f}s")

    return " ".join(parts)


def parse_duration_string(duration_str: str) -> float:
    """Parse human-readable duration string to seconds.

    Args:
        duration_str: Duration string (e.g., "1h 30m", "45s", "2m 15.5s")

    Returns:
        Duration in seconds

    Raises:
        ValueError: If duration string format is invalid

    Example:
        >>> parse_duration_string("1h 30m")
        5400.0
        >>> parse_duration_string("45.5s")
        45.5
    """
    import re

    if not duration_str:
        return 0.0

    duration_str = duration_str.strip().lower()
    total_seconds = 0.0

    # Pattern to match time components: number followed by unit
    pattern = r"(\d+(?:\.\d+)?)\s*([hms])"
    matches = re.findall(pattern, duration_str)

    if not matches:
        raise ValueError(f"Invalid duration format: {duration_str}")

    for value_str, unit in matches:
        value = float(value_str)

        if unit == "h":
            total_seconds += value * 3600
        elif unit == "m":
            total_seconds += value * 60
        elif unit == "s":
            total_seconds += value

    return total_seconds


def get_test_timeout(
    base_timeout: float = 30.0, multiplier: float = 1.0
) -> float:
    """Get timeout value adjusted for test environment.

    Args:
        base_timeout: Base timeout in seconds
        multiplier: Multiplier for timeout (useful for slow environments)

    Returns:
        Adjusted timeout value

    Example:
        >>> timeout = get_test_timeout(30.0, multiplier=2.0)
        >>> timeout
        60.0
    """
    # Could be enhanced to read from environment variables
    # or detect CI environment for automatic adjustment
    return base_timeout * multiplier


def create_timeout_message(
    operation: str,
    timeout_seconds: float,
    additional_info: str = "",
) -> str:
    """Create standardized timeout error message.

    Args:
        operation: Description of operation that timed out
        timeout_seconds: Timeout value used
        additional_info: Additional context information

    Returns:
        Formatted timeout message

    Example:
        >>> msg = create_timeout_message("page load", 30.0, "URL: test.com")
        >>> "page load" in msg and "30.0" in msg
        True
    """
    formatted_timeout = format_duration(timeout_seconds)
    message = f"Timeout after {formatted_timeout} waiting for: {operation}"

    if additional_info:
        message += f" ({additional_info})"

    return message


def sleep_with_jitter(
    base_delay: float,
    jitter_percent: float = 10.0,
) -> None:
    """Sleep with random jitter to avoid synchronized timing issues.

    Args:
        base_delay: Base delay in seconds
        jitter_percent: Percentage of jitter to add (0-100)

    Example:
        >>> import time
        >>> start = time.time()
        >>> sleep_with_jitter(0.1, jitter_percent=20.0)
        >>> duration = time.time() - start
        >>> 0.08 <= duration <= 0.12  # 0.1 ± 20%
        True
    """
    import random

    if jitter_percent < 0 or jitter_percent > 100:
        raise ValueError("Jitter percent must be between 0 and 100")

    jitter_amount = base_delay * (jitter_percent / 100)
    jitter = random.uniform(-jitter_amount, jitter_amount)
    actual_delay = max(0, base_delay + jitter)

    time.sleep(actual_delay)


def retry_with_backoff(
    func: Callable[[], Any],
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
) -> Any:
    """Retry function with exponential backoff.

    Args:
        func: Function to retry
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay between attempts
        backoff_factor: Factor to increase delay by
        max_delay: Maximum delay between attempts

    Returns:
        Result of successful function call

    Raises:
        Last exception if all attempts fail

    Example:
        >>> counter = 0
        >>> def flaky_func():
        ...     global counter
        ...     counter += 1
        ...     if counter < 3:
        ...         raise ValueError("Not ready")
        ...     return "success"
        >>> result = retry_with_backoff(flaky_func, max_attempts=3)
        >>> result
        'success'
    """
    last_exception = None
    delay = initial_delay

    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_exception = e

            if attempt < max_attempts - 1:  # Not the last attempt
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)

    # All attempts failed, raise the last exception
    if last_exception:
        raise last_exception


def calculate_rate_per_second(count: int, duration_seconds: float) -> float:
    """Calculate rate per second from count and duration.

    Args:
        count: Number of events/operations
        duration_seconds: Duration in seconds

    Returns:
        Rate per second

    Example:
        >>> rate = calculate_rate_per_second(100, 10.0)
        >>> rate
        10.0
    """
    if duration_seconds <= 0:
        return 0.0

    return count / duration_seconds


def time_range_overlaps(
    start1: float,
    end1: float,
    start2: float,
    end2: float,
) -> bool:
    """Check if two time ranges overlap.

    Args:
        start1: Start time of first range
        end1: End time of first range
        start2: Start time of second range
        end2: End time of second range

    Returns:
        True if ranges overlap, False otherwise

    Example:
        >>> time_range_overlaps(1.0, 5.0, 3.0, 7.0)
        True
        >>> time_range_overlaps(1.0, 3.0, 5.0, 7.0)
        False
    """
    return max(start1, start2) < min(end1, end2)
