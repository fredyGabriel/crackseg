"""String manipulation utilities for E2E testing.

This module provides utilities for text processing, validation, and
formatting commonly used in Selenium testing and assertion validation.
"""

import re
from re import Pattern


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text for consistent comparisons.

    Args:
        text: Input text to normalize

    Returns:
        Text with normalized whitespace

    Example:
        >>> normalize_whitespace("  Hello   world  \\n\\t  ")
        'Hello world'
    """
    if not text:
        return ""

    # Replace multiple whitespace characters with single space
    normalized = re.sub(r"\s+", " ", text.strip())
    return normalized


def clean_element_text(text: str) -> str:
    """Clean text extracted from web elements.

    Args:
        text: Raw text from web element

    Returns:
        Cleaned text suitable for assertions

    Example:
        >>> clean_element_text("\\n  Button Text  \\t\\r")
        'Button Text'
    """
    if not text:
        return ""

    # Remove common artifacts from web element text
    cleaned = text.strip()
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Remove invisible characters
    cleaned = re.sub(r"[\u200b-\u200d\ufeff]", "", cleaned)

    return cleaned.strip()


def extract_numbers_from_text(text: str) -> list[float]:
    """Extract numeric values from text.

    Args:
        text: Text containing numbers

    Returns:
        List of extracted numeric values

    Example:
        >>> extract_numbers_from_text("Training: 85.5% complete, 42 epochs")
        [85.5, 42.0]
    """
    if not text:
        return []

    # Pattern to match integers and floats, including percentages
    number_pattern = r"-?\d+\.?\d*"
    matches = re.findall(number_pattern, text)

    numbers = []
    for match in matches:
        try:
            # Convert to float to handle both integers and decimals
            numbers.append(float(match))
        except ValueError:
            continue

    return numbers


def validate_text_contains(
    actual_text: str,
    expected_text: str,
    case_sensitive: bool = False,
    exact_match: bool = False,
) -> bool:
    """Validate that text contains expected content.

    Args:
        actual_text: Text to validate
        expected_text: Expected text content
        case_sensitive: Whether comparison is case-sensitive
        exact_match: Whether to require exact match

    Returns:
        True if validation passes, False otherwise

    Example:
        >>> validate_text_contains(
        ...     "Hello World", "hello", case_sensitive=False
        ... )
        True
        >>> validate_text_contains("Hello", "World", exact_match=True)
        False
    """
    if not actual_text or not expected_text:
        return False

    actual = actual_text if case_sensitive else actual_text.lower()
    expected = expected_text if case_sensitive else expected_text.lower()

    if exact_match:
        return actual == expected
    else:
        return expected in actual


def sanitize_filename(filename: str, replacement: str = "_") -> str:
    """Sanitize filename by removing invalid characters.

    Args:
        filename: Original filename
        replacement: Character to replace invalid characters

    Returns:
        Sanitized filename safe for filesystem

    Example:
        >>> sanitize_filename("test<file>name.txt")
        'test_file_name.txt'
    """
    if not filename:
        return "untitled"

    # Define invalid characters for most filesystems
    invalid_chars = r'[<>:"/\\|?*]'

    # Replace invalid characters
    sanitized = re.sub(invalid_chars, replacement, filename)

    # Remove control characters
    sanitized = re.sub(r"[\x00-\x1f\x7f-\x9f]", replacement, sanitized)

    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip(". ")

    # Ensure filename is not empty
    if not sanitized:
        sanitized = "untitled"

    return sanitized


def format_test_name(test_name: str) -> str:
    """Format test name for consistent display and logging.

    Args:
        test_name: Original test name

    Returns:
        Formatted test name

    Example:
        >>> format_test_name("test_user_can_upload_file")
        'Test User Can Upload File'
    """
    if not test_name:
        return "Unknown Test"

    # Remove test_ prefix if present
    formatted = re.sub(r"^test_", "", test_name)

    # Replace underscores with spaces
    formatted = formatted.replace("_", " ")

    # Capitalize words
    formatted = " ".join(word.capitalize() for word in formatted.split())

    # Add "Test" prefix if not present
    if not formatted.lower().startswith("test"):
        formatted = f"Test {formatted}"

    return formatted


def extract_error_message(text: str) -> str:
    """Extract error message from text that might contain stack traces.

    Args:
        text: Text containing error information

    Returns:
        Clean error message

    Example:
        >>> extract_error_message("Error: File not found\\nTraceback: ...")
        'File not found'
    """
    if not text:
        return "Unknown error"

    # Look for common error patterns
    error_patterns = [
        r"Error:\s*(.+?)(?:\n|$)",
        r"Exception:\s*(.+?)(?:\n|$)",
        r"Failed:\s*(.+?)(?:\n|$)",
        r"Invalid:\s*(.+?)(?:\n|$)",
    ]

    for pattern in error_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # If no pattern matches, return first non-empty line
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return lines[0] if lines else "Unknown error"


def truncate_text(
    text: str, max_length: int = 100, suffix: str = "..."
) -> str:
    """Truncate text to specified maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add when truncating

    Returns:
        Truncated text

    Example:
        >>> truncate_text("This is a very long text", max_length=10)
        'This is...'
    """
    if not text or len(text) <= max_length:
        return text

    # Account for suffix length
    effective_length = max_length - len(suffix)

    if effective_length <= 0:
        return suffix[:max_length]

    return text[:effective_length] + suffix


def parse_version_string(version_text: str) -> tuple[int, int, int] | None:
    """Parse version string into major, minor, patch tuple.

    Args:
        version_text: Version string (e.g., "1.2.3", "v2.0.1")

    Returns:
        Tuple of (major, minor, patch) or None if parsing fails

    Example:
        >>> parse_version_string("v1.2.3")
        (1, 2, 3)
        >>> parse_version_string("2.0")
        (2, 0, 0)
    """
    if not version_text:
        return None

    # Remove common prefixes
    cleaned = re.sub(r"^[vV]", "", version_text.strip())

    # Extract version numbers
    match = re.match(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", cleaned)

    if not match:
        return None

    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    patch = int(match.group(3) or 0)

    return (major, minor, patch)


def format_duration_text(duration_seconds: float) -> str:
    """Format duration in seconds to human-readable text.

    Args:
        duration_seconds: Duration in seconds

    Returns:
        Formatted duration string

    Example:
        >>> format_duration_text(125.5)
        '2m 5.5s'
        >>> format_duration_text(3661)
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
        if seconds == int(seconds):
            parts.append(f"{int(seconds)}s")
        else:
            parts.append(f"{seconds:.1f}s")

    return " ".join(parts)


def escape_regex_special_chars(text: str) -> str:
    """Escape special regex characters in text for literal matching.

    Args:
        text: Text that might contain regex special characters

    Returns:
        Text with regex special characters escaped

    Example:
        >>> escape_regex_special_chars("test[1].png")
        'test\\[1\\]\\.png'
    """
    if not text:
        return ""

    return re.escape(text)


def create_text_pattern(
    text: str,
    case_insensitive: bool = False,
    word_boundaries: bool = False,
) -> Pattern[str]:
    """Create regex pattern for text matching.

    Args:
        text: Text to create pattern for
        case_insensitive: Whether pattern should be case-insensitive
        word_boundaries: Whether to add word boundaries

    Returns:
        Compiled regex pattern

    Example:
        >>> pattern = create_text_pattern("hello", case_insensitive=True)
        >>> bool(pattern.search("Hello World"))
        True
    """
    escaped_text = escape_regex_special_chars(text)

    if word_boundaries:
        pattern_text = rf"\b{escaped_text}\b"
    else:
        pattern_text = escaped_text

    flags = re.IGNORECASE if case_insensitive else 0

    return re.compile(pattern_text, flags)
