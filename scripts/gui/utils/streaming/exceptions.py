"""Exception classes for log streaming system."""


class LogStreamingError(Exception):
    """Base exception for log streaming errors.

    Raised when log streaming operations fail due to:
    - File access permission errors
    - Thread synchronization issues
    - Invalid log source configuration
    - Stream processing failures
    - Resource cleanup problems

    Examples:
        >>> raise LogStreamingError("Failed to access log file")
        >>> raise LogStreamingError("Stream thread terminated unexpectedly")
    """

    pass


class LogSourceError(LogStreamingError):
    """Exception for log source specific errors.

    Raised when log source operations fail:
    - Log file not found or created
    - Directory watching failures
    - File permissions denied
    - Invalid log format detected
    """

    pass


class StreamProcessingError(LogStreamingError):
    """Exception for stream processing errors.

    Raised during log line processing:
    - Invalid log line format
    - Encoding/decoding errors
    - Queue overflow conditions
    - Callback execution failures
    """

    pass
