"""Custom exceptions for the monitoring framework."""


class MonitoringError(Exception):
    """Base exception for all monitoring-related errors."""


class MetricCollectionError(MonitoringError):
    """Raised when there is an error collecting a metric."""


class CallbackError(MonitoringError):
    """Raised when a callback encounters an error."""
