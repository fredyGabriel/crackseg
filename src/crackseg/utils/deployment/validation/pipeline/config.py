"""Configuration classes for validation pipeline.

This module contains all configuration dataclasses used by the validation
pipeline system.
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation operation."""

    success: bool
    functional_tests_passed: bool = False
    performance_score: float = 0.0
    security_scan_passed: bool = False

    # Performance metrics
    inference_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    throughput_requests_per_second: float = 0.0

    # Security metrics
    vulnerabilities_found: int = 0
    security_score: float = 0.0

    # Error information
    error_message: str | None = None
    validation_details: dict[str, Any] | None = None


@dataclass
class ValidationThresholds:
    """Thresholds for validation checks."""

    # Performance thresholds
    max_inference_time_ms: float = 1000.0  # Max 1 second
    max_memory_usage_mb: float = 2048.0  # Max 2GB
    min_throughput_rps: float = 10.0  # Min 10 requests/second

    # Security thresholds
    max_vulnerabilities: int = 0  # Zero tolerance
    min_security_score: float = 8.0  # Min score out of 10

    # Test timeouts
    functional_test_timeout: int = 300  # seconds
    performance_test_timeout: int = 600  # seconds
    security_scan_timeout: int = 900  # seconds
