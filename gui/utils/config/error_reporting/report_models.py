"""
Data models for error reporting system.

This module defines the core data structures used for error reporting,
including error severity levels, categories, and report containers.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorSeverity(Enum):
    """Error severity levels for categorization."""

    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ErrorCategory(Enum):
    """Error categories for better organization."""

    SYNTAX = "syntax"
    SCHEMA = "schema"
    TYPE = "type"
    VALUE = "value"
    COMPATIBILITY = "compatibility"
    PERFORMANCE = "performance"
    SECURITY = "security"


@dataclass
class ErrorReport:
    """Comprehensive error report container."""

    title: str
    description: str
    severity: ErrorSeverity
    category: ErrorCategory
    field_name: str | None = None
    line: int | None = None
    column: int | None = None
    suggestions: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    related_docs: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
