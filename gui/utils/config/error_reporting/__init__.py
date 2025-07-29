"""
Error reporting module for CrackSeg configuration system.

This module provides comprehensive error reporting capabilities with detailed
context, suggestions, and actionable recommendations for fixing configuration
validation errors.
"""

from .core_reporter import ConfigErrorReporter
from .formatters import ErrorReportFormatter
from .report_models import ErrorCategory, ErrorReport, ErrorSeverity
from .utils import generate_error_report

__all__ = [
    "ConfigErrorReporter",
    "ErrorReport",
    "ErrorSeverity",
    "ErrorCategory",
    "ErrorReportFormatter",
    "generate_error_report",
]
