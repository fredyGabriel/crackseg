"""
Comprehensive Error Reporting System for Configuration Validation.

This module provides advanced error reporting capabilities with detailed
context, suggestions, and actionable recommendations for fixing configuration
validation errors in crack segmentation projects.

Key Features:
- Categorized error reporting with severity levels
- Context-aware error messages with line/column information
- Intelligent suggestions based on error patterns
- HTML and plain text report generation
- Error aggregation and deduplication
- User-friendly explanations for technical issues
"""

from .error_reporting import (
    ConfigErrorReporter,
    ErrorCategory,
    ErrorReport,
    ErrorSeverity,
    generate_error_report,
)

# Re-export the main functions and classes for backward compatibility
__all__ = [
    "generate_error_report",
    "ConfigErrorReporter",
    "ErrorReport",
    "ErrorSeverity",
    "ErrorCategory",
]
