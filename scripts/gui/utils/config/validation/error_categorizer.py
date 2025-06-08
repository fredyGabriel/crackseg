"""
Error categorization engine for YAML configuration validation.

This module provides advanced error categorization, severity assessment,
and contextual suggestion generation for improved user experience.
"""

import re
from enum import Enum
from typing import Any

from ..exceptions import ValidationError


class ErrorSeverity(Enum):
    """Error severity levels for user-friendly categorization."""

    CRITICAL = "critical"  # Prevents configuration from being used
    WARNING = "warning"  # Configuration may work but has issues
    INFO = "info"  # Suggestions for improvement
    SUGGESTION = "suggestion"  # Best practices recommendations


class ErrorCategory(Enum):
    """Categories of configuration errors."""

    SYNTAX = "syntax"  # YAML syntax issues
    STRUCTURE = "structure"  # Missing required sections
    TYPE = "type"  # Wrong data types
    VALUE = "value"  # Invalid values or ranges
    COMPATIBILITY = "compatibility"  # Hydra/framework compatibility
    PERFORMANCE = "performance"  # Performance-related suggestions
    SECURITY = "security"  # Security considerations


class CategorizedError:
    """Enhanced error with categorization and user-friendly messaging."""

    def __init__(
        self,
        original_error: ValidationError,
        severity: ErrorSeverity,
        category: ErrorCategory,
        user_message: str,
        suggestions: list[str],
        context: dict[str, Any] | None = None,
        quick_fixes: list[str] | None = None,
    ) -> None:
        """Initialize categorized error.

        Args:
            original_error: Original ValidationError instance.
            severity: Error severity level.
            category: Error category.
            user_message: User-friendly error message.
            suggestions: List of actionable suggestions.
            context: Additional context information.
            quick_fixes: List of quick fix suggestions.
        """
        self.original_error = original_error
        self.severity = severity
        self.category = category
        self.user_message = user_message
        self.suggestions = suggestions
        self.context = context or {}
        self.quick_fixes = quick_fixes or []

    @property
    def line(self) -> int | None:
        """Get line number from original error."""
        return self.original_error.line

    @property
    def column(self) -> int | None:
        """Get column number from original error."""
        return self.original_error.column

    @property
    def field(self) -> str | None:
        """Get field name from original error."""
        return self.original_error.field

    @property
    def emoji(self) -> str:
        """Get emoji representation for severity."""
        return {
            ErrorSeverity.CRITICAL: "🔴",
            ErrorSeverity.WARNING: "🟡",
            ErrorSeverity.INFO: "🔵",
            ErrorSeverity.SUGGESTION: "💡",
        }[self.severity]

    @property
    def color(self) -> str:
        """Get color name for UI styling."""
        return {
            ErrorSeverity.CRITICAL: "error",
            ErrorSeverity.WARNING: "warning",
            ErrorSeverity.INFO: "info",
            ErrorSeverity.SUGGESTION: "success",
        }[self.severity]


class ErrorCategorizer:
    """Advanced error categorization engine."""

    def __init__(self) -> None:
        """Initialize the error categorizer."""
        # Pattern recognition for common errors
        self.syntax_patterns = {
            r"could not find expected.*:": {
                "message": "Missing colon (:) after a key",
                "suggestions": [
                    "Add ':' after the key name",
                    "Ensure all keys follow the 'key: value' format",
                ],
                "quick_fixes": ["Add ':' after '{key}'"],
            },
            r"found unexpected end of stream": {
                "message": "YAML file is incomplete or malformed",
                "suggestions": [
                    "Check for missing closing braces or quotes",
                    "Ensure all blocks are complete",
                    "Make sure the file is not truncated",
                ],
                "quick_fixes": ["Complete missing blocks"],
            },
            r"mapping values are not allowed": {
                "message": "Mapping structure formatting issue",
                "suggestions": [
                    "Check indentation - it must be consistent",
                    "Ensure there are no unescaped special characters",
                    "Use correct YAML syntax",
                ],
                "quick_fixes": [
                    "Fix indentation",
                    "Escape special characters",
                ],
            },
            r"found character.*that cannot start": {
                "message": "Invalid character at the start of a value",
                "suggestions": [
                    "Wrap strings in quotes if they contain special "
                    "characters",
                    "Check for extra spaces before the value",
                    "Verify YAML syntax for the data type",
                ],
                "quick_fixes": [
                    "Add quotes to the value",
                    "Remove extra spaces",
                ],
            },
        }

        self.structure_patterns = {
            r"missing.*required.*section": {
                "message": "Missing required section in configuration",
                "suggestions": [
                    "Add the missing section using the standard template",
                    "Check examples in the configs/ directory",
                    "Ensure all main sections are present",
                ],
                "quick_fixes": ["Add section from template"],
            },
            r"missing.*required.*field": {
                "message": "Missing required field",
                "suggestions": [
                    "Add the required field with a valid value",
                    "Check the documentation for default values",
                    "Review similar configurations as reference",
                ],
                "quick_fixes": ["Add field with default value"],
            },
        }

        self.type_patterns = {
            r"expected.*got": {
                "message": "Data type does not match expected",
                "suggestions": [
                    "Change the value to the correct type",
                    "Check the documentation for the expected format",
                    "Use quotes for strings, no quotes for integers",
                ],
                "quick_fixes": ["Convert to correct type"],
            },
        }

        self.value_patterns = {
            r"unknown.*architecture": {
                "message": "Unknown model architecture",
                "suggestions": [
                    "Use a valid architecture: unet, deeplabv3plus, swin_unet",
                    "Check the list of available architectures",
                    "Review the model documentation",
                ],
                "quick_fixes": ["Select from valid list"],
            },
            r"unknown.*encoder": {
                "message": "Unknown encoder",
                "suggestions": [
                    "Use a valid encoder: resnet50, efficientnet_b4, "
                    "swin_base",
                    "Check available encoders in src/model/encoder/",
                    "Verify compatibility with the selected architecture",
                ],
                "quick_fixes": ["Select compatible encoder"],
            },
            r"invalid.*value.*must be positive": {
                "message": "Value must be a positive number",
                "suggestions": [
                    "Use a number greater than 0",
                    "Check that it is appropriate for the parameter",
                    "See recommended ranges in the documentation",
                ],
                "quick_fixes": ["Change to positive value"],
            },
        }

    def categorize_error(
        self, error: ValidationError, content: str = ""
    ) -> CategorizedError:
        """Categorize a validation error with enhanced context.

        Args:
            error: Original validation error.
            content: Full YAML content for context analysis.

        Returns:
            Categorized error with user-friendly messaging.
        """
        # Get the original message (first argument passed to ValidationError)
        error_message = (
            error.args[0].lower() if error.args else str(error).lower()
        )

        # Determine category and severity
        category, severity = self._determine_category_and_severity(
            error_message
        )

        # Generate user-friendly message and suggestions
        user_message, suggestions, quick_fixes = self._generate_messaging(
            error, error_message, content
        )

        # Add contextual information
        context = self._build_context(error, content)

        return CategorizedError(
            original_error=error,
            severity=severity,
            category=category,
            user_message=user_message,
            suggestions=suggestions,
            context=context,
            quick_fixes=quick_fixes,
        )

    def categorize_errors(
        self, errors: list[ValidationError], content: str = ""
    ) -> list[CategorizedError]:
        """Categorize multiple errors and sort by severity.

        Args:
            errors: List of validation errors.
            content: Full YAML content for context analysis.

        Returns:
            List of categorized errors sorted by severity.
        """
        categorized = [
            self.categorize_error(error, content) for error in errors
        ]

        # Sort by severity (critical first)
        severity_order = {
            ErrorSeverity.CRITICAL: 0,
            ErrorSeverity.WARNING: 1,
            ErrorSeverity.INFO: 2,
            ErrorSeverity.SUGGESTION: 3,
        }

        return sorted(categorized, key=lambda e: severity_order[e.severity])

    def _determine_category_and_severity(
        self, error_message: str
    ) -> tuple[ErrorCategory, ErrorSeverity]:
        """Determine error category and severity from message using
        precise patterns.
        """

        # Define specific regex patterns for each category
        syntax_patterns = [
            r"yaml\s+syntax\s+error",
            r"could not find expected\s*[:\'\"]",
            r"found unexpected end of stream",
            r"found character.*that cannot start",
            r"mapping values are not allowed",
            r"while parsing.*expected",
            r"scanner error",
            r"parser error",
        ]

        structure_patterns = [
            r"missing\s+(required\s+)?(section|field)",
            r"required\s+(section|field).*missing",
            r"configuration.*missing.*section",
            r"no such.*section",
        ]

        type_patterns = [
            r"expected\s+\w+,?\s+got\s+\w+",
            r"invalid\s+type.*expected.*got",
            r"type\s+error.*expected",
            r"must\s+be\s+(int|str|float|bool|list|dict)",
            r"cannot\s+convert.*to\s+\w+",
        ]

        value_patterns = [
            r"unknown\s+(model\s+)?architecture",
            r"unknown\s+encoder",
            r"invalid\s+value.*range",
            r"value.*out\s+of\s+range",
            r"not\s+allowed.*value",
            r"unsupported.*value",
        ]

        compatibility_patterns = [
            r"hydra.*error",
            r"composition.*error",
            r"override.*error",
            r"compatibility.*issue",
        ]

        # Check patterns in order of specificity (most specific first)

        # 1. Check syntax errors (CRITICAL)
        for pattern in syntax_patterns:
            if re.search(pattern, error_message, re.IGNORECASE):
                return ErrorCategory.SYNTAX, ErrorSeverity.CRITICAL

        # 2. Check structure errors (CRITICAL)
        for pattern in structure_patterns:
            if re.search(pattern, error_message, re.IGNORECASE):
                return ErrorCategory.STRUCTURE, ErrorSeverity.CRITICAL

        # 3. Check type errors (WARNING)
        for pattern in type_patterns:
            if re.search(pattern, error_message, re.IGNORECASE):
                return ErrorCategory.TYPE, ErrorSeverity.WARNING

        # 4. Check value errors (WARNING)
        for pattern in value_patterns:
            if re.search(pattern, error_message, re.IGNORECASE):
                return ErrorCategory.VALUE, ErrorSeverity.WARNING

        # 5. Check compatibility issues (INFO)
        for pattern in compatibility_patterns:
            if re.search(pattern, error_message, re.IGNORECASE):
                return ErrorCategory.COMPATIBILITY, ErrorSeverity.INFO

        # 6. Fallback categorization based on keywords
        if any(
            keyword in error_message for keyword in ["deprecated", "obsolete"]
        ):
            return ErrorCategory.COMPATIBILITY, ErrorSeverity.INFO

        if any(
            keyword in error_message for keyword in ["warning", "recommend"]
        ):
            return ErrorCategory.PERFORMANCE, ErrorSeverity.SUGGESTION

        # Default to structure/info for unrecognized errors
        return ErrorCategory.STRUCTURE, ErrorSeverity.INFO

    def _generate_messaging(
        self, error: ValidationError, error_message: str, content: str
    ) -> tuple[str, list[str], list[str]]:
        """Generate user-friendly message and suggestions."""
        # Try to match against known patterns
        for pattern_dict in [
            self.syntax_patterns,
            self.structure_patterns,
            self.type_patterns,
            self.value_patterns,
        ]:
            for pattern, info in pattern_dict.items():
                if re.search(pattern, error_message):
                    return (
                        str(info["message"]),
                        list(info["suggestions"]),
                        list(info.get("quick_fixes", [])),
                    )

        # Generate contextual suggestions for unmatched errors
        suggestions = self._generate_contextual_suggestions(error, content)
        quick_fixes = self._generate_quick_fixes(error)

        # Create user-friendly message
        original_message = error.args[0] if error.args else str(error)
        user_message = self._humanize_error_message(original_message)

        return user_message, suggestions, quick_fixes

    def _generate_contextual_suggestions(
        self, error: ValidationError, content: str
    ) -> list[str]:
        """Generate context-aware suggestions based on error and content."""
        suggestions = []

        # Add line-specific suggestions if we have line info
        if error.line is not None:
            suggestions.extend(self._analyze_line_context(error.line, content))

        # Add field-specific suggestions if we have field info
        if error.field:
            suggestions.extend(self._analyze_field_context(error.field))

        # Add general suggestions from original error
        suggestions.extend(error.suggestions)

        # Remove duplicates and return
        return list(dict.fromkeys(suggestions))

    def _generate_quick_fixes(self, error: ValidationError) -> list[str]:
        """Generate quick fix suggestions."""
        quick_fixes = []

        if error.field:
            if "architecture" in error.field:
                quick_fixes.extend(
                    [
                        "Select architecture: unet, deeplabv3plus, swin_unet",
                        "Available architecture options in model config",
                    ]
                )
            elif "encoder" in error.field:
                quick_fixes.extend(
                    [
                        "Use encoder: resnet50, efficientnet_b4",
                        "Check encoder compatibility with architecture",
                    ]
                )
            elif "epochs" in error.field:
                quick_fixes.extend(
                    [
                        "Use positive number: epochs: 100",
                        "Typical values: 50-200 epochs",
                    ]
                )
            elif "batch_size" in error.field:
                quick_fixes.extend(
                    [
                        "Use power of 2: batch_size: 16",
                        "RTX 3070 Ti recommended: 16-32",
                    ]
                )

        return quick_fixes

    def _build_context(
        self, error: ValidationError, content: str
    ) -> dict[str, Any]:
        """Build additional context information."""
        context = {}

        if error.line is not None and content:
            lines = content.split("\n")
            if 0 <= error.line - 1 < len(lines):
                # Get context around the error line
                start_line = max(0, error.line - 3)
                end_line = min(len(lines), error.line + 2)

                context["line_content"] = lines[error.line - 1]
                context["context_lines"] = lines[start_line:end_line]
                context["context_start_line"] = start_line + 1

        if error.field:
            context["field_path"] = error.field.split(".")
            context["section"] = (
                error.field.split(".")[0]
                if "." in error.field
                else error.field
            )

        return context

    def _analyze_line_context(self, line_num: int, content: str) -> list[str]:
        """Analyze the context around a specific line."""
        suggestions = []

        if not content:
            return suggestions

        lines = content.split("\n")
        if 0 <= line_num - 1 < len(lines):
            line = lines[line_num - 1]

            # Check for common issues
            if "\t" in line:
                suggestions.append(
                    "Use spaces instead of tabs for indentation"
                )

            if line.rstrip() != line:
                suggestions.append(
                    "Remove trailing spaces at the end of the line"
                )

            if line.strip().endswith(":") and line_num < len(lines):
                next_line = lines[line_num] if line_num < len(lines) else ""
                if next_line and not next_line.startswith(" "):
                    suggestions.append("The next line must be indented")

        return suggestions

    def _analyze_field_context(self, field: str) -> list[str]:
        """Analyze context based on field name."""
        field_suggestions = {
            "model.architecture": [
                "Available architectures: unet, deeplabv3plus, swin_unet",
                "See docs/guides/model_architectures.md",
            ],
            "model.encoder": [
                "Popular encoders: resnet50, efficientnet_b4, swin_base",
                "Check compatibility with the architecture",
            ],
            "training.epochs": [
                "Typical values: 50-200 for initial training",
                "Consider early stopping to avoid overfitting",
            ],
            "training.batch_size": [
                "Use powers of 2: 8, 16, 32 (depending on GPU memory)",
                "RTX 3070 Ti: batch_size 16-32 recommended",
            ],
            "training.learning_rate": [
                "Typical ranges: 1e-4 to 1e-2",
                "Consider using a learning rate scheduler",
            ],
        }

        return field_suggestions.get(field, [])

    def _humanize_error_message(self, message: str) -> str:
        """Convert technical error message to user-friendly format."""
        # Remove technical prefixes
        message = re.sub(
            r"^(YAML syntax error: |Hydra validation error: )", "", message
        )

        # Replace technical terms
        replacements = {
            "mapping values are not allowed": "mapping format issue",
            "could not find expected ':'": "missing colon (:)",
            "found unexpected end of stream": "incomplete file",
            "found character": "invalid character",
            "expected": "should be",
            "but found": "but got",
        }

        for technical, friendly in replacements.items():
            message = message.replace(technical, friendly)

        return message.capitalize()
