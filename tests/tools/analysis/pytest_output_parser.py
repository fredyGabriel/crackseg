from __future__ import annotations

import re
from typing import Any

from .test_failure_analysis import ErrorSeverity, FailureCategory, TestFailure


class PytestOutputParser:
    """Parses the output of a pytest run to extract failure information."""

    def __init__(self):
        self.error_patterns = {
            FailureCategory.IMPORT_ERROR: [
                r"ImportError|ModuleNotFoundError|cannot import  name",
                r"No module named",
            ],
            FailureCategory.MOCK_ERROR: [
                r"MagicMock|assert_called|mock",
                r"Expected.*to be called",
                r"patch|Mock",
            ],
            FailureCategory.CONFIG_ERROR: [
                r"hydra|config|configuration",
                r"YAML|yaml",
                r"ValidationError.*config",
            ],
            FailureCategory.ASSERTION_ERROR: [
                r"AssertionError",
                r"assert.*==|assert.*!=",
                r"Expected.*but got",
            ],
            FailureCategory.STREAMLIT_ERROR: [
                r"streamlit|session_state",
                r"st\.|MockSessionState",
            ],
            FailureCategory.ATTRIBUTE_ERROR: [
                r"AttributeError",
                r"has no attribute",
                r"object has no attribute",
            ],
            FailureCategory.TYPE_ERROR: [
                r"TypeError",
                r"takes.*positional arguments",
                r"unexpected keyword argument",
            ],
            FailureCategory.VALUE_ERROR: [
                r"ValueError",
                r"invalid literal",
                r"could not convert",
            ],
        }

    def parse(self, output: str) -> list[TestFailure]:
        """Parses pytest output and returns a list of TestFailure objects."""
        failures = []
        lines = output.split("\\n")
        current_failure_info: dict[str, Any] | None = None
        in_failure_section = False
        stack_trace_lines: list[str] = []

        for line in lines:
            if "FAILURES" in line or "ERRORS" in line:
                in_failure_section = True
                continue
            if in_failure_section and line.startswith("_"):
                if current_failure_info:
                    failures.append(
                        self._process_failure(
                            current_failure_info, stack_trace_lines
                        )
                    )
                current_failure_info = {
                    "test_name": self._extract_test_name(line),
                    "test_file": "",
                }
                stack_trace_lines = []
                continue
            if current_failure_info and in_failure_section:
                if line.strip():
                    stack_trace_lines.append(line)
                    if "tests/" in line and ".py:" in line:
                        file_match = re.search(r"(tests/[^:]+\\.py)", line)
                        if file_match:
                            current_failure_info["test_file"] = (
                                file_match.group(1)
                            )
            if "short test summary info" in line:
                if current_failure_info:
                    failures.append(
                        self._process_failure(
                            current_failure_info, stack_trace_lines
                        )
                    )
                break
        return failures

    def _extract_test_name(self, line: str) -> str:
        cleaned = line.strip("_").strip()
        if "::" in cleaned:
            parts = cleaned.split("::")
            return "::".join(parts[-2:]) if len(parts) >= 2 else cleaned
        return cleaned

    def _process_failure(
        self, failure_info: dict[str, Any], stack_trace_lines: list[str]
    ) -> TestFailure:
        stack_trace = "\\n".join(stack_trace_lines)
        error_message = "Unknown error"
        for line in reversed(stack_trace_lines):
            if line.strip().startswith("E   "):
                error_message = line.strip()[4:]
                break

        failure_type = self._determine_failure_type(stack_trace)
        category = self._categorize_failure(stack_trace, error_message)
        severity = self._determine_severity(
            category, error_message, stack_trace
        )
        affected_modules = self._extract_affected_modules(stack_trace)
        root_causes = self._identify_root_causes(
            category, error_message, stack_trace
        )
        suggested_fixes = self._generate_fix_suggestions(
            category, error_message
        )

        return TestFailure(
            test_name=failure_info["test_name"],
            test_file=failure_info.get("test_file", "unknown"),
            failure_type=failure_type,
            error_message=error_message,
            stack_trace=stack_trace,
            category=category,
            severity=severity,
            affected_modules=affected_modules,
            potential_root_causes=root_causes,
            suggested_fixes=suggested_fixes,
        )

    def _determine_failure_type(self, stack_trace: str) -> str:
        if (
            "ImportError" in stack_trace
            or "ModuleNotFoundError" in stack_trace
        ):
            return "ImportError"
        elif "AssertionError" in stack_trace:
            return "AssertionError"
        elif "AttributeError" in stack_trace:
            return "AttributeError"
        elif "TypeError" in stack_trace:
            return "TypeError"
        elif "ValueError" in stack_trace:
            return "ValueError"
        elif "ValidationError" in stack_trace:
            return "ValidationError"
        else:
            return "UnknownError"

    def _categorize_failure(
        self, stack_trace: str, error_message: str
    ) -> FailureCategory:
        combined_text = f"{stack_trace}\\n{error_message}".lower()
        for category, patterns in self.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    return category
        return FailureCategory.UNKNOWN_ERROR

    def _determine_severity(
        self, category: FailureCategory, error_message: str, stack_trace: str
    ) -> ErrorSeverity:
        if category in [
            FailureCategory.IMPORT_ERROR,
            FailureCategory.INFRASTRUCTURE_ERROR,
        ]:
            return ErrorSeverity.CRITICAL
        if category in [
            FailureCategory.CONFIG_ERROR,
            FailureCategory.STREAMLIT_ERROR,
        ]:
            if any(
                keyword in error_message.lower()
                for keyword in ["config", "session_state", "critical"]
            ):
                return ErrorSeverity.HIGH
        if category in [
            FailureCategory.MOCK_ERROR,
            FailureCategory.ATTRIBUTE_ERROR,
        ]:
            return ErrorSeverity.MEDIUM
        if category == FailureCategory.ASSERTION_ERROR:
            return ErrorSeverity.MEDIUM
        return ErrorSeverity.LOW

    def _extract_affected_modules(self, stack_trace: str) -> list[str]:
        modules: set[str] = set()
        module_patterns = [
            r"scripts\\.gui\\.[\\w.]+",
            r"src\\.[\\w.]+",
            r"tests\\.[\\w.]+",
        ]
        for pattern in module_patterns:
            matches = re.findall(pattern, stack_trace)
            modules.update(matches)
        return sorted(modules)

    def _identify_root_causes(
        self, category: FailureCategory, error_message: str, stack_trace: str
    ) -> list[str]:
        root_causes: list[str] = []
        if category == FailureCategory.IMPORT_ERROR:
            if "cannot import name" in error_message:
                root_causes.append(
                    "Missing or renamed function/class in module"
                )
            if "No module named" in error_message:
                root_causes.append("Missing module or incorrect import path")
        elif category == FailureCategory.MOCK_ERROR:
            if "assert_called" in error_message:
                root_causes.append("Mock method not called as expected")
        # ... (rest of the logic)
        return root_causes

    def _generate_fix_suggestions(
        self, category: FailureCategory, error_message: str
    ) -> list[str]:
        suggestions: list[str] = []
        if category == FailureCategory.IMPORT_ERROR:
            suggestions.extend(
                [
                    "Check import paths and module structure",
                    "Verify all required modules are installed",
                ]
            )
        # ... (rest of the logic)
        return suggestions
