#!/usr/bin/env python3
"""
Test Quality Validator - CrackSeg Project

This script validates the quality of test files to ensure they meet
project standards for documentation, structure, and completeness.

Usage:
    python scripts/validate_test_quality.py [test_files...]
"""

import ast
import sys
from pathlib import Path
from typing import Any


class TestQualityValidator:
    """Validates test file quality according to project standards."""

    def __init__(self) -> None:
        """Initialize the validator."""
        self.required_imports = {"pytest", "torch", "unittest.mock"}
        self.issues: list[dict[str, Any]] = []

    def validate_file(self, test_file: str) -> list[dict[str, Any]]:
        """
        Validate a single test file.

        Args:
            test_file: Path to test file

        Returns:
            List of validation issues found
        """
        issues = []
        file_path = Path(test_file)

        if not file_path.exists():
            issues.append(
                {
                    "type": "error",
                    "file": str(file_path),
                    "message": "Test file does not exist",
                    "line": 0,
                }
            )
            return issues

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            # Validate file structure
            issues.extend(self._validate_file_structure(tree, file_path))

            # Validate imports
            issues.extend(self._validate_imports(tree, file_path))

            # Validate test functions
            issues.extend(self._validate_test_functions(tree, file_path))

            # Validate docstrings
            issues.extend(self._validate_docstrings(tree, file_path))

            # Validate type annotations
            issues.extend(self._validate_type_annotations(tree, file_path))

        except SyntaxError as e:
            issues.append(
                {
                    "type": "error",
                    "file": str(file_path),
                    "message": f"Syntax error: {e}",
                    "line": e.lineno or 0,
                }
            )
        except Exception as e:
            issues.append(
                {
                    "type": "error",
                    "file": str(file_path),
                    "message": f"Validation error: {e}",
                    "line": 0,
                }
            )

        return issues

    def _validate_file_structure(
        self, tree: ast.AST, file_path: Path
    ) -> list[dict[str, Any]]:
        """Validate overall file structure."""
        issues = []

        # Check for module docstring - cast to Module since we know it's parsed
        # from file
        if isinstance(tree, ast.Module) and not ast.get_docstring(tree):
            issues.append(
                {
                    "type": "warning",
                    "file": str(file_path),
                    "message": "Test file missing module docstring",
                    "line": 1,
                }
            )

        # Check for test functions
        test_functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name.startswith("test_")
        ]

        if not test_functions:
            issues.append(
                {
                    "type": "warning",
                    "file": str(file_path),
                    "message": (
                        "No test functions found (functions should start "
                        "with test_)"
                    ),
                    "line": 1,
                }
            )

        # Check for class structure if present
        test_classes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test")
        ]

        for cls in test_classes:
            if not ast.get_docstring(cls):
                issues.append(
                    {
                        "type": "warning",
                        "file": str(file_path),
                        "message": f"Test class {cls.name} missing docstring",
                        "line": cls.lineno,
                    }
                )

        return issues

    def _validate_imports(
        self, tree: ast.AST, file_path: Path
    ) -> list[dict[str, Any]]:
        """Validate import statements."""
        issues = []
        imports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])

        # Check for required imports
        if "pytest" not in imports:
            issues.append(
                {
                    "type": "warning",
                    "file": str(file_path),
                    "message": (
                        "Missing pytest import - required for test framework"
                    ),
                    "line": 1,
                }
            )

        # Check for common testing patterns
        if "unittest.mock" not in str(imports) and "mock" not in imports:
            issues.append(
                {
                    "type": "info",
                    "file": str(file_path),
                    "message": (
                        "Consider importing unittest.mock for mocking "
                        "capabilities"
                    ),
                    "line": 1,
                }
            )

        return issues

    def _validate_test_functions(
        self, tree: ast.AST, file_path: Path
    ) -> list[dict[str, Any]]:
        """Validate test functions and methods."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith(
                "test_"
            ):
                # Check for assertions
                if not self._has_assertions(node):
                    issues.append(
                        {
                            "type": "warning",
                            "file": str(file_path),
                            "message": (
                                f"Test function {node.name} has no assertions"
                            ),
                            "line": node.lineno,
                        }
                    )

                # Check for docstring
                if not ast.get_docstring(node):
                    issues.append(
                        {
                            "type": "warning",
                            "file": str(file_path),
                            "message": (
                                f"Test function {node.name} missing docstring"
                            ),
                            "line": node.lineno,
                        }
                    )

        return issues

    def _validate_docstrings(
        self, tree: ast.AST, file_path: Path
    ) -> list[dict[str, Any]]:
        """Validate docstring presence and quality."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.ClassDef):
                docstring = ast.get_docstring(node)
                # Check if it's a function and if it's a fixture before adding
                # issue
                should_check = True
                if isinstance(node, ast.FunctionDef):
                    should_check = not self._is_fixture_function(node)

                if not docstring and should_check:
                    issues.append(
                        {
                            "type": "info",
                            "file": str(file_path),
                            "message": (
                                f"{node.__class__.__name__} {node.name} "
                                f"could benefit from a docstring"
                            ),
                            "line": node.lineno,
                        }
                    )

        return issues

    def _validate_type_annotations(
        self, tree: ast.AST, file_path: Path
    ) -> list[dict[str, Any]]:
        """Validate type annotations on test functions."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check return annotation
                if not node.returns and node.name.startswith("test_"):
                    issues.append(
                        {
                            "type": "info",
                            "file": str(file_path),
                            "message": (
                                f"Test function {node.name} missing return "
                                f"type annotation (should be -> None)"
                            ),
                            "line": node.lineno,
                        }
                    )

                # Check parameter annotations
                for arg in node.args.args:
                    if not arg.annotation and arg.arg != "self":
                        issues.append(
                            {
                                "type": "info",
                                "file": str(file_path),
                                "message": (
                                    f"Parameter {arg.arg} in {node.name} "
                                    f"missing type annotation"
                                ),
                                "line": node.lineno,
                            }
                        )

        return issues

    def _has_assertions(self, func_node: ast.FunctionDef) -> bool:
        """Check if function contains assertions."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Assert):
                return True
            # Check for pytest assertions
            if isinstance(node, ast.Call) and isinstance(
                node.func, ast.Attribute
            ):
                if node.func.attr in [
                    "assert_called",
                    "assert_called_with",
                    "assert_not_called",
                ]:
                    return True
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id.startswith("assert"):
                    return True
        return False

    def _is_fixture_function(self, func_node: ast.FunctionDef) -> bool:
        """Check if function is a pytest fixture."""
        for decorator in func_node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "fixture":
                return True
            if (
                isinstance(decorator, ast.Attribute)
                and decorator.attr == "fixture"
            ):
                return True
        return False

    def validate_files(
        self, test_files: list[str]
    ) -> tuple[bool, list[dict[str, Any]]]:
        """
        Validate multiple test files.

        Args:
            test_files: List of test file paths

        Returns:
            Tuple of (all_valid, all_issues)
        """
        all_issues = []

        for test_file in test_files:
            issues = self.validate_file(test_file)
            all_issues.extend(issues)

        # Consider validation successful if no errors (warnings are ok)
        has_errors = any(issue["type"] == "error" for issue in all_issues)

        return not has_errors, all_issues


def format_issues(issues: list[dict[str, Any]]) -> str:
    """Format validation issues for display."""
    if not issues:
        return "✅ All test files meet quality standards!"

    output = []
    output.append(f"Found {len(issues)} quality issues:\n")

    # Group by file
    by_file: dict[str, list[dict[str, Any]]] = {}
    for issue in issues:
        file_path = issue["file"]
        if file_path not in by_file:
            by_file[file_path] = []
        by_file[file_path].append(issue)

    for file_path, file_issues in by_file.items():
        output.append(f"📁 {file_path}")
        for issue in file_issues:
            icon = (
                "🚨"
                if issue["type"] == "error"
                else "⚠️" if issue["type"] == "warning" else "ℹ️"
            )
            output.append(f"  {icon} Line {issue['line']}: {issue['message']}")
        output.append("")

    return "\n".join(output)


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("No test files provided to validate.")
        return 0

    test_files = sys.argv[1:]
    validator = TestQualityValidator()

    all_valid, issues = validator.validate_files(test_files)

    print(format_issues(issues))

    if not all_valid:
        print("\n💡 Test Quality Guidelines:")
        print("- All test files should have module docstrings")
        print("- All test classes and functions should have docstrings")
        print("- Test functions should contain assertions")
        print("- Import pytest and relevant project modules")
        print("- Use type annotations for fixture functions")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
