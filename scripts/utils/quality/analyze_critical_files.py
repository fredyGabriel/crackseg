#!/usr/bin/env python3
"""Analyze critical oversized files and generate refactor plan."""

import sys
from pathlib import Path


def analyze_file_structure(file_path: Path) -> dict:
    """Analyze the structure of a Python file to identify refactor opportunities."""
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        # Count classes, functions, imports
        classes = []
        functions = []
        imports = []

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("class "):
                class_name = (
                    stripped.split("class ")[1]
                    .split("(")[0]
                    .split(":")[0]
                    .strip()
                )
                classes.append((i, class_name))
            elif stripped.startswith("def ") and not stripped.startswith(
                "def __"
            ):
                func_name = stripped.split("def ")[1].split("(")[0].strip()
                functions.append((i, func_name))
            elif stripped.startswith(("import ", "from ")):
                imports.append((i, stripped))

        return {
            "total_lines": len(lines),
            "classes": classes,
            "functions": functions,
            "imports": imports,
            "non_empty_lines": len([line for line in lines if line.strip()]),
            "comment_lines": len(
                [line for line in lines if line.strip().startswith("#")]
            ),
        }
    except Exception as e:
        return {"error": str(e)}


def generate_refactor_suggestions(
    file_path: Path, analysis: dict
) -> list[str]:
    """Generate specific refactor suggestions based on file analysis."""
    suggestions = []

    if "error" in analysis:
        return [f"Error analyzing file: {analysis['error']}"]

    # Class-based suggestions
    if len(analysis["classes"]) > 3:
        suggestions.append(
            f"Split into multiple files: {len(analysis['classes'])} classes found"
        )

    # Function-based suggestions
    if len(analysis["functions"]) > 10:
        suggestions.append(
            f"Extract utility functions: {len(analysis['functions'])} functions found"
        )

    # Import-based suggestions
    if len(analysis["imports"]) > 20:
        suggestions.append(
            f"Consolidate imports: {len(analysis['imports'])} import statements"
        )

    # Size-based suggestions
    if analysis["total_lines"] > 600:
        suggestions.append(
            "Critical: File exceeds 600 lines - immediate refactor needed"
        )
    elif analysis["total_lines"] > 500:
        suggestions.append(
            "High priority: File exceeds 500 lines - refactor soon"
        )

    return suggestions


def main():
    project_root = Path(".").resolve()
    src = project_root / "src"

    # Get critical files (>500 lines)
    critical_files = []

    for py_file in src.rglob("*.py"):
        try:
            with py_file.open("r", encoding="utf-8", errors="ignore") as f:
                line_count = sum(1 for _ in f)

            if line_count > 500:
                rel_path = py_file.relative_to(project_root)
                critical_files.append((line_count, rel_path))
        except Exception:
            continue

    critical_files.sort(reverse=True)

    print("🔍 CRITICAL FILES ANALYSIS (>500 lines)")
    print("=" * 60)

    for line_count, file_path in critical_files:
        print(f"\n📁 {file_path} ({line_count} lines)")
        print("-" * 40)

        analysis = analyze_file_structure(project_root / file_path)
        suggestions = generate_refactor_suggestions(file_path, analysis)

        if "error" not in analysis:
            print(f"  Classes: {len(analysis['classes'])}")
            print(f"  Functions: {len(analysis['functions'])}")
            print(f"  Imports: {len(analysis['imports'])}")
            print(f"  Non-empty lines: {analysis['non_empty_lines']}")

        print("  💡 Suggestions:")
        for suggestion in suggestions:
            print(f"    • {suggestion}")

    print(
        f"\n🎯 SUMMARY: {len(critical_files)} critical files need immediate attention"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
