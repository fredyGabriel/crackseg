#!/usr/bin/env python3
"""List Python files exceeding 400 lines for refactoring prioritization."""

import sys
from pathlib import Path


def main():
    project_root = Path(".").resolve()
    src = project_root / "src"

    oversized_files = []

    for py_file in src.rglob("*.py"):
        try:
            with py_file.open("r", encoding="utf-8", errors="ignore") as f:
                line_count = sum(1 for _ in f)

            if line_count > 400:
                rel_path = py_file.relative_to(project_root)
                oversized_files.append((line_count, str(rel_path)))
        except Exception as e:
            print(f"Error reading {py_file}: {e}", file=sys.stderr)

    # Sort by line count descending
    oversized_files.sort(reverse=True)

    print(f"Found {len(oversized_files)} files exceeding 400 lines:")
    print("-" * 60)
    print(f"{'Lines':<6} {'File'}")
    print("-" * 60)

    for line_count, file_path in oversized_files:
        print(f"{line_count:<6} {file_path}")

    return 0 if not oversized_files else 1


if __name__ == "__main__":
    sys.exit(main())
