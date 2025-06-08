"""
Generate a Markdown file with the project directory structure, with optional
exclusion of files ignored by git (.gitignore).

This script scans the project root and writes a Markdown-formatted
structure to docs/reports/project_tree.md. By default, it omits files and
directories ignored by .gitignore, but you can include all files with the
--include-ignored flag.

Usage:
    python scripts/utils/generate_project_tree.py [--include-ignored]

Arguments:
    --include-ignored, -a   Include all files, even those ignored by .gitignore

Requirements:
    - Python 3.12+
    - No external dependencies (optional: pathspec for .gitignore support)

Output:
    docs/reports/project_tree.md
"""

from collections.abc import Callable
from pathlib import Path

try:
    import pathspec
except ImportError:
    pathspec = None
    print("[WARN] 'pathspec' not installed. .gitignore will not be respected.")


def never_ignore(_: Path) -> bool:
    return False


def load_gitignore_matcher(project_root: Path) -> Callable[[Path], bool]:
    """
    Load a matcher function that returns True if a path should be ignored.
    """
    gitignore_path = project_root / ".gitignore"
    if pathspec is None or not gitignore_path.exists():
        # No pathspec or no .gitignore: never ignore
        return lambda p: False
    with open(gitignore_path, encoding="utf-8") as f:
        patterns = f.read().splitlines()
    spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)

    def is_ignored(path: Path) -> bool:
        rel_path = path.relative_to(project_root).as_posix()
        return spec.match_file(rel_path)

    return is_ignored


def build_tree(
    root: Path,
    prefix: str = "",
    is_last: bool = True,
    max_depth: int = 6,
    depth: int = 0,
    ignore_func: Callable[[Path], bool] = lambda p: False,
    project_root: Path | None = None,
) -> list[str]:
    """Recursively build the directory tree as a list of strings, excluding
    ignored files.

    Args:
        root: Root directory to scan.
        prefix: Prefix for the current line (for tree formatting).
        is_last: Whether this is the last entry in the parent directory.
        max_depth: Maximum depth to scan.
        depth: Current recursion depth.
        ignore_func: Function to determine if a path should be ignored.
        project_root: Project root for relative path calculation.

    Returns:
        List of strings representing the tree.
    """
    if depth > max_depth:
        return []
    if ignore_func(root) and (project_root is None or root != project_root):
        return []

    lines: list[str] = []
    connector = "└── " if is_last else "├── "
    lines.append(
        f"{prefix}{connector}{root.name}/"
        if root.is_dir()
        else f"{prefix}{connector}{root.name}"
    )

    if root.is_dir():
        entries = sorted(
            [e for e in root.iterdir() if not e.name.startswith(".")],
            key=lambda x: (not x.is_dir(), x.name.lower()),
        )
        # Exclude ignored entries
        entries = [e for e in entries if not ignore_func(e)]
        for i, entry in enumerate(entries):
            is_last_entry = i == len(entries) - 1
            extension = "    " if is_last else "│   "
            lines.extend(
                build_tree(
                    entry,
                    prefix + extension,
                    is_last_entry,
                    max_depth,
                    depth + 1,
                    ignore_func,
                    project_root,
                )
            )
    return lines


def parse_args() -> bool:
    """
    Parse command-line arguments. Returns True if all files should be included.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a Markdown file with the project directory structure.",  # noqa: E501
        add_help=True,
    )
    parser.add_argument(
        "--include-ignored",
        "-a",
        action="store_true",
        help="Include all files, even those ignored by .gitignore",
    )
    args = parser.parse_args()
    return args.include_ignored


def main() -> None:
    """Main entry point for the script."""
    project_root = Path(__file__).parent.parent.parent.resolve()
    output_path = project_root / "docs" / "reports" / "project_tree.md"
    cursor_output_path = (
        project_root / ".cursor" / "rules" / "project_tree.mdc"
    )

    include_ignored = parse_args()
    if include_ignored:
        ignore_func = never_ignore
        tree_title = (
            "# Project Directory Structure (all files, including "
            "`.gitignore`)\n\n"
        )
    else:
        ignore_func = load_gitignore_matcher(project_root)
        tree_title = "# Project Directory Structure (excluding .gitignore)\n\n"

    # Build the tree structure
    tree_lines = build_tree(
        project_root,
        prefix="",
        is_last=True,
        max_depth=6,
        depth=0,
        ignore_func=ignore_func,
        project_root=project_root,
    )

    # Prepare markdown content
    md_content = tree_title + "```txt\n" + "\n".join(tree_lines) + "\n```\n"

    # Prepare Cursor-specific content with additional header
    cursor_header = (
        "# Current Project Directory Structure (excluding .gitignore)\n\n"
        "> **Note**: For navigation guidance and best practices, see "
        "[project-structure.mdc](@/rules/project-structure.mdc)\n\n"
        "Can be updated with [generate_project_tree.py](mdc:scripts/utils/generate_project_tree.py)\n\n"  # noqa: E501
    )
    cursor_content = (
        cursor_header + "```txt\n" + "\n".join(tree_lines) + "\n```\n"
    )

    # Ensure output directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cursor_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write both files
    output_path.write_text(md_content, encoding="utf-8")
    cursor_output_path.write_text(cursor_content, encoding="utf-8")

    print("✅ Project structure written to:")
    print(f"   - {output_path}")
    print(f"   - {cursor_output_path}")

    if not include_ignored and pathspec is None:
        print(
            "[WARN] Install 'pathspec' for .gitignore support: pip install "
            "pathspec"
        )


if __name__ == "__main__":
    main()
