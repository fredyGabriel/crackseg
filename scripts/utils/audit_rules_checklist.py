"""
Audit all .mdc rule files in .cursor/rules/ against the rules_checklist.mdc.

This script checks for key checklist items (header, structure, references,
examples, update notes)
and prints a summary report to the console.

Usage:
    python scripts/utils/audit_rules_checklist.py

Requirements:
    - Python 3.12+
    - No external dependencies
"""

import re
from collections.abc import Iterator
from pathlib import Path

RULES_DIR = Path(".cursor/rules")
CHECKLIST_FILE = RULES_DIR / "rules_checklist.mdc"

# Checklist items to verify (simplified for automation)
CHECKLIST_ITEMS = [
    (
        "Header with description, globs, alwaysApply",
        re.compile(r"^---[\s\S]+?---", re.MULTILINE),
    ),
    (
        "Section organization (introduction, main points, examples, references)",  # noqa: E501
        re.compile(r"# .+\n", re.MULTILINE),
    ),
    ("References to related rules", re.compile(r"\[.+\.mdc\]\(mdc:.+\.mdc\)")),
    ("References to project guides or files", re.compile(r"\[.+\]\(mdc:.+\)")),
    ("Concrete code examples", re.compile(r"```[a-zA-Z]*[\s\S]+?```")),
    (
        "Version history or update notes",
        re.compile(r"version|update|history|changelog", re.IGNORECASE),
    ),
]


def iter_rule_files(rules_dir: Path) -> Iterator[Path]:
    for file in rules_dir.glob("*.mdc"):
        if file.name != "rules_checklist.mdc":
            yield file


def audit_rule_file(path: Path) -> dict[str, bool]:
    text = path.read_text(encoding="utf-8")
    results = {}
    for label, pattern in CHECKLIST_ITEMS:
        results[label] = bool(pattern.search(text))
    return results


def main() -> None:
    print("\nRule Audit Report (against rules_checklist.mdc):\n")
    for rule_file in iter_rule_files(RULES_DIR):
        results = audit_rule_file(rule_file)
        print(f"- {rule_file.name}")
        for label, passed in results.items():
            mark = "✅" if passed else "❌"
            print(f"    {mark} {label}")
        print()
    print("Audit complete. Review ❌ items for improvement.")


if __name__ == "__main__":
    main()
