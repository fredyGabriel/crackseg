"""
Scan Python modules for oversize (line count) and produce a refactor-priority report.

Generates: docs/reports/analysis-reports/architecture/oversized_modules_report.md
Rules: preferred <= 300 lines; hard max 400 lines (coding-standards).
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E402

from scripts.utils.common.io_utils import (  # noqa: E402
    read_text,
    write_text,
)

PROJECT_ROOT = PROJECT_ROOT
SRC_ROOT = PROJECT_ROOT / "src"
REPORT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "reports"
    / "analysis-reports"
    / "architecture"
    / "oversized_modules_report.md"
)


PREFERRED_MAX = 300
HARD_MAX = 400


@dataclass
class ModuleStat:
    path: Path
    lines: int
    package: str

    @property
    def severity(self) -> str:
        if self.lines > HARD_MAX:
            return "critical"
        if self.lines > PREFERRED_MAX:
            return "warning"
        return "ok"


def iter_py_files(root: Path) -> Iterable[Path]:
    yield from root.rglob("*.py")


def dotted_package(p: Path) -> str:
    try:
        rel = p.relative_to(SRC_ROOT)
    except ValueError:
        return ""
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    elif parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    return ".".join(parts)


def count_lines(p: Path) -> int:
    try:
        return len(read_text(p).splitlines())
    except Exception:
        return 0


def collect_stats() -> list[ModuleStat]:
    stats: list[ModuleStat] = []
    for p in iter_py_files(SRC_ROOT):
        lines = count_lines(p)
        stats.append(
            ModuleStat(path=p, lines=lines, package=dotted_package(p))
        )
    return stats


def render_report(stats: list[ModuleStat]) -> str:
    oversized = [s for s in stats if s.lines > PREFERRED_MAX]
    oversized.sort(key=lambda s: s.lines, reverse=True)

    lines: list[str] = []
    lines.append("<!-- markdownlint-disable-file -->")
    lines.append("# Oversized Modules Report")
    lines.append("")
    lines.append(
        f"Preferred max: {PREFERRED_MAX} lines | Hard max: {HARD_MAX} lines"
    )
    lines.append("")
    if not oversized:
        lines.append("No modules exceed preferred limit.")
        return "\n".join(lines)

    lines.append("Module | Lines | Severity")
    lines.append(":-- | --: | :--:")
    for s in oversized[:200]:
        lines.append(f"`{s.package}` | {s.lines} | {s.severity}")

    lines.append("")
    lines.append("## Top 10 by size")
    lines.append("")
    for s in oversized[:10]:
        lines.append(f"- `{s.package}` — {s.lines} lines ({s.severity})")

    return "\n".join(lines)


def main() -> None:
    stats = collect_stats()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_text(REPORT_PATH, render_report(stats))


if __name__ == "__main__":
    main()
