"""
Guardrail: enforce per-file line limits.

Preferred <= 300 lines; hard max 400 lines. Exits non-zero if any file exceeds 400.
Writes a markdown report to docs/reports/analysis-reports/architecture/line_limit_guardrail.md
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
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
    / "line_limit_guardrail.md"
)

PREFERRED_MAX = 300
HARD_MAX = 400


@dataclass
class FileStat:
    path: Path
    lines: int

    @property
    def severity(self) -> str:
        if self.lines > HARD_MAX:
            return "critical"
        if self.lines > PREFERRED_MAX:
            return "warning"
        return "ok"


def count_lines(p: Path) -> int:
    try:
        return len(read_text(p).splitlines())
    except Exception:
        return 0


def iter_py_files(root: Path):
    yield from root.rglob("*.py")


def collect_stats() -> list[FileStat]:
    stats: list[FileStat] = []
    for p in iter_py_files(SRC_ROOT):
        stats.append(FileStat(path=p, lines=count_lines(p)))
    return stats


def render_report(stats: list[FileStat]) -> str:
    warnings = [s for s in stats if s.severity == "warning"]
    criticals = [s for s in stats if s.severity == "critical"]
    warnings.sort(key=lambda s: s.lines, reverse=True)
    criticals.sort(key=lambda s: s.lines, reverse=True)

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(p)

    lines: list[str] = []
    lines.append("<!-- markdownlint-disable-file -->")
    lines.append("# Line Limit Guardrail Report")
    lines.append("")
    lines.append(f"Preferred max: {PREFERRED_MAX} | Hard max: {HARD_MAX}")
    lines.append("")
    lines.append("## Critical (> hard max)")
    lines.append("")
    if not criticals:
        lines.append("None")
    else:
        lines.append("File | Lines")
        lines.append(":-- | --:")
        for s in criticals:
            lines.append(f"`{rel(s.path)}` | {s.lines}")
    lines.append("")

    lines.append("## Warnings (> preferred)")
    lines.append("")
    if not warnings:
        lines.append("None")
    else:
        lines.append("File | Lines")
        lines.append(":-- | --:")
        for s in warnings[:200]:
            lines.append(f"`{rel(s.path)}` | {s.lines}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    stats = collect_stats()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_text(REPORT_PATH, render_report(stats))
    has_crit = any(s.severity == "critical" for s in stats)
    if has_crit:
        print(
            "Line limit guardrail failed: files exceed hard max.",
            file=sys.stderr,
        )
        return 1
    print("Line limit guardrail passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
