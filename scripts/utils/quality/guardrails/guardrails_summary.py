"""
Aggregate guardrails effectiveness into machine-readable and human-readable summaries.

Outputs:
 - docs/reports/analysis-reports/architecture/guardrails_validation_summary.json
 - docs/reports/analysis-reports/architecture/guardrails_validation_summary.md
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Allow importing from scripts/ as a package root
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.analysis import (  # type: ignore  # noqa: E402
    scan_oversized_modules as som,
)
from utils.quality.guardrails import (  # type: ignore  # noqa: E402
    line_limit_check as llc,
)
from utils.quality.guardrails import (  # type: ignore  # noqa: E402
    link_checker as lkc,
)

REPORT_DIR = (
    PROJECT_ROOT / "docs" / "reports" / "analysis-reports" / "architecture"
)
JSON_PATH = REPORT_DIR / "guardrails_validation_summary.json"
MD_PATH = REPORT_DIR / "guardrails_validation_summary.md"


@dataclass
class GuardrailCounts:
    critical: int
    warning: int
    ok: int


def compute_line_limit_counts() -> (
    tuple[GuardrailCounts, list[tuple[str, int]]]
):
    stats = llc.collect_stats()
    crit = [s for s in stats if s.severity == "critical"]
    warn = [s for s in stats if s.severity == "warning"]
    ok = len(stats) - len(crit) - len(warn)

    # Top offenders by lines
    top = sorted(stats, key=lambda s: s.lines, reverse=True)[:10]
    top_list = [(str(s.path.relative_to(PROJECT_ROOT)), s.lines) for s in top]
    return GuardrailCounts(len(crit), len(warn), ok), top_list


def compute_oversized_counts() -> (
    tuple[GuardrailCounts, list[tuple[str, int]]]
):
    stats = som.collect_stats()
    crit = [s for s in stats if s.lines > som.HARD_MAX]
    warn = [s for s in stats if som.PREFERRED_MAX < s.lines <= som.HARD_MAX]
    ok = len(stats) - len(crit) - len(warn)
    # Top by size
    top = sorted(stats, key=lambda s: s.lines, reverse=True)[:10]
    top_list = [(s.package, s.lines) for s in top]
    return GuardrailCounts(len(crit), len(warn), ok), top_list


def compute_link_counts() -> dict[str, Any]:
    registry = lkc.create_default_registry()
    docs_issues = lkc.check_directory_links(PROJECT_ROOT / "docs", registry)
    infra_issues = lkc.check_directory_links(
        PROJECT_ROOT / "infrastructure", registry
    )
    all_issues = docs_issues + infra_issues
    errors = [i for i in all_issues if i.get("severity") == "error"]
    warnings = [i for i in all_issues if i.get("severity") == "warning"]
    return {
        "errors": len(errors),
        "warnings": len(warnings),
    }


def render_md(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("<!-- markdownlint-disable-file -->")
    lines.append("# Guardrails Validation Summary")
    lines.append("")
    lines.append(f"Overall status: **{summary['overall_status'].upper()}**")
    lines.append("")
    lines.append("## Line Limit")
    ll = summary["line_limit"]
    lines.append(
        f"Critical: {ll['counts']['critical']} | Warnings: {ll['counts']['warning']} | OK: {ll['counts']['ok']}"
    )
    lines.append("")
    lines.append("Top offenders:")
    for path, n in summary["line_limit"]["top"][:10]:
        lines.append(f"- `{path}` — {n} lines")
    lines.append("")
    lines.append("## Oversized Modules")
    ov = summary["oversized"]
    lines.append(
        f"Critical: {ov['counts']['critical']} | Warnings: {ov['counts']['warning']} | OK: {ov['counts']['ok']}"
    )
    lines.append("")
    lines.append("Top offenders:")
    for pkg, n in summary["oversized"]["top"][:10]:
        lines.append(f"- `{pkg}` — {n} lines")
    lines.append("")
    lines.append("## Link Checker")
    lc = summary["link_checker"]
    lines.append(f"Errors: {lc['errors']} | Warnings: {lc['warnings']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    ll_counts, ll_top = compute_line_limit_counts()
    ov_counts, ov_top = compute_oversized_counts()
    link_counts = compute_link_counts()

    overall_status = (
        "pass"
        if (ll_counts.critical == 0 and link_counts["errors"] == 0)
        else "fail"
    )

    summary: dict[str, Any] = {
        "overall_status": overall_status,
        "line_limit": {
            "counts": asdict(ll_counts),
            "top": ll_top,
        },
        "oversized": {
            "counts": asdict(ov_counts),
            "top": ov_top,
        },
        "link_checker": link_counts,
    }

    JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    MD_PATH.write_text(render_md(summary), encoding="utf-8")
    print(f"Wrote summary to: {JSON_PATH}\n{MD_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
