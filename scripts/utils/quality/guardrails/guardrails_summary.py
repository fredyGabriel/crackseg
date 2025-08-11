"""
Aggregate guardrails effectiveness into machine-readable and human-readable summaries.

Outputs:
 - docs/reports/analysis-reports/architecture/guardrails_validation_summary.json
 - docs/reports/analysis-reports/architecture/guardrails_validation_summary.md
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Allow importing project root so "scripts" is a package root
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E402

from scripts.utils.analysis import (  # noqa: E402
    scan_oversized_modules as som,
)
from scripts.utils.quality.guardrails import (  # noqa: E402
    line_limit_check as llc,
)
from scripts.utils.quality.guardrails import (  # noqa: E402
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


def compute_duplicate_counts() -> dict[str, Any]:
    """Run duplicate scan and compare against baseline to summarize status."""
    scan_script = PROJECT_ROOT / "scripts" / "reports" / "duplicate_scan.py"
    reports_dir = PROJECT_ROOT / "docs" / "reports" / "project-reports"
    current_json = reports_dir / "duplicate_scan_report.json"
    baseline_json = reports_dir / "duplicate_scan_baseline.json"

    # Refresh current scan (ignore failures to keep summary resilient)
    try:
        subprocess.run([sys.executable, str(scan_script)], check=True)
    except Exception:
        pass

    from scripts.utils.common.io_utils import read_json  # noqa: E402

    current_groups = 0
    baseline_groups = 0
    new_groups = 0

    try:
        current = read_json(current_json)
        current_hashes = {
            str(item.get("hash", "")) for item in current if "hash" in item
        }
        current_groups = len(current_hashes)
    except Exception:
        current_hashes = set()

    try:
        baseline = read_json(baseline_json)
        baseline_hashes = {
            str(item.get("hash", "")) for item in baseline if "hash" in item
        }
        baseline_groups = len(baseline_hashes)
    except Exception:
        baseline_hashes = set()

    new_groups = len(current_hashes - baseline_hashes)

    return {
        "current": current_groups,
        "baseline": baseline_groups,
        "new": new_groups,
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
    lines.append("## Duplicate Code")
    dc = summary["duplicates"]
    lines.append(
        f"Groups: {dc['current']} | Baseline: {dc['baseline']} | New vs. baseline: {dc['new']}"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    ll_counts, ll_top = compute_line_limit_counts()
    ov_counts, ov_top = compute_oversized_counts()
    link_counts = compute_link_counts()
    dup_counts = compute_duplicate_counts()

    overall_status = "pass"
    if ll_counts.critical > 0 or link_counts["errors"] > 0:
        overall_status = "fail"

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
        "duplicates": dup_counts,
    }

    from scripts.utils.common.io_utils import (  # noqa: E402
        write_json,
        write_text,
    )

    write_json(JSON_PATH, summary, indent=2)
    write_text(MD_PATH, render_md(summary))
    print(f"Wrote summary to: {JSON_PATH}\n{MD_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
