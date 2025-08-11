"""
Guardrail: prevent introduction of new duplicate code groups.

This guardrail runs the duplicate scanner, compares the result against a
baseline, and fails if new duplicate groups (by normalized hash) are detected
beyond an allowed delta.

Outputs/reads:
- Runs scripts/reports/duplicate_scan.py to refresh the current report
- Reads current: docs/reports/project-reports/duplicate_scan_report.json
- Baseline default: docs/reports/project-reports/duplicate_scan_baseline.json
- Writes status: docs/reports/project-reports/duplicate_guardrail_status.md
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E402

from scripts.utils.common.io_utils import (  # noqa: E402
    read_json,
    write_json,
    write_text,
)

REPORTS_DIR = PROJECT_ROOT / "docs" / "reports" / "project-reports"
SCAN_JSON = REPORTS_DIR / "duplicate_scan_report.json"
BASELINE_JSON = REPORTS_DIR / "duplicate_scan_baseline.json"
STATUS_MD = REPORTS_DIR / "duplicate_guardrail_status.md"


def run_duplicate_scan() -> None:
    scan_script = PROJECT_ROOT / "scripts" / "reports" / "duplicate_scan.py"
    subprocess.run([sys.executable, str(scan_script)], check=True)


def load_group_hashes(path: Path) -> set[str]:
    data: list[dict[str, Any]] = read_json(path)
    return {str(item.get("hash", "")) for item in data if "hash" in item}


def build_status_md(
    new_count: int, total_current: int, added: list[str]
) -> str:
    lines: list[str] = []
    lines.append("# Duplicate Guardrail Status\n")
    lines.append(f"Current duplicate groups: {total_current}\n\n")
    if new_count == 0:
        lines.append("✅ No new duplicate groups introduced.\n")
    else:
        lines.append(f"❌ New duplicate groups introduced: {new_count}\n\n")
        lines.append("New group hashes:\n\n")
        for h in added[:50]:
            lines.append(f"- `{h}`\n")
        if len(added) > 50:
            lines.append("\n… truncated …\n")
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Guardrail to prevent new duplicate code groups"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=str(BASELINE_JSON),
        help="Path to baseline JSON file",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Update baseline to current scan and exit successfully",
    )
    parser.add_argument(
        "--max-delta",
        type=int,
        default=0,
        help="Allow up to N new duplicate groups before failing",
    )

    args = parser.parse_args()
    baseline_path = Path(args.baseline)

    # 1) Run the scanner to refresh reports
    run_duplicate_scan()

    # 2) Load current scan results
    if not SCAN_JSON.exists():
        print(f"Scan output not found: {SCAN_JSON}", file=sys.stderr)
        return 2
    current: list[dict[str, Any]] = read_json(SCAN_JSON)
    current_hashes = {
        str(item.get("hash", "")) for item in current if "hash" in item
    }

    # 3) Initialize or update baseline if requested
    if args.update_baseline or not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(baseline_path, current, indent=2)
        status_md = build_status_md(0, len(current_hashes), [])
        STATUS_MD.parent.mkdir(parents=True, exist_ok=True)
        write_text(STATUS_MD, status_md)
        if args.update_baseline:
            print(f"Baseline updated: {baseline_path}")
        else:
            print(f"Baseline created: {baseline_path}")
        return 0

    # 4) Compare with baseline
    baseline_hashes = load_group_hashes(baseline_path)
    added = sorted(current_hashes - baseline_hashes)
    new_count = len(added)

    # 5) Write status report
    status_md = build_status_md(new_count, len(current_hashes), added)
    STATUS_MD.parent.mkdir(parents=True, exist_ok=True)
    write_text(STATUS_MD, status_md)

    # 6) Enforce delta
    if new_count > args.max_delta:
        print(
            f"Duplicate guardrail failed: {new_count} new groups (max allowed {args.max_delta}).",
            file=sys.stderr,
        )
        print(f"See: {STATUS_MD}")
        return 1

    print(
        f"Duplicate guardrail passed: {new_count} new groups (max allowed {args.max_delta})."
    )
    print(f"See: {STATUS_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
