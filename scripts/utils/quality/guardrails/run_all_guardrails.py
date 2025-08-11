"""
Run all CI guardrails locally:
- Line limit check
- Hydra smoke test
- Regenerate analysis reports (dependency graph, hydra-registry, oversized)

Exits non-zero if any guardrail fails.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def call(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def main() -> int:
    rc = 0
    # 1) Line limit
    rc = (
        call(
            [
                sys.executable,
                str(
                    PROJECT_ROOT
                    / "scripts"
                    / "utils"
                    / "quality"
                    / "guardrails"
                    / "line_limit_check.py"
                ),
            ]
        )
        or rc
    )
    # 2) Hydra smoke
    rc = (
        call(
            [
                sys.executable,
                str(
                    PROJECT_ROOT
                    / "scripts"
                    / "utils"
                    / "quality"
                    / "guardrails"
                    / "hydra_smoke.py"
                ),
            ]
        )
        or rc
    )
    # 3) Analysis reports
    rc = (
        call(
            [
                sys.executable,
                str(
                    PROJECT_ROOT
                    / "scripts"
                    / "utils"
                    / "maintenance"
                    / "regenerate_analysis_reports.py"
                ),
            ]
        )
        or rc
    )
    # 4) Layering rules
    rc = (
        call(
            [
                sys.executable,
                str(
                    PROJECT_ROOT
                    / "scripts"
                    / "utils"
                    / "quality"
                    / "guardrails"
                    / "layering_rules_check.py"
                ),
            ]
        )
        or rc
    )
    # 5) Link checker (docs and infrastructure)
    rc = (
        call(
            [
                sys.executable,
                str(
                    PROJECT_ROOT
                    / "scripts"
                    / "utils"
                    / "quality"
                    / "guardrails"
                    / "link_checker.py"
                ),
                "--directories",
                "docs",
                "infrastructure",
            ]
        )
        or rc
    )
    if rc:
        print("Guardrails failed.")
        return 1
    print("Guardrails passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
