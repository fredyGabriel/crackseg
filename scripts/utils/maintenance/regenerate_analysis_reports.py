"""
Regenerate all analysis reports affected by refactors/migration.

Runs: dependency graph, hydra-registry alignment, oversized modules, line-limit guardrail.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    cmds = [
        [
            sys.executable,
            str(
                PROJECT_ROOT
                / "scripts"
                / "utils"
                / "analysis"
                / "generate_dependency_graph.py"
            ),
        ],
        [
            sys.executable,
            str(
                PROJECT_ROOT
                / "scripts"
                / "utils"
                / "analysis"
                / "validate_hydra_registry_alignment.py"
            ),
        ],
        [
            sys.executable,
            str(
                PROJECT_ROOT
                / "scripts"
                / "utils"
                / "analysis"
                / "scan_oversized_modules.py"
            ),
        ],
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
        ],
    ]
    rc = 0
    for cmd in cmds:
        rc = run(cmd) or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
