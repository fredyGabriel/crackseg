"""
Hydra smoke test guardrail.

Runs a minimal configuration through the main `run.py` entrypoint to verify
that the configuration system and model assembly are functional. Produces a
markdown report artifact and a non-zero exit status on failure.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from scripts.utils.common.io_utils import write_text  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUNNER = PROJECT_ROOT / "run.py"
REPORT = (
    PROJECT_ROOT
    / "docs"
    / "reports"
    / "analysis-reports"
    / "architecture"
    / "hydra_smoke_report.md"
)


def run_smoke() -> int:
    start = time.time()
    cmd = [
        sys.executable,
        str(RUNNER),
        "--config-name=basic_verification",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    write_text(
        REPORT,
        "\n".join(
            [
                "<!-- markdownlint-disable-file -->",
                "# Hydra Smoke Test",
                "",
                f"- Command: `{cmd}`",
                f"- CWD: `{PROJECT_ROOT}`",
                f"- Exit code: {proc.returncode}",
                f"- Elapsed: {elapsed:.2f}s",
                "",
                "## Stdout",
                "",
                "```text",
                proc.stdout.strip(),
                "```",
                "",
                "## Stderr",
                "",
                "```text",
                proc.stderr.strip(),
                "```",
            ]
        ),
    )
    return int(proc.returncode != 0)


def main() -> int:
    return run_smoke()


if __name__ == "__main__":
    raise SystemExit(main())
