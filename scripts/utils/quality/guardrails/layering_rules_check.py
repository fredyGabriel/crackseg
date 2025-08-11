"""Fail on import boundary violations and bypassing public shims.

Rules:
- Forbid imports from `crackseg.training` inside `crackseg.model`.
- Forbid direct imports of internal storage/reporting when public shims exist.
  - Use `crackseg.utils.storage` instead of `crackseg.utils.checkpointing.*` or `crackseg.utils.config.standardized_storage`.
  - Use `crackseg.utils.reporting` instead of
    `crackseg.evaluation.utils.results`, `crackseg.utils.experiment_saver`,
    or `crackseg.utils.deployment.validation.reporting.core`.
"""

from __future__ import annotations

import re
from pathlib import Path

from scripts.utils.common.io_utils import read_text  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src" / "crackseg"


FORBIDDEN_IMPORTS: list[tuple[re.Pattern[str], str]] = [
    # model -> training
    (
        re.compile(
            r"^from\s+crackseg\.training\b|^import\s+crackseg\.training\b"
        ),
        "crackseg.model",
    ),
]

BYPASS_IMPORTS: list[tuple[re.Pattern[str], str]] = [
    # storage internals
    (
        re.compile(r"crackseg\.utils\.checkpointing\b"),
        "Use crackseg.utils.storage",
    ),
    (
        re.compile(r"crackseg\.utils\.config\.standardized_storage\b"),
        "Use crackseg.utils.storage",
    ),
    # reporting internals
    (
        re.compile(r"crackseg\.utils\.experiment_saver\b"),
        "Use crackseg.utils.reporting",
    ),
    (
        re.compile(r"crackseg\.evaluation\.utils\.results\b"),
        "Use crackseg.utils.reporting",
    ),
    (
        re.compile(
            r"crackseg\.utils\.deployment\.validation\.reporting\.core\b"
        ),
        "Use crackseg.utils.reporting",
    ),
]

# Allowlist files that are the public shims themselves
ALLOWED_BYPASS_FILES = {
    Path("utils/reporting/__init__.py"),
    Path("utils/storage/__init__.py"),
}


def file_text(p: Path) -> str:
    try:
        return read_text(p)
    except Exception:
        return ""


def is_under(package: str, file: Path) -> bool:
    # e.g., package="crackseg.model" → path contains /src/crackseg/model/
    parts = package.split(".")
    return SRC_ROOT.joinpath(*parts[1:]) in file.parents


def main() -> int:
    violations: list[str] = []

    for py in SRC_ROOT.rglob("*.py"):
        text = file_text(py)
        if not text:
            continue

        # Boundary violations
        for pattern, scope in FORBIDDEN_IMPORTS:
            if is_under(scope, py) and pattern.search(text):
                violations.append(f"{py}: forbidden import in scope '{scope}'")

        # Shim bypasses anywhere
        for pattern, hint in BYPASS_IMPORTS:
            if pattern.search(text):
                rel = py.relative_to(SRC_ROOT)
                if rel not in ALLOWED_BYPASS_FILES:
                    violations.append(f"{py}: bypassing public shim ({hint})")

    if violations:
        print("Layering rules violations:")
        for v in violations:
            print(" -", v)
        return 1
    print("Layering rules check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
