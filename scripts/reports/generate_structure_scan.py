"""Generate structure scan summaries for key project roots.

Outputs JSON reports under docs/reports/project-reports/ with lists of objects
{"path": <dir>, "py_files": <count>} for directories containing Python files.

Generated files:
- structure_scan_src.json
- structure_scan_gui.json (if exists)
- structure_scan_scripts.json (if exists)
- structure_scan_configs.json (if exists)
- structure_scan_tests.json (if exists)
- structure_scan_overview.json (per-root totals)

For backward compatibility, also writes structure_scan_summary.json for src/.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

from scripts.utils.common.io_utils import (  # noqa: E402
    read_text,
    write_json,
    write_text,
)


class DirReport(TypedDict):
    path: str
    py_files: int


class RootOverview(TypedDict):
    root: str
    directories_with_py: int
    total_py_files: int


def generate_structure_summary(root: Path) -> list[DirReport]:
    root = root.resolve()
    report: list[DirReport] = []

    for base, _dirs, files in os.walk(root):
        py_files = [f for f in files if f.endswith(".py")]
        if not py_files:
            continue
        rel = Path(base).relative_to(Path.cwd())
        report.append(
            DirReport(path=str(rel).replace("\\", "/"), py_files=len(py_files))
        )

    report.sort(key=lambda x: x["path"])  # type: ignore[index]
    return report


if __name__ == "__main__":
    out_dir = Path("docs/reports/project-reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    roots: dict[str, Path] = {
        "src": Path("src"),
        "gui": Path("gui"),
        "scripts": Path("scripts"),
        "configs": Path("configs"),
        "tests": Path("tests"),
    }

    overview: list[RootOverview] = []

    for name, path in roots.items():
        if not path.exists():
            continue
        report = generate_structure_summary(path)
        write_json(out_dir / f"structure_scan_{name}.json", report, indent=2)

        total_py_files: int = sum(item["py_files"] for item in report)
        overview.append(
            RootOverview(
                root=name,
                directories_with_py=len(report),
                total_py_files=total_py_files,
            )
        )

    write_json(out_dir / "structure_scan_overview.json", overview, indent=2)

    # Back-compat summary for src
    src_json = out_dir / "structure_scan_src.json"
    if src_json.exists():
        write_text(
            out_dir / "structure_scan_summary.json",
            read_text(src_json),
        )

    print("OK")
