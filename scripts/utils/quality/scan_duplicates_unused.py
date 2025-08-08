"""
Detect duplicate files and unused Python modules.

Outputs a markdown report at:
  docs/reports/project-reports/technical/duplicates_and_unused_report.md

Features:
- Respects .gitignore via pathspec
- Duplicate detection: group by size, then sha1 hash
- Unused modules: modules under src/ not imported anywhere (heuristic)

Notes:
- Dynamic imports and tooling may yield false positives. Review before action.
- Excludes heavy data/artifacts directories to focus on source footprint.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path

try:
    import pathspec  # type: ignore
except Exception:  # pragma: no cover
    pathspec = None  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "reports"
    / "project-reports"
    / "technical"
    / "duplicates_and_unused_report.md"
)

INCLUDE_DIRS = [
    "configs",
    "docs",
    "gui",
    "infrastructure",
    "scripts",
    "src",
    "tests",
]
HARD_EXCLUDES = {
    "artifacts",
    "data",
    ".git",
    ".cursor",
    ".taskmaster",
    "__pycache__",
}

MAX_HASHED_FILES = 5000


def load_gitignore_matcher(project_root: Path):
    if pathspec is None:
        return lambda _p: False
    gi = project_root / ".gitignore"
    if not gi.exists():
        return lambda _p: False
    spec = pathspec.PathSpec.from_lines(
        "gitwildmatch", gi.read_text(encoding="utf-8").splitlines()
    )

    def is_ignored(p: Path) -> bool:
        try:
            rel = p.relative_to(project_root).as_posix()
        except ValueError:
            return False
        return spec.match_file(rel)

    return is_ignored


def iter_files(root: Path, ignore) -> list[Path]:
    results: list[Path] = []
    for base_rel in INCLUDE_DIRS:
        base = root / base_rel
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if any(part in HARD_EXCLUDES for part in p.parts):
                continue
            if ignore(p):
                continue
            if p.is_file():
                results.append(p)
    return results


def detect_duplicate_groups(files: list[Path]) -> list[list[Path]]:
    by_size: dict[int, list[Path]] = defaultdict(list)
    for p in files:
        try:
            size = p.stat().st_size
        except OSError:
            continue
        by_size[size].append(p)

    dup_groups: list[list[Path]] = []
    candidates = [g for g in by_size.values() if len(g) > 1]
    # Heuristic cap to avoid hashing too many files
    total = sum(len(g) for g in candidates)
    if total > MAX_HASHED_FILES:
        candidates.sort(key=len)
    processed = 0

    for group in candidates:
        if processed >= MAX_HASHED_FILES:
            break
        hash_to_paths: dict[str, list[Path]] = defaultdict(list)
        for p in group:
            if processed >= MAX_HASHED_FILES:
                break
            try:
                h = sha1()
                with p.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                        h.update(chunk)
                digest = h.hexdigest()
                hash_to_paths[digest].append(p)
            except OSError:
                continue
            finally:
                processed += 1
        for paths in hash_to_paths.values():
            if len(paths) > 1:
                dup_groups.append(paths)
    return dup_groups


IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))", re.MULTILINE
)


def dotted_module_from_src_path(p: Path) -> str | None:
    try:
        rel = p.relative_to(PROJECT_ROOT / "src")
    except ValueError:
        return None
    if p.suffix != ".py":
        return None
    if p.name in {"__init__.py", "__main__.py"}:
        return None
    parts = list(rel.parts)
    parts[-1] = parts[-1][:-3]  # strip .py
    return ".".join([*parts]).replace("/", ".")


def collect_import_usage(py_files: list[Path]) -> set[str]:
    used: set[str] = set()
    for p in py_files:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in IMPORT_RE.finditer(text):
            pkg = m.group(1) or m.group(2)
            if not pkg:
                continue
            used.add(pkg)
            # Also add parent packages to reduce false positives
            while "." in pkg:
                pkg = pkg.rsplit(".", 1)[0]
                used.add(pkg)
    return used


def find_unused_modules(files: list[Path]) -> list[Path]:
    py_files = [p for p in files if p.suffix == ".py"]
    used = collect_import_usage(py_files)
    unused: list[Path] = []
    for p in py_files:
        dotted = dotted_module_from_src_path(p)
        if dotted is None:
            continue
        # Consider module used if any used import matches module or parent
        if dotted in used:
            continue
        # Also if parent package appears, give benefit of doubt
        parent = dotted.rsplit(".", 1)[0] if "." in dotted else None
        if parent and parent in used:
            continue
        unused.append(p)
    return unused


def build_report(dups: list[list[Path]], unused: list[Path]) -> str:
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S %Z")
    lines: list[str] = []
    lines.append("<!-- markdownlint-disable-file -->")
    lines.append(f"# Duplicates and unused modules report — {ts}")
    lines.append("")
    lines.append("Heuristic scan; please review before deletion/refactor.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"Duplicate groups: {len(dups)}")
    lines.append(f"Unused modules under src/: {len(unused)}")
    lines.append("")

    lines.append("## Potential duplicate groups (size+hash)")
    lines.append("")
    if not dups:
        lines.append("None detected.")
    else:
        for idx, group in enumerate(
            sorted(dups, key=lambda g: len(g), reverse=True), start=1
        ):
            lines.append(f"- Group {idx}:")
            for p in group:
                rel = p.relative_to(PROJECT_ROOT).as_posix()
                lines.append(f"  - `{rel}`")
        lines.append("")

    lines.append("## Unused Python modules under src/")
    lines.append("")
    if not unused:
        lines.append("None detected.")
    else:
        lines.append("Path | Module")
        lines.append(":-- | :--")
        for p in sorted(unused, key=lambda x: x.as_posix()):
            rel = p.relative_to(PROJECT_ROOT).as_posix()
            dotted = dotted_module_from_src_path(p) or "?"
            lines.append(f"`{rel}` | `{dotted}`")
        lines.append("")

    lines.append("## Recommended next steps")
    lines.append("")
    lines.append(
        "- For duplicates: consolidate into a single canonical location; update references"
    )
    lines.append(
        "- For unused modules: verify via grep/tests; delete or integrate as needed"
    )
    lines.append(
        "- Add guard checks into CI to prevent re-introduction of duplicates"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ignore = load_gitignore_matcher(PROJECT_ROOT)
    files = iter_files(PROJECT_ROOT, ignore)
    dups = detect_duplicate_groups(files)
    unused = find_unused_modules(files)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(dups, unused), encoding="utf-8")


if __name__ == "__main__":
    main()
