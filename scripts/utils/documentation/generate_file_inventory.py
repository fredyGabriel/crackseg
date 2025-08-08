"""
Generate repository file inventory details and append a report to
docs/reports/file_inventory.md.

Features:
- Respects .gitignore via pathspec when available
- Scans selected directories (code and docs) to keep runtime reasonable
- Produces extension distribution (count, total size)
- Lists top-N largest files and flags > 50 MB
- Detects potential duplicates by first grouping by size, then hashing

Usage:
    python scripts/utils/documentation/generate_file_inventory.py

Notes:
- Data-heavy folders (e.g., data/, artifacts/) are excluded deliberately
  from duplicate checks to focus on source/docs footprint.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path

try:
    import pathspec  # type: ignore
except Exception:  # pragma: no cover
    pathspec = None  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORT_PATH = PROJECT_ROOT / "docs" / "reports" / "file_inventory.md"

INCLUDE_DIRS = [
    "configs",
    "docs",
    "gui",
    "infrastructure",
    "scripts",
    "src",
    "tests",
]

# Exclude these directories entirely from scanning
HARD_EXCLUDES = {
    "artifacts",
    "data",
    ".git",
    ".cursor",
    ".taskmaster",
    "__pycache__",
}

LARGE_FILE_THRESHOLD_BYTES = 50 * 1024 * 1024  # 50 MB
TOP_N_LARGEST = 50
MAX_DUPLICATE_GROUP_FILES = 2000  # safety cap
VENDORED_DIR_NAMES = {
    "vendor",
    "third_party",
    "third-party",
    "thirdparty",
    "external",
    "extern",
    "deps",
    "dependencies",
    "site-packages",
    "dist",
    "build",
    "node_modules",
}


def load_gitignore_matcher(project_root: Path) -> Callable[[Path], bool]:
    if pathspec is None:
        return lambda _p: False
    gitignore = project_root / ".gitignore"
    if not gitignore.exists():
        return lambda _p: False
    spec = pathspec.PathSpec.from_lines(
        "gitwildmatch", gitignore.read_text(encoding="utf-8").splitlines()
    )

    def is_ignored(p: Path) -> bool:
        try:
            rel = p.relative_to(project_root).as_posix()
        except ValueError:
            return False
        return spec.match_file(rel)

    return is_ignored


@dataclass
class FileInfo:
    path: Path
    size: int


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def iter_files(root: Path, ignore: Callable[[Path], bool]) -> list[FileInfo]:
    results: list[FileInfo] = []
    for include in INCLUDE_DIRS:
        base = root / include
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if any(part in HARD_EXCLUDES for part in p.parts):
                continue
            if ignore(p):
                continue
            if p.is_file():
                try:
                    size = p.stat().st_size
                except OSError:
                    continue
                results.append(FileInfo(path=p, size=size))
    return results


def build_extension_distribution(
    files: list[FileInfo],
) -> list[tuple[str, int, int]]:
    by_ext_count: dict[str, int] = defaultdict(int)
    by_ext_size: dict[str, int] = defaultdict(int)
    for f in files:
        ext = f.path.suffix.lower() or "no-ext"
        by_ext_count[ext] += 1
        by_ext_size[ext] += f.size
    rows = [(ext, by_ext_count[ext], by_ext_size[ext]) for ext in by_ext_count]
    rows.sort(key=lambda r: r[2], reverse=True)
    return rows


def top_largest(files: list[FileInfo], n: int) -> list[FileInfo]:
    return sorted(files, key=lambda f: f.size, reverse=True)[:n]


def detect_duplicates(files: list[FileInfo]) -> list[list[Path]]:
    # Group by size first
    by_size: dict[int, list[FileInfo]] = defaultdict(list)
    for f in files:
        by_size[f.size].append(f)

    duplicates: list[list[Path]] = []
    candidates = [g for g in by_size.values() if len(g) > 1]

    # Limit the total number of files to hash for performance
    total_candidates = sum(len(g) for g in candidates)
    if total_candidates > MAX_DUPLICATE_GROUP_FILES:
        # Heuristic: only consider the smallest groups first
        candidates.sort(key=len)
    processed = 0

    for group in candidates:
        if processed >= MAX_DUPLICATE_GROUP_FILES:
            break
        # Hash each file in the size-group
        hash_to_paths: dict[str, list[Path]] = defaultdict(list)
        for fi in group:
            if processed >= MAX_DUPLICATE_GROUP_FILES:
                break
            try:
                h = sha1()
                with fi.path.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                        h.update(chunk)
                digest = h.hexdigest()
                hash_to_paths[digest].append(fi.path)
            except OSError:
                continue
            finally:
                processed += 1
        for paths in hash_to_paths.values():
            if len(paths) > 1:
                duplicates.append(paths)
    return duplicates


def detect_vendored_candidates(
    files: list[FileInfo], vendored_names: set[str] = VENDORED_DIR_NAMES
) -> dict[str, tuple[int, int]]:
    summary: dict[str, tuple[int, int]] = {}
    for f in files:
        rel_parts = f.path.relative_to(PROJECT_ROOT).parts
        for idx, part in enumerate(rel_parts):
            if part in vendored_names:
                key = Path(*rel_parts[: idx + 1]).as_posix()
                count, total = summary.get(key, (0, 0))
                summary[key] = (count + 1, total + f.size)
                break
    return summary


def append_report(
    report_path: Path,
    ext_rows: list[tuple[str, int, int]],
    largest: list[FileInfo],
    dups: list[list[Path]],
    vendored: dict[str, tuple[int, int]],
) -> None:
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S %Z")
    lines: list[str] = []
    # Auto-generated file: disable markdownlint to avoid noise from long paths and tables
    lines.append("<!-- markdownlint-disable-file -->")
    lines.append(f"# Scan Report — {ts}")
    lines.append("")

    # Extension distribution
    lines.append("## Extension distribution (selected dirs)")
    lines.append("")
    lines.append("Extension | Count | Total Size")
    lines.append(":-- | --: | --:")
    for ext, cnt, total in ext_rows:
        lines.append(f"`{ext}` | {cnt} | {human_size(total)}")
    lines.append("")

    # Top largest files
    lines.append(
        f"## Top {len(largest)} largest files — flags > {human_size(LARGE_FILE_THRESHOLD_BYTES)}"
    )
    lines.append("")
    # Long paths trigger MD013 (line length); disable for this table only
    lines.append("<!-- markdownlint-disable MD013 -->")
    lines.append("Size | Path | Flag")
    lines.append("--: | :-- | :--")
    for fi in largest:
        flag = "LARGE" if fi.size >= LARGE_FILE_THRESHOLD_BYTES else ""
        rel = fi.path.relative_to(PROJECT_ROOT).as_posix()
        lines.append(f"{human_size(fi.size)} | `{rel}` | {flag}")
    lines.append("")
    lines.append("<!-- markdownlint-enable MD013 -->")

    # Potential duplicates
    lines.append("## Potential duplicates — by size+hash, selected dirs")
    lines.append("")
    # Long paths trigger MD013 (line length); disable for this list only
    lines.append("<!-- markdownlint-disable MD013 -->")
    if not dups:
        lines.append("None detected in selected directories.")
    else:
        for idx, group in enumerate(dups, start=1):
            lines.append(f"- Group {idx}:")
            for p in group:
                rel = p.relative_to(PROJECT_ROOT).as_posix()
                lines.append(f"  - `{rel}`")
    lines.append("")
    lines.append("<!-- markdownlint-enable MD013 -->")

    # Vendored candidates
    lines.append("## Vendored/third-party candidates")
    lines.append("")
    if not vendored:
        lines.append("None detected in selected directories.")
    else:
        lines.append("Dir | Files | Total size")
        lines.append(":-- | --: | --:")
        for key, (count, total) in sorted(
            vendored.items(), key=lambda kv: kv[1][1], reverse=True
        ):
            lines.append(f"`{key}` | {count} | {human_size(total)}")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ignore = load_gitignore_matcher(PROJECT_ROOT)
    files = iter_files(PROJECT_ROOT, ignore)

    # Build summaries
    ext_rows = build_extension_distribution(files)
    largest = top_largest(files, TOP_N_LARGEST)

    # Duplicate detection on selected dirs only (already enforced by iter_files)
    dups = detect_duplicates(files)
    vendored = detect_vendored_candidates(files)

    append_report(REPORT_PATH, ext_rows, largest, dups, vendored)


if __name__ == "__main__":
    main()
