"""
Audit repository for generated/temporary artifacts and large binaries.

Outputs a markdown report at:
  docs/reports/project-reports/technical/artifacts_and_binaries_audit.md

Capabilities:
- Scans entire repo (excluding .git and known tool dirs)
- Flags typical artifact patterns (caches, temp, bytecode, logs)
- Finds large files above a size threshold (default 50 MB)
- Notes whether a path is ignored per .gitignore (via pathspec) and suggests
  .gitignore or Git LFS actions where appropriate

No heavy deps. Safe to run in CI/local. English-only output.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
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
    / "artifacts_and_binaries_audit.md"
)

HARD_EXCLUDES = {
    ".git",
    ".cursor",
    ".taskmaster",
}

LARGE_FILE_THRESHOLD_BYTES = 50 * 1024 * 1024  # 50 MB

# Common artifact patterns (anywhere in path or by suffix)
ARTIFACT_DIR_NAMES = {
    "__pycache__",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    ".coverage",
    ".venv",
    "build",
    "dist",
    "node_modules",
    "site-packages",
}
ARTIFACT_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".log",
    ".tmp",
    ".bak",
    ".swp",
    ".swo",
}

LARGE_FILE_HINT_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".pdf",
    ".onnx",
    ".pt",
    ".pth",
    ".h5",
    ".ckpt",
    ".npy",
    ".npz",
}


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


def iter_all_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if any(part in HARD_EXCLUDES for part in p.parts):
            continue
        if p.is_file():
            yield p


@dataclass
class ArtifactFinding:
    path: Path
    kind: str  # dir-name or suffix
    ignored: bool


@dataclass
class LargeFileFinding:
    path: Path
    size: int
    suffix: str
    ignored: bool


def is_artifact_path(p: Path) -> tuple[bool, str | None]:
    if any(name in ARTIFACT_DIR_NAMES for name in p.parts):
        return True, "dir-name"
    if p.suffix.lower() in ARTIFACT_SUFFIXES:
        return True, "suffix"
    return False, None


def build_report(
    artifacts: list[ArtifactFinding], large_files: list[LargeFileFinding]
) -> str:
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S %Z")
    lines: list[str] = []
    lines.append("<!-- markdownlint-disable-file -->")
    lines.append(f"# Artifacts and large binaries audit — {ts}")
    lines.append("")
    lines.append(
        "This report highlights generated/temporary artifacts and large binaries."
    )
    lines.append("Use it to update .gitignore or configure Git LFS as needed.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    total_artifacts = len(artifacts)
    total_large = len(large_files)
    lines.append(f"Artifacts detected: {total_artifacts}")
    lines.append(
        f"Large files (>{LARGE_FILE_THRESHOLD_BYTES // (1024 * 1024)} MB): {total_large}"
    )
    lines.append("")

    lines.append("## Artifacts (caches, temp, bytecode, logs)")
    lines.append("")
    if not artifacts:
        lines.append("None detected.")
    else:
        lines.append("Path | Kind | Ignored by .gitignore")
        lines.append(":-- | :-- | :--:")
        for a in sorted(artifacts, key=lambda x: x.path.as_posix()):
            rel = a.path.relative_to(PROJECT_ROOT).as_posix()
            lines.append(
                f"`{rel}` | {a.kind} | {'yes' if a.ignored else 'no'}"
            )
    lines.append("")

    lines.append(
        f"## Large files (>{LARGE_FILE_THRESHOLD_BYTES // (1024 * 1024)} MB) — consider Git LFS for appropriate types"
    )
    lines.append("")
    if not large_files:
        lines.append("None detected.")
    else:
        lines.append("Size | Path | Ext | Ignored by .gitignore")
        lines.append("--: | :-- | :--: | :--:")
        for lf in sorted(large_files, key=lambda x: x.size, reverse=True):
            rel = lf.path.relative_to(PROJECT_ROOT).as_posix()
            size_mb = lf.size / (1024 * 1024)
            lines.append(
                f"{size_mb:.2f} MB | `{rel}` | `{lf.suffix or 'no-ext'}` | {'yes' if lf.ignored else 'no'}"
            )
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    lines.append(
        "- Add common artifact patterns to .gitignore if not present (see below)"
    )
    lines.append(
        "- Track large, important binary assets with Git LFS (e.g., .pt, .onnx, .ckpt, media)"
    )
    lines.append(
        "- Avoid committing generated logs or caches; clean them in CI"
    )
    lines.append("")
    lines.append("### Suggested .gitignore entries")
    lines.append("")
    lines.append("```")
    lines.extend(sorted({f"{name}/" for name in ARTIFACT_DIR_NAMES}))
    lines.extend(sorted(ARTIFACT_SUFFIXES))
    lines.append("```")
    lines.append("")

    lines.append("### Suggested Git LFS patterns (example)")
    lines.append("")
    lines.append("```")
    lines.append("git lfs track '*.pt'")
    lines.append("git lfs track '*.pth'")
    lines.append("git lfs track '*.onnx'")
    lines.append("git lfs track '*.ckpt'")
    lines.append("git lfs track '*.mp4' '*.mov' '*.avi'")
    lines.append("git lfs track '*.zip' '*.tar' '*.gz' '*.7z'")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    is_ignored = load_gitignore_matcher(PROJECT_ROOT)
    artifacts: list[ArtifactFinding] = []
    large_files: list[LargeFileFinding] = []
    for p in iter_all_files(PROJECT_ROOT):
        try:
            size = p.stat().st_size
        except OSError:
            continue

        art, kind = is_artifact_path(p)
        ignored = is_ignored(p)
        if art:
            artifacts.append(
                ArtifactFinding(
                    path=p, kind=kind or "unknown", ignored=ignored
                )
            )

        if size >= LARGE_FILE_THRESHOLD_BYTES:
            suffix = p.suffix.lower()
            large_files.append(
                LargeFileFinding(
                    path=p, size=size, suffix=suffix, ignored=ignored
                )
            )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        build_report(artifacts, large_files), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
