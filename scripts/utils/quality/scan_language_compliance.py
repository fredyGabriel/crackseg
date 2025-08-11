"""
Scan technical files for non-English content (Spanish detection heuristic).

Outputs a markdown report at:
  docs/reports/project-reports/documentation/language_compliance_report.md

Heuristics (no heavy dependencies):
- Respects .gitignore via pathspec (if available)
- Flags presence of Spanish-specific chars (áéíóúÁÉÍÓÚñÑ¿¡)
- Counts Spanish stopwords across text (excluding fenced code blocks in .md)
- Reports reasons and examples to aid remediation (translate/merge/remove)

Targets: .py, .md, .mdc, .yaml, .yml, .json, .ini, .txt
Excludes: artifacts/, data/, .git/, .cursor/, .taskmaster/, __pycache__/
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
    / "documentation"
    / "language_compliance_report.md"
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

TARGET_EXTS = {".py", ".md", ".mdc", ".yaml", ".yml", ".json", ".ini", ".txt"}

SPANISH_CHARS = set("áéíóúÁÉÍÓÚñÑ¿¡")
SPANISH_STOPWORDS = {
    "de",
    "la",
    "el",
    "y",
    "en",
    "que",
    "los",
    "del",
    "se",
    "las",
    "por",
    "con",
    "para",
    "una",
    "su",
    "al",
    "como",
    "más",
    "sí",
    "no",
    "este",
    "esta",
    "estas",
    "estos",
    "entre",
    "sobre",
}


def load_gitignore_matcher(project_root: Path):
    if pathspec is None:
        return lambda _p: False
    gi = project_root / ".gitignore"
    if not gi.exists():
        return lambda _p: False
    from scripts.utils.common.io_utils import read_text  # noqa: E402

    spec = pathspec.PathSpec.from_lines(
        "gitwildmatch", read_text(gi).splitlines()
    )

    def is_ignored(p: Path) -> bool:
        try:
            rel = p.relative_to(project_root).as_posix()
        except ValueError:
            return False
        return spec.match_file(rel)

    return is_ignored


@dataclass
class Finding:
    path: Path
    reason: str
    evidence: str


def iter_candidate_files(root: Path, ignore) -> Iterable[Path]:
    for base_rel in INCLUDE_DIRS:
        base = root / base_rel
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if any(part in HARD_EXCLUDES for part in p.parts):
                continue
            if ignore(p):
                continue
            if p.is_file() and p.suffix.lower() in TARGET_EXTS:
                yield p


def strip_md_code_blocks(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    fenced = False
    for line in lines:
        if line.strip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            out.append(line)
    return "\n".join(out)


def tokenize(text: str) -> list[str]:
    buf = []
    word = []
    for ch in text:
        if ch.isalpha() or ch in SPANISH_CHARS:
            word.append(ch.lower())
        else:
            if word:
                buf.append("".join(word))
                word = []
    if word:
        buf.append("".join(word))
    return buf


def analyze_text(text: str, is_markdown: bool) -> tuple[bool, str, str]:
    if is_markdown:
        text = strip_md_code_blocks(text)
    has_spanish_char = any(c in SPANISH_CHARS for c in text)
    tokens = tokenize(text)
    total_tokens = max(len(tokens), 1)
    sw_hits = sum(1 for t in tokens if t in SPANISH_STOPWORDS)
    sw_ratio = sw_hits / total_tokens

    suspicious = has_spanish_char or sw_hits >= 10 or sw_ratio >= 0.06
    reason_parts: list[str] = []
    if has_spanish_char:
        reason_parts.append("contains Spanish diacritics/punctuation")
    if sw_hits:
        reason_parts.append(f"{sw_hits} Spanish stopwords ({sw_ratio:.2%})")
    reason = "; ".join(reason_parts) or "heuristic threshold"

    evidence = ""
    if suspicious:
        # extract a short snippet with a few Spanish tokens if possible
        top_tokens: list[str] = []
        seen: set[str] = set()
        for t in tokens:
            if t in SPANISH_STOPWORDS and t not in seen:
                seen.add(t)
                top_tokens.append(t)
            if len(top_tokens) >= 8:
                break
        evidence = ", ".join(top_tokens)
        if not evidence and has_spanish_char:
            evidence = "diacritics present"
    return suspicious, reason, evidence


def build_report(findings: list[Finding]) -> str:
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S %Z")
    lines: list[str] = []
    lines.append("<!-- markdownlint-disable-file -->")
    lines.append(f"# Language compliance report — {ts}")
    lines.append("")
    lines.append(
        "This report flags files that likely contain non-English (Spanish) content."
    )
    lines.append(
        "Heuristic-based, no external models. Please review before taking action."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"Total findings: {len(findings)}")
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    if not findings:
        lines.append("No suspicious files detected in targeted directories.")
    else:
        lines.append("Path | Reason | Evidence")
        lines.append(":-- | :-- | :--")
        for f in findings:
            rel = f.path.relative_to(PROJECT_ROOT).as_posix()
            lines.append(f"`{rel}` | {f.reason} | {f.evidence}")
    lines.append("")
    # Remediation plan summary
    categories: dict[str, list[Finding]] = {
        "legacy-docs": [],
        "markdown-docs": [],
        "python-code": [],
        "configs": [],
        "tests": [],
        "others": [],
    }
    for f in findings:
        p = f.path
        if "legacy" in p.as_posix():
            categories["legacy-docs"].append(f)
        elif p.suffix.lower() in {".md", ".mdc"}:
            categories["markdown-docs"].append(f)
        elif p.suffix.lower() == ".py":
            categories["python-code"].append(f)
        elif p.suffix.lower() in {".yaml", ".yml", ".json", ".ini"}:
            categories["configs"].append(f)
        elif "tests/" in p.as_posix():
            categories["tests"].append(f)
        else:
            categories["others"].append(f)

    lines.append("## Remediation plan (by category)")
    lines.append("")
    lines.append("Category | Files | Suggested action")
    lines.append(":-- | --: | :--")
    suggestions = {
        "legacy-docs": "Archive under docs/.../legacy or translate key sections",
        "markdown-docs": "Translate to English; consolidate duplicates",
        "python-code": "Translate comments/strings; enforce English-only",
        "configs": "Ensure keys/values and comments are English-only",
        "tests": "Translate test names/messages if user-facing",
        "others": "Case-by-case review",
    }
    for key in [
        "legacy-docs",
        "markdown-docs",
        "python-code",
        "configs",
        "tests",
        "others",
    ]:
        group = categories[key]
        count = len(group)
        if count == 0:
            continue
        lines.append(f"{key} | {count} | {suggestions[key]}")

    lines.append("")
    lines.append("## Recommended next steps")
    lines.append("")
    lines.append("- Translate Spanish content to English where appropriate")
    lines.append("- Consolidate duplicated docs (prefer English versions)")
    lines.append(
        "- Move legacy or non-essential Spanish docs to a legacy folder or archive"
    )
    lines.append(
        "- Ensure code comments, strings, and configuration keys are English-only"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ignore = load_gitignore_matcher(PROJECT_ROOT)
    findings: list[Finding] = []
    for p in iter_candidate_files(PROJECT_ROOT, ignore):
        try:
            from scripts.utils.common.io_utils import read_text  # noqa: E402

            text = read_text(p)
        except OSError:
            continue
        is_md = p.suffix.lower() in {".md", ".mdc"}
        suspicious, reason, evidence = analyze_text(text, is_md)
        if suspicious:
            findings.append(Finding(path=p, reason=reason, evidence=evidence))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    from scripts.utils.common.io_utils import write_text  # noqa: E402

    write_text(REPORT_PATH, build_report(findings))


if __name__ == "__main__":
    main()
