from __future__ import annotations

import ast
import hashlib
import io
import os
import tokenize
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from scripts.utils.common.io_utils import read_text, write_json, write_text


@dataclass
class Occurrence:
    path: str
    kind: str  # "function" | "class"
    name: str
    lineno: int
    end_lineno: int


class DuplicateGroup(TypedDict):
    hash: str
    count: int
    preview: str
    occurrences: list[dict[str, Any]]


def iter_python_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        for base, _dirs, files in os.walk(root):
            for f in files:
                if f.endswith(".py"):
                    yield Path(base) / f


def read_text_compat(path: Path) -> str:
    # Back-compat wrapper to use shared IO utils
    return read_text(path)


def normalize_code(code: str) -> str:
    tokens: list[str] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type in (
                tokenize.COMMENT,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENCODING,
            ):
                continue
            tokens.append(tok.string)
    except tokenize.TokenError:
        # Fallback: basic whitespace collapse
        return " ".join(code.split())
    return " ".join(tokens)


def hash_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8", errors="ignore")).hexdigest()


def extract_snippet(src: str, start: int, end: int) -> str:
    lines = src.splitlines()
    # lineno is 1-based
    start_idx = max(1, start) - 1
    end_idx = max(start_idx, end - 1)
    snippet_lines = lines[start_idx : end_idx + 1]
    return "\n".join(snippet_lines)


def scan_file(path: Path) -> list[tuple[str, Occurrence, str]]:
    """Return list of (hash, occurrence, preview) for this file.

    The hash is computed from a token-normalized snippet of each function/class.
    """
    src = read_text_compat(path)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    results: list[tuple[str, Occurrence, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            kind = "function"
            name = node.name
        elif isinstance(node, ast.ClassDef):
            kind = "class"
            name = node.name
        else:
            continue

        lineno = getattr(node, "lineno", None)
        end_lineno = getattr(node, "end_lineno", None)
        if lineno is None or end_lineno is None:
            continue

        raw_snippet = extract_snippet(src, lineno, end_lineno)
        normalized = normalize_code(raw_snippet)

        # Skip very small snippets
        if len(normalized) < 200:  # ~40-60 tokens threshold
            continue

        h = hash_code(normalized)
        occ = Occurrence(
            path=str(path).replace("\\", "/"),
            kind=kind,
            name=name,
            lineno=lineno,
            end_lineno=end_lineno,
        )
        preview = "\n".join(raw_snippet.splitlines()[:10])
        results.append((h, occ, preview))

    return results


def main() -> None:
    roots = [Path("src"), Path("gui"), Path("scripts")]
    hash_to_occurrences: dict[str, list[Occurrence]] = {}
    hash_to_preview: dict[str, str] = {}

    for file_path in iter_python_files(roots):
        for h, occ, preview in scan_file(file_path):
            hash_to_occurrences.setdefault(h, []).append(occ)
            if h not in hash_to_preview:
                hash_to_preview[h] = preview

    duplicates: list[DuplicateGroup] = []
    for h, occs in hash_to_occurrences.items():
        if len(occs) < 2:
            continue
        group: DuplicateGroup = {
            "hash": h,
            "count": len(occs),
            "preview": hash_to_preview.get(h, ""),
            "occurrences": [occ.__dict__ for occ in occs],
        }
        duplicates.append(group)

    duplicates.sort(key=lambda d: d["count"], reverse=True)

    out_dir = Path("docs/reports/project-reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "duplicate_scan_report.json", duplicates, indent=2)

    # Markdown summary
    lines: list[str] = ["# Duplicate Scan Report\n", "\n"]
    lines.append(f"Total duplicate groups: {len(duplicates)}\n\n")
    for group in duplicates[:20]:
        lines.append(f"## Group (count={group['count']})\n\n")
        lines.append("```python\n")
        lines.append(str(group.get("preview", "")))
        lines.append("\n```\n\n")
        for occ in group["occurrences"]:  # type: ignore[index]
            lines.append(
                f"- {occ['kind']} {occ['name']} @ {occ['path']}:{occ['lineno']}-{occ['end_lineno']}\n"
            )
        lines.append("\n")

    write_text(out_dir / "duplicate_scan_report.md", "".join(lines))

    print("OK")


if __name__ == "__main__":
    main()
