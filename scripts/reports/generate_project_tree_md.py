from __future__ import annotations

import os
from pathlib import Path


def build_tree(root: Path, max_depth: int = 6) -> list[str]:
    lines: list[str] = []
    root = root.resolve()

    for base, dirs, files in os.walk(root):
        rel = Path(base).relative_to(root)
        depth = len(rel.parts) if rel.parts else 0
        if depth > max_depth:
            continue

        # Sort entries for stable output
        dirs.sort()
        files.sort()

        if depth == 0:
            lines.append(f"{root.name}/")
        else:
            indent = "  " * depth
            lines.append(f"{indent}- {rel.name}/")

        for f in files:
            indent = "  " * (depth + 1)
            lines.append(f"{indent}- {f}")

    return lines


def write_project_tree_md(out_file: Path, max_depth: int = 6) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Project Tree\n", "\n", "``text\n"]
    lines.extend(build_tree(Path.cwd(), max_depth=max_depth))
    lines.append("\n``\n")
    out_file.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    write_project_tree_md(Path("docs/reports/project_tree.md"), max_depth=6)
    print("OK")
