"""
Generate import dependency graph for src/ and report cycles and coupling.

Outputs: docs/reports/analysis-reports/architecture/dependency_graph.md

Features:
- Parses Python imports (import/from) within src/
- Builds directed graph module->module (dotted paths)
- Reports strongly connected components (cycles), fan-in/out, and top hotspots
- Optional layer checks via simple prefix ordering (configurable list)
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from scripts.utils.common.io_utils import write_text  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
REPORT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "reports"
    / "analysis-reports"
    / "architecture"
    / "dependency_graph.md"
)

IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))", re.MULTILINE
)


def iter_py_files(root: Path) -> Iterable[Path]:
    yield from root.rglob("*.py")


def dotted_module_from_src_path(p: Path) -> str | None:
    try:
        rel = p.relative_to(SRC_ROOT)
    except ValueError:
        return None
    parts = list(rel.parts)
    if not parts:
        return None
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    elif parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    return ".".join(parts) if parts else None


def parse_imports(text: str) -> set[str]:
    mods: set[str] = set()
    for m in IMPORT_RE.finditer(text):
        pkg = m.group(1) or m.group(2)
        if not pkg:
            continue
        mods.add(pkg)
        # Add parent packages to mitigate false negatives
        while "." in pkg:
            pkg = pkg.rsplit(".", 1)[0]
            mods.add(pkg)
    return mods


def build_graph() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    graph: dict[str, set[str]] = defaultdict(set)
    nodes: set[str] = set()
    for p in iter_py_files(SRC_ROOT):
        me = dotted_module_from_src_path(p)
        if me is None or me.endswith(".__main__"):
            continue
        nodes.add(me)
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for dep in parse_imports(text):
            # only keep edges within our src namespace
            if not dep.startswith("crackseg"):
                continue
            graph[me].add(dep)
    # ensure all nodes present
    for n in list(nodes):
        graph.setdefault(n, set())
    # reverse graph
    reverse: dict[str, set[str]] = defaultdict(set)
    for a, outs in graph.items():
        for b in outs:
            reverse[b].add(a)
    return graph, reverse


def scc_kosaraju(graph: dict[str, set[str]]) -> list[list[str]]:
    visited: set[str] = set()
    order: list[str] = []

    def dfs(v: str) -> None:
        visited.add(v)
        for w in graph.get(v, ()):  # type: ignore[arg-type]
            if w not in visited:
                dfs(w)
        order.append(v)

    for v in graph.keys():
        if v not in visited:
            dfs(v)

    # build reverse
    rev: dict[str, set[str]] = defaultdict(set)
    for a, outs in graph.items():
        for b in outs:
            rev[b].add(a)

    comps: list[list[str]] = []
    visited.clear()

    def dfs_rev(v: str, acc: list[str]) -> None:
        visited.add(v)
        acc.append(v)
        for w in rev.get(v, ()):  # type: ignore[arg-type]
            if w not in visited:
                dfs_rev(w, acc)

    for v in reversed(order):
        if v not in visited:
            acc: list[str] = []
            dfs_rev(v, acc)
            comps.append(acc)
    return comps


def build_report(
    graph: dict[str, set[str]], reverse: dict[str, set[str]]
) -> str:
    total_nodes = len(graph)
    total_edges = sum(len(v) for v in graph.values())
    comps = scc_kosaraju(graph)
    cycles = [c for c in comps if len(c) > 1]

    lines: list[str] = []
    lines.append("<!-- markdownlint-disable-file -->")
    lines.append("# Dependency graph report")
    lines.append("")
    lines.append(
        f"Nodes: {total_nodes} | Edges: {total_edges} | Cycles: {len(cycles)}"
    )
    lines.append("")

    # Top fan-out and fan-in
    fanout = sorted(
        ((n, len(deps)) for n, deps in graph.items()),
        key=lambda x: x[1],
        reverse=True,
    )[:20]
    fanin = sorted(
        ((n, len(reverse.get(n, ()))) for n in graph.keys()),
        key=lambda x: x[1],
        reverse=True,
    )[:20]

    lines.append("## Top fan-out (dependencies)")
    lines.append("")
    lines.append("Module | Outgoing")
    lines.append(":-- | --:")
    for n, k in fanout:
        lines.append(f"`{n}` | {k}")
    lines.append("")

    lines.append("## Top fan-in (dependents)")
    lines.append("")
    lines.append("Module | Incoming")
    lines.append(":-- | --:")
    for n, k in fanin:
        lines.append(f"`{n}` | {k}")
    lines.append("")

    lines.append("## Cycles (SCCs > 1)")
    lines.append("")
    if not cycles:
        lines.append("None detected.")
    else:
        for idx, comp in enumerate(
            sorted(cycles, key=len, reverse=True), start=1
        ):
            lines.append(f"- Cycle {idx} ({len(comp)} modules):")
            for m in sorted(comp):
                lines.append(f"  - `{m}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    graph, reverse = build_graph()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_text(REPORT_PATH, build_report(graph, reverse))


if __name__ == "__main__":
    main()
