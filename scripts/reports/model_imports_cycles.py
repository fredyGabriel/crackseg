"""
Model import s cycle detection. This script analyzes import
dependencies to detect circular import s in the model directory.
"""

import json
import os
from collections import defaultdict
from typing import Any

# Type definitions
type ImportEntry = dict[str, Any]
type ImportCatalog = list[ImportEntry]
type ModuleName = str
type DependencyGraph = defaultdict[ModuleName, set[ModuleName]]
type CyclePath = list[ModuleName]
type CycleList = list[CyclePath]

CATALOG_PATH = os.path.join(
    os.path.dirname(__file__), "model_import s_catalog.json"
)
CYCLES_PATH = os.path.join(
    os.path.dirname(__file__), "model_import s_cycles.json"
)

# Construir grafo de dependencias
with open(CATALOG_PATH, encoding="utf-8") as f:
    catalog: ImportCatalog = json.load(f)

graph: DependencyGraph = defaultdict(set)
modules: set[ModuleName] = set()

for entry in catalog:
    src = entry["file"]
    dst = entry["module"]
    # Solo considerar import s internos
    if dst and (dst.startswith("src.model") or dst.startswith(".")):
        graph[src].add(dst)
        modules.add(src)
        modules.add(dst)


# Normalizar nodos (convertir rutas relativas a absolutas si es posible)
def normalize(module: ModuleName) -> ModuleName:
    """Normalize module name for consistent comparison."""
    if module.startswith("."):
        return module  # Mantener relativo, requiere análisis más profundo
    return module


graph_norm: DependencyGraph = defaultdict(set)
for src, dsts in graph.items():
    src_n = normalize(src)
    for dst in dsts:
        dst_n = normalize(dst)
        graph_norm[src_n].add(dst_n)


# Detección de ciclos (DFS)
def find_cycles(graph: DependencyGraph) -> CycleList:
    """Find circular dependencies in the import  graph."""
    visited: set[ModuleName] = set()
    cycles: CycleList = []

    def visit(node: ModuleName, path: CyclePath) -> None:
        if node in path:
            cycle = path[path.index(node) :] + [node]
            cycles.append(cycle)
            return
        if node in visited:
            return
        visited.add(node)
        for neighbor in graph.get(node, []):
            visit(neighbor, path + [node])

    for node in graph:
        visit(node, [])
    # Eliminar duplicados
    unique_cycles: CycleList = []
    seen: set[tuple[ModuleName, ...]] = set()
    for cycle in cycles:
        key = tuple(sorted(cycle))
        if key not in seen:
            unique_cycles.append(cycle)
            seen.add(key)
    return unique_cycles


cycles = find_cycles(graph_norm)

with open(CYCLES_PATH, "w", encoding="utf-8") as f:
    json.dump(cycles, f, indent=2, ensure_ascii=False)

if cycles:
    print(
        f"Se detectaron {len(cycles)} ciclos de import ación. Ver "
        f"{CYCLES_PATH}"
    )
else:
    print("No se detectaron ciclos de import ación.")


#
