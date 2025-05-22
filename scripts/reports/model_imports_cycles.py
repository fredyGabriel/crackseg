import json
import os
from collections import defaultdict

CATALOG_PATH = os.path.join(
    os.path.dirname(__file__), "model_imports_catalog.json"
)
CYCLES_PATH = os.path.join(
    os.path.dirname(__file__), "model_imports_cycles.json"
)

# Construir grafo de dependencias
with open(CATALOG_PATH, encoding="utf-8") as f:
    catalog = json.load(f)

graph = defaultdict(set)
modules = set()

for entry in catalog:
    src = entry["file"]
    dst = entry["module"]
    # Solo considerar imports internos
    if dst and (dst.startswith("src.model") or dst.startswith(".")):
        graph[src].add(dst)
        modules.add(src)
        modules.add(dst)


# Normalizar nodos (convertir rutas relativas a absolutas si es posible)
def normalize(module):
    if module.startswith("."):
        return module  # Mantener relativo, requiere análisis más profundo
    return module


graph_norm = defaultdict(set)
for src, dsts in graph.items():
    src_n = normalize(src)
    for dst in dsts:
        dst_n = normalize(dst)
        graph_norm[src_n].add(dst_n)


# Detección de ciclos (DFS)
def find_cycles(graph):
    visited = set()
    cycles = []

    def visit(node, path):
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
    unique_cycles = []
    seen = set()
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
        f"Se detectaron {len(cycles)} ciclos de importación. "
        f"Ver {CYCLES_PATH}"
    )
else:
    print("No se detectaron ciclos de importación.")
