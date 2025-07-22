"""
Model structure comparison. This script compares the actual model
directory structure against the expected structure and reports
differences.
"""

import json
from typing import Any

# Type definitions
type InventoryEntry = dict[str, Any]
type Inventory = list[InventoryEntry]
type ExpectedStructure = dict[str, list[str]]
type FileLocation = tuple[str, str]  # (folder, filename)
type FileSet = set[FileLocation]
type MisplacedList = list[FileLocation]
type StructureReport = dict[str, list[FileLocation] | MisplacedList]

# Cargar inventario real
with open("scripts/reports/model_inventory.json", encoding="utf-8") as f:
    inventory: Inventory = json.load(f)

# Cargar estructura esperada
with open(
    "scripts/reports/model_expected_structure.json", encoding="utf-8"
) as f:
    expected: ExpectedStructure = json.load(f)

# Construir sets para comparación
actual_files: FileSet = set()
for entry in inventory:
    rel_path: str = entry["relative_path"]
    # Determinar carpeta base
    parts = rel_path.split("/")
    if len(parts) == 1:
        folder = "root"
        fname = parts[0]
    else:
        folder = parts[0]
        fname = (
            "/".join(parts[1:])
            if len(parts) > 2  # noqa: PLR2004
            else parts[1]  # noqa: PLR2004
        )
    actual_files.add((folder, fname))

expected_files: FileSet = set()
for folder, files in expected.items():
    for fname in files:
        expected_files.add((folder, fname))

# Archivos faltantes y archivos inesperados
missing: list[FileLocation] = sorted(expected_files - actual_files)
unexpected: list[FileLocation] = sorted(actual_files - expected_files)

# Archivos en ubicaciones incorrectas (nombre esperado pero carpeta incorrecta)
expected_names: set[str] = {fname for _, fname in expected_files}
actual_names: set[str] = {fname for _, fname in actual_files}
misplaced: MisplacedList = []
for folder, fname in actual_files:
    if fname in expected_names and (folder, fname) not in expected_files:
        misplaced.append((folder, fname))

# Generar reporte
report: StructureReport = {
    "missing_files": missing,
    "unexpected_files": unexpected,
    "misplaced_files": misplaced,
}

with open(
    "scripts/reports/model_structure_diff.json", "w", encoding="utf-8"
) as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(
    "Comparison complete. Report written to "
    "scripts/reports/model_structure_diff.json"
)
