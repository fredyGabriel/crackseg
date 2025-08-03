"""
Inventory training scripts that import from crackseg.model and direct model
instantiations.

Scans 'scripts/', 'src/training/', and 'src/' for .py files, parses them
with ast, and outputs a CSV with import and instantiation details.
"""

import ast
import csv
import os
from typing import Any

SEARCH_DIRS = ["scripts", os.path.join("src", "training"), "src"]
OUTPUT_CSV = "training_imports_inventory.csv"
MODEL_IMPORT_ROOT = "src.model"


def find_py_files(base_dirs: list[str]) -> list[str]:
    files = []
    for base in base_dirs:
        for root, _, filenames in os.walk(base):
            for fname in filenames:
                if fname.endswith(".py"):
                    files.append(os.path.join(root, fname))
    return files


def analyze_file(filepath: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    try:
        with open(filepath, encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except Exception:
        return results

    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(MODEL_IMPORT_ROOT):
                    results.append(
                        {
                            "file": filepath,
                            "import_type": "import",
                            "imported_module": alias.name,
                            "imported_name": None,
                            "alias": alias.asname,
                            "line_number": node.lineno,
                            "instantiates_model_class": False,
                        }
                    )
                    imported_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith(MODEL_IMPORT_ROOT):
                for alias in node.names:
                    results.append(
                        {
                            "file": filepath,
                            "import_type": "from",
                            "imported_module": node.module,
                            "imported_name": alias.name,
                            "alias": alias.asname,
                            "line_number": node.lineno,
                            "instantiates_model_class": False,
                        }
                    )
                    imported_names.add(alias.asname or alias.name)

    # Detect instantiations of imported model classes
    class ModelClassVisitor(ast.NodeVisitor):
        def __init__(self, imported_names: set[str]) -> None:
            self.imported_names = imported_names
            self.instantiations: list[tuple[str, int]] = []

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if isinstance(func, ast.Name) and func.id in self.imported_names:
                self.instantiations.append((func.id, node.lineno))
            elif isinstance(func, ast.Attribute):
                value = func.value
                if (
                    isinstance(value, ast.Name)
                    and value.id in self.imported_names
                ):
                    self.instantiations.append(
                        (f"{value.id}.{func.attr}", node.lineno)
                    )
            self.generic_visit(node)

    visitor = ModelClassVisitor(imported_names)
    visitor.visit(tree)
    for name, lineno in visitor.instantiations:
        results.append(
            {
                "file": filepath,
                "import_type": "instantiation",
                "imported_module": None,
                "imported_name": name,
                "alias": None,
                "line_number": lineno,
                "instantiates_model_class": True,
            }
        )
    return results


def main() -> None:
    py_files = find_py_files(SEARCH_DIRS)
    all_results = []
    for fpath in py_files:
        all_results.extend(analyze_file(fpath))
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "file",
                "import_type",
                "imported_module",
                "imported_name",
                "alias",
                "line_number",
                "instantiates_model_class",
            ],
        )
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
    print(f"Inventory complete. Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
