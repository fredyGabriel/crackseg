import ast
import csv
import os

# ruff: noqa: E501
TEST_DIRS = ["tests/unit", "tests/integration"]
OUTPUT_CSV = "outputs/prd_project_refinement/test_suite_evaluation/reports/test_inventory.csv"

# Columnas del CSV
CSV_HEADER = [
    "Test ID",
    "File Path",
    "Test Name",
    "Purpose",
    "Type",
    "Coverage Area",
    "Dependencies",
    "Functionality/Component Tag",
    "Status",
    "Author",
    "Last Modified Date",
    "Orphaned/Deprecated",
]

# Explicación inicial
CSV_EXPLANATION = "# Test inventory for all unit and integration tests. Each row describes a test function, its location, and key metadata for tracking and analysis."


def get_test_functions(file_path):
    """Extrae funciones de test y sus docstrings de un archivo Python."""
    with open(file_path, encoding="utf-8") as f:
        node = ast.parse(f.read(), filename=file_path)
    tests = []
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
            docstring = ast.get_docstring(item) or "N/A"
            tests.append((item.name, docstring))
    return tests


def get_type_from_path(path):
    if "unit" in path:
        return "unit"
    if "integration" in path:
        return "integration"
    return "N/A"


def main():
    rows = []
    for test_dir in TEST_DIRS:
        for root, _, files in os.walk(test_dir):
            for fname in files:
                if fname.startswith("test_") and fname.endswith(".py"):
                    file_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(file_path)
                    test_type = get_type_from_path(rel_path)
                    try:
                        test_functions = get_test_functions(file_path)
                    except Exception as e:
                        print(f"Error parsing {file_path}: {e}")
                        continue
                    for idx, (func_name, docstring) in enumerate(
                        test_functions, 1
                    ):
                        test_id = f"{test_type}-{os.path.basename(root)}-"
                        f"{fname.replace('.py','')}-{idx}"
                        row = [
                            test_id,
                            rel_path.replace("\\", "/"),
                            func_name,
                            docstring.replace("\n", " ").strip(),
                            test_type,
                            "N/A",  # Coverage Area
                            "N/A",  # Dependencies
                            "N/A",  # Functionality/Component Tag
                            "N/A",  # Status
                            "N/A",  # Author
                            "N/A",  # Last Modified Date
                            "N/A",  # Orphaned/Deprecated
                        ]
                        rows.append(row)
    # Escribir CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        f.write(CSV_EXPLANATION + "\n")
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)
    print(f"Test inventory generated: {OUTPUT_CSV} ({len(rows)} tests)")


if __name__ == "__main__":
    main()
