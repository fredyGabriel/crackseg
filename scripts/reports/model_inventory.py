import json
import os
from datetime import datetime

# Obtener la ruta absoluta de src/model desde la raíz del proyecto
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
root_dir = os.path.join(project_root, "src", "model")
outfile = os.path.join(os.path.dirname(__file__), "model_inventory.json")

inventory = []

for dirpath, _dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".py") or filename.endswith(".md"):
            filepath = os.path.join(dirpath, filename)
            relpath = os.path.relpath(filepath, start=root_dir)
            stat = os.stat(filepath)
            inventory.append(
                {
                    "relative_path": relpath.replace("\\", "/"),
                    "size_bytes": stat.st_size,
                    "last_modified": datetime.fromtimestamp(
                        stat.st_mtime
                    ).isoformat(),
                }
            )
            # Depuración: imprimir cada archivo encontrado
            print(f"Found: {relpath.replace('\\', '/')}")

with open(outfile, "w", encoding="utf-8") as f:
    json.dump(inventory, f, indent=2, ensure_ascii=False)

print(f"Inventory written to {outfile}")
