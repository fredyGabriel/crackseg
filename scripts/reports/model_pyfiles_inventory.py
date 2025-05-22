import json
import os

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
root_dir = os.path.join(project_root, "src", "model")
outfile = os.path.join(os.path.dirname(__file__), "model_pyfiles.json")

pyfiles = []

for dirpath, dirnames, filenames in os.walk(root_dir):
    # Ignorar carpetas __pycache__
    dirnames[:] = [d for d in dirnames if d != "__pycache__"]
    for filename in filenames:
        if filename.endswith(".py") and not filename.endswith(".pyc"):
            filepath = os.path.join(dirpath, filename)
            relpath = os.path.relpath(filepath, start=root_dir)
            pyfiles.append(relpath.replace("\\", "/"))

with open(outfile, "w", encoding="utf-8") as f:
    json.dump(pyfiles, f, indent=2, ensure_ascii=False)

print(f"Python file inventory written to {outfile}")
