import ast
import json
import os
import re
import shutil

# Paths to files
REPORT_PATH = os.path.join(
    os.path.dirname(__file__), "model_imports_invalid.json"
)
CATALOG_PATH = os.path.join(
    os.path.dirname(__file__), "model_imports_catalog.json"
)
BACKUP_DIR = os.path.join(os.path.dirname(__file__), "autofix_backups")
LOG_PATH = os.path.join(
    os.path.dirname(__file__), "model_imports_autofix_log.json"
)

# Replacement patterns: (old, new)
REPLACEMENTS = [
    (
        r"^(base|core|factory|common|components|bottleneck|decoder|encoder|"
        r"architectures|config)\.",
        r"src.model.\1.",
    ),
    (
        r"^(base|core|factory|common|components|bottleneck|decoder|encoder|"
        r"architectures|config)/",
        r"src.model.\1/",
    ),
]

# Create backup dir if it doesn't exist
os.makedirs(BACKUP_DIR, exist_ok=True)

with open(REPORT_PATH, encoding="utf-8") as f:
    invalid_imports = json.load(f)

# Group by file
files_to_fix: dict[str, list[dict]] = {}
for entry in invalid_imports:
    fname = entry["file"]
    files_to_fix.setdefault(fname, []).append(entry)

log = []

for fname, entries in files_to_fix.items():
    abs_path = os.path.abspath(fname)
    # Backup
    backup_path = os.path.join(BACKUP_DIR, os.path.basename(fname))
    shutil.copy2(abs_path, backup_path)
    with open(abs_path, encoding="utf-8") as f:
        lines = f.readlines()
    changed = False
    for entry in entries:
        lineno = entry["line"] - 1  # 0-based
        orig_line = lines[lineno]
        new_line = orig_line
        for pat, repl in REPLACEMENTS:
            # Only replace if the pattern is at the beginning of the import
            new_line = re.sub(pat, repl, new_line)
        # Validate with AST if it still is a valid import
        try:
            node = ast.parse(new_line.strip())
            if not (
                isinstance(node.body[0], ast.Import)
                or isinstance(node.body[0], ast.ImportFrom)
            ):
                raise ValueError("Not an import")
            lines[lineno] = new_line
            changed = True
            log.append(
                {
                    "file": fname,
                    "line": lineno + 1,
                    "old": orig_line.strip(),
                    "new": new_line.strip(),
                    "status": "fixed",
                }
            )
        except Exception as e:
            log.append(
                {
                    "file": fname,
                    "line": lineno + 1,
                    "old": orig_line.strip(),
                    "new": new_line.strip(),
                    "status": f"error: {e}",
                }
            )
    if changed:
        with open(abs_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

with open(LOG_PATH, "w", encoding="utf-8") as f:
    json.dump(log, f, indent=2, ensure_ascii=False)

print(f"Autofix finished. Log in {LOG_PATH}. Backups in {BACKUP_DIR}")
