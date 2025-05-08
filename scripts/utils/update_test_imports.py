import os
import glob
import shutil

# Mapeo de imports antiguos a nuevos
IMPORT_MAP = {
    "from src.model.hybrid_registry import":
        "from src.model.factory.hybrid_registry import",
    "from src.model.registry import":
        "from src.model.factory.registry import",
    "from src.model.factory_utils import":
        "from src.model.factory.factory_utils import"
}

TESTS_ROOT = "tests"
BACKUP_EXT = ".bak"
LOG_FILE = "update_test_imports.log"


def update_imports_in_file(filepath, import_map):
    changed = False
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        for old, new in import_map.items():
            if old in line:
                line = line.replace(old, new)
                changed = True
        new_lines.append(line)

    if changed:
        # Backup original file
        shutil.copy2(filepath, filepath + BACKUP_EXT)
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    return changed


def main():
    log_entries = []
    py_files = glob.glob(os.path.join(TESTS_ROOT, "**", "*.py"),
                         recursive=True)
    for filepath in py_files:
        if update_imports_in_file(filepath, IMPORT_MAP):
            log_entries.append(f"Updated: {filepath}")
    # Guardar log
    with open(LOG_FILE, "w", encoding="utf-8") as logf:
        for entry in log_entries:
            logf.write(entry + "\n")
    print(
        f"Import update complete. {len(log_entries)} files modified. "
        f"Log: {LOG_FILE}"
    )


if __name__ == "__main__":
    main()
