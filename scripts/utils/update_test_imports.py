import glob
import os
import shutil

# ruff: noqa: E501
IMPORT_MAP = {
    "from crackseg.model.hybrid_registry import": "from crackseg.model.factory.hybrid_registry import",
    "from crackseg.model.registry import": "from crackseg.model.factory.registry import",
    "from crackseg.model.factory_utils import": "from crackseg.model.factory.factory_utils import",
}

TESTS_ROOT = "tests"
BACKUP_EXT = ".bak"
LOG_FILE = "update_test_imports.log"


def update_imports_in_file(filepath: str, import_map: dict[str, str]) -> bool:
    changed = False
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        new_line = line
        for old, new in import_map.items():
            if old in new_line:
                new_line = new_line.replace(old, new)
                changed = True
        new_lines.append(new_line)

    if changed:
        # Backup original file
        shutil.copy2(filepath, filepath + BACKUP_EXT)
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    return changed


def main():
    log_entries = []
    py_files = glob.glob(
        os.path.join(TESTS_ROOT, "**", "*.py"), recursive=True
    )
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
