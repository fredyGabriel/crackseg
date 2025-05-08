import os
import json
from datetime import datetime

root_dir = os.path.join(
    os.path.dirname(__file__), '..', 'src', 'model'
)
outfile = os.path.join(
    os.path.dirname(__file__), 'model_inventory.json'
)

inventory = []

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if (
            filename.endswith('.py') or filename.endswith('.md') or
            filename == '__init__.py'
        ):
            filepath = os.path.join(dirpath, filename)
            relpath = os.path.relpath(filepath, start=root_dir)
            stat = os.stat(filepath)
            inventory.append({
                'relative_path': relpath.replace('\\', '/'),
                'size_bytes': stat.st_size,
                'last_modified': datetime.fromtimestamp(
                    stat.st_mtime
                ).isoformat()
            })

with open(outfile, 'w', encoding='utf-8') as f:
    json.dump(inventory, f, indent=2, ensure_ascii=False)

print(f"Inventory written to {outfile}")
