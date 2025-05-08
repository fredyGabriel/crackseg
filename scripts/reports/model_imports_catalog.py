import os
import json
import ast

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                            '..'))
root_dir = os.path.join(project_root, 'src', 'model')
pyfiles_path = os.path.join(os.path.dirname(__file__), 'model_pyfiles.json')
outfile = os.path.join(os.path.dirname(__file__), 'model_imports_catalog.json')

with open(pyfiles_path, encoding='utf-8') as f:
    pyfiles = json.load(f)

catalog = []

for relpath in pyfiles:
    abspath = os.path.join(root_dir, relpath)
    with open(abspath, encoding='utf-8') as f:
        source = f.read()
    try:
        tree = ast.parse(source, filename=relpath)
    except Exception as e:
        print(f"Error parsing {relpath}: {e}")
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                catalog.append({
                    'file': relpath,
                    'line': node.lineno,
                    'type': 'import',
                    'module': alias.name,
                    'statement': f"import {alias.name}"
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module if node.module else ''
            for alias in node.names:
                catalog.append({
                    'file': relpath,
                    'line': node.lineno,
                    'type': 'from-import',
                    'module': module,
                    'statement': f"from {module} import {alias.name}"
                })

with open(outfile, 'w', encoding='utf-8') as f:
    json.dump(catalog, f, indent=2, ensure_ascii=False)

print(f"Import catalog written to {outfile}")
