import json
import re
import os

catalog_path = os.path.join(os.path.dirname(__file__),
                            'model_imports_catalog.json')
outfile = os.path.join(os.path.dirname(__file__),
                       'model_imports_invalid.json')

with open(catalog_path, encoding='utf-8') as f:
    catalog = json.load(f)

invalid = []

# Patrones válidos para imports internos
valid_patterns = [
    r'^src\.model\.',
    r'^src/model/',
    r'^\.',  # relative imports
]

# Patrones inválidos (antiguos)
invalid_patterns = [
    r'^(base|core|factory|common|components|bottleneck|decoder|encoder|\
architectures|config)\.',
    r'^(base|core|factory|common|components|bottleneck|decoder|encoder|\
architectures|config)/',
]

for entry in catalog:
    module = entry['module']
    if not module:
        continue
    # Solo analizar imports internos (no stdlib ni externos)
    if any(module.startswith(p) for p in ['src.model', 'src/model', '.',
                                          'torch', 'typing', 'os', 'json',
                                          'ast', 'logging', 'warnings',
                                          'pytest', 'unittest', 'hydra',
                                          'omegaconf']):
        # Buscar patrones inválidos
        for pat in invalid_patterns:
            if re.match(pat, module):
                invalid.append({
                    'file': entry['file'],
                    'line': entry['line'],
                    'statement': entry['statement'],
                    'error': 'Old-style import path (should use src.model.* \
or relative import)'
                })
                break

with open(outfile, 'w', encoding='utf-8') as f:
    json.dump(invalid, f, indent=2, ensure_ascii=False)

print(f"Invalid imports report written to {outfile}")
