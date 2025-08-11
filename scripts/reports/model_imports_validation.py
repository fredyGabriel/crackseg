"""
Model import s validation. This script validates import statements in
the model catalog and identifies invalid import patterns that should
be updated.
"""

import os
import re
from typing import Any

from scripts.utils.common.io_utils import read_json, write_json  # noqa: E402

# Type definitions
type ImportEntry = dict[str, Any]
type ImportCatalog = list[ImportEntry]
type InvalidEntry = dict[str, Any]
type InvalidImports = list[InvalidEntry]

catalog_path = os.path.join(
    os.path.dirname(__file__), "model_import s_catalog.json"
)
outfile = os.path.join(
    os.path.dirname(__file__), "model_import s_invalid.json"
)

catalog: ImportCatalog = read_json(catalog_path)

invalid: InvalidImports = []

# Patrones válidos para import s internos
valid_patterns = [
    r"^src\.model\.",
    r"^src/model/",
    r"^\.",  # relative import s
]

# Patrones inválidos (antiguos)
invalid_patterns = [
    r"^(base|core|factory|common|components|bottleneck|decoder|encoder|\
architectures|config)\.",
    r"^(base|core|factory|common|components|bottleneck|decoder|encoder|\
architectures|config)/",
]

for entry in catalog:
    module = entry["module"]
    if not module:
        continue
    # Solo analizar import s internos (no stdlib ni externos)
    if any(
        module.startswith(p)
        for p in [
            "src.model",
            "src/model",
            ".",
            "torch",
            "typing",
            "os",
            "json",
            "ast",
            "logging",
            "warnings",
            "pytest",
            "unittest",
            "hydra",
            "omegaconf",
        ]
    ):
        # Buscar patrones inválidos
        for pat in invalid_patterns:
            if re.match(pat, module):
                invalid.append(
                    {
                        "file": entry["file"],
                        "line": entry["line"],
                        "statement": entry["statement"],
                        "error": (
                            "Old-style import  path (should use src.model.* "
                            "or relative import )"
                        ),
                    }
                )
                break

write_json(outfile, invalid, indent=2, ensure_ascii=False)

print(f"Invalid import s report written to {outfile}")
