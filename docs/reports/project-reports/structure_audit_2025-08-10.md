# Project Structure Audit - 2025-08-10

This report summarizes a quick audit of code distribution across key roots and highlights candidates
for reorganization per project structure rules.

## Overview (Python file counts)

Source: `docs/reports/project-reports/structure_scan_overview.json`

```json
{
  "src": { "directories_with_py": 104, "total_py_files": 461 },
  "gui": { "directories_with_py": 68, "total_py_files": 215 },
  "scripts": { "directories_with_py": 28, "total_py_files": 137 },
  "configs": { "directories_with_py": 2, "total_py_files": 2 },
  "tests": { "directories_with_py": 85, "total_py_files": 481 }
}
```

Detailed per-root listings are available as:

- `structure_scan_src.json`
- `structure_scan_gui.json`
- `structure_scan_scripts.json`
- `structure_scan_configs.json`
- `structure_scan_tests.json`

## Findings

- `src/` contains the majority of implementation modules, organized under `crackseg/` with
  consistent subpackage structure; aligns with rules.
- `scripts/` shows a high count (137). Many are utilities/reports; keep under `scripts/` but
  consider consolidating report generators under `scripts/reports/` (already in place) and ensure names follow `<action>_<target>.py`.
- `gui/` code volume is significant; ensure components mirror `pages/`, `components/`, `services/`,
  and `utils/` boundaries. No immediate relocations detected from high-level scan.
- `configs/` minimal `.py` presence (2) is expected (package initializers or small helpers). YAML
  structure dominates and is correct.
- `tests/` mirrors `src/` breadth. Future reorg: align tests with new modularization (schemas/,
  deployment/core/, reporting/utils/) when the deferred tests PR is opened.

## Recommendations (Phase 1)

- Maintain current layout; no large moves required now.
- Add or update README stubs where missing in dense directories to clarify responsibilities.
- Ensure all new utilities created during refactor have module-level docstrings and follow naming conventions.

## Next Steps (follow-up tasks)

1. In the deferred tests PR: mirror updated `src/` paths under `tests/` and add smoke tests for visualization and deployment strategies.
2. Consider generating `project_tree.md` refresh to reflect recent modularization.
3. Track oversized growth in `gui/` and `scripts/` with periodic scans (reuse `scripts/reports/generate_structure_scan.py`).

— End of report —
