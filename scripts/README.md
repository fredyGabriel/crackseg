# scripts/

This directory contains auxiliary, experimental, and example scripts for the project. The organization is designed to facilitate easy navigation and maintenance according to each script's purpose.

## Structure

- **experiments/**
  - Experimentation scripts, model tests, benchmarks, and prototypes.
  - Example: `test_swin_encoder.py`, `benchmark_aspp.py`

- **utils/**
  - Utilities and tools for the workspace or project management.
  - Example: `clean_workspace.py`, `model_summary.py`

- **reports/**
  - Generated reports, analysis files, example PRDs, and auxiliary documentation.
  - Example: `task-complexity-report.json`, `prd.txt`, `example_prd.txt`

- **examples/**
  - Integration examples, API usage, and demonstration scripts.
  - Example: `factory_registry_integration.py`

## Best Practices

- Scripts in this directory are **not** part of the project core and should **not** be imported by main modules.
- Remove `__pycache__` and temporary files regularly to keep the workspace clean.
- When adding a new script, place it in the appropriate subfolder and update this README if necessary.

---

_This organization helps keep the repository clean, professional, and easy to navigate for all contributors._
