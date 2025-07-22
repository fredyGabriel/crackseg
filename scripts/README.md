# scripts/

This directory contains auxiliary, experimental, and example scripts for the project. The
organization is designed to facilitate easy navigation and maintenance according to each script's purpose.

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

- **archive/**
  - Archived files no longer actively used but kept for historical reference.
  - Contains completed plans and temporary documentation.

## Key Utilities

### **clean_workspace.py** (Enhanced)

Comprehensive workspace maintenance tool combining cleanup and auditing:

```bash
# Clean workspace only
python scripts/utils/clean_workspace.py

# Clean + audit file sizes
python scripts/utils/clean_workspace.py --audit

# Audit only (no cleanup)
python scripts/utils/clean_workspace.py --audit-only

# Detailed audit with all files
python scripts/utils/clean_workspace.py --audit --verbose
```

**Features:**

- Removes cache directories (`__pycache__`, `.pytest_cache`)
- Cleans temporary files (`.pyc`, log files)
- Organizes `outputs/` directory
- **NEW**: File size auditing (300 line preferred, 400 max)
- **NEW**: Refactoring priority suggestions

### **check_updates.py**

Monitors dependency versions and suggests updates:

```bash
python scripts/utils/check_updates.py
```

## Best Practices

- Scripts in this directory are **not** part of the project core and should **not** be imported by
main modules.
- Use `clean_workspace.py --audit` regularly to monitor code quality.
- When adding a new script, place it in the appropriate subfolder and update this README if necessary.

---

_This organization helps keep the repository clean, professional, and easy to navigate for all contributors._
