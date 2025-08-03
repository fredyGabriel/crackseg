# Maintenance Utilities

This directory contains utilities for maintaining the CrackSeg project.

## Scripts

### `clean_workspace.py`

Cleans up temporary files, cache directories, and obsolete files in the workspace.

- Removes `__pycache__` and `.pytest_cache` directories
- Removes `.pyc` files
- Cleans log files and output directories

### `check_updates.py`

Checks for available updates for main dependencies.

- Compares current versions with latest PyPI/conda versions
- Reports outdated packages

### `audit_rules_checklist.py`

Audits all `.mdc` rule files against the rules checklist.

- Checks for key checklist items (header, structure, references, examples)
- Prints a summary report

### `validate-rule-references.py`

Validates rule cross-references in the consolidated workspace rules.

- Checks that all `mdc:` links point to existing files
- Maintains integrity of the rule system

### `verify_setup.py`

Verifies the project setup and configuration.

- Checks environment configuration
- Validates project structure

## Usage

```bash
# Clean workspace
python scripts/utils/maintenance/clean_workspace.py

# Check for updates
python scripts/utils/maintenance/check_updates.py

# Audit rules
python scripts/utils/maintenance/audit_rules_checklist.py

# Validate rule references
python scripts/utils/maintenance/validate-rule-references.py

# Verify setup
python scripts/utils/maintenance/verify_setup.py
```
