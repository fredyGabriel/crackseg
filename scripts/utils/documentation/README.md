# Documentation Utilities

This directory contains utilities for managing project documentation.

## Scripts

### `generate_project_tree.py`

Generates a Markdown file with the project directory structure.

- Scans the project root and writes a Markdown-formatted structure
- Optionally excludes files ignored by `.gitignore`
- Output: `docs/reports/project_tree.md`

### `catalog_documentation.py`

Catalogs and analyzes project documentation.

- Scans documentation files
- Reports documentation coverage
- Identifies missing or outdated documentation

### `organize_reports.py`

Organizes and manages project reports.

- Consolidates report files
- Maintains report structure
- Ensures consistent reporting

## Usage

```bash
# Generate project tree
python scripts/utils/documentation/generate_project_tree.py [--include-ignored]

# Catalog documentation
python scripts/utils/documentation/catalog_documentation.py

# Organize reports
python scripts/utils/documentation/organize_reports.py
```

## Purpose

These utilities help maintain:

- Project structure documentation
- Documentation coverage and quality
- Report organization and consistency
- Documentation standards compliance
