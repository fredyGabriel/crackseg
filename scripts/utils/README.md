# Utils Directory

This directory contains organized utility scripts for the CrackSeg project, following ML project
best practices.

## Structure

```bash
scripts/utils/
├── maintenance/           # Project maintenance utilities
│   ├── clean_workspace.py
│   ├── check_updates.py
│   ├── audit_rules_checklist.py
│   ├── validate-rule-references.py
│   └── verify_setup.py
├── analysis/             # Code analysis utilities
│   └── inventory_training_imports.py
├── documentation/        # Documentation utilities
│   ├── generate_project_tree.py
│   ├── catalog_documentation.py
│   └── organize_reports.py
├── model_tools/          # ML model utilities
│   ├── model_summary.py
│   ├── unet_diagram.py
│   └── example_override.py
└── test_suite_refinement/ # Test suite utilities
    └── [test refinement scripts]
```

## Categories

### 🛠️ Maintenance

Utilities for maintaining project health and cleanliness.

### 📊 Analysis

Tools for analyzing code structure and dependencies.

### 📝 Documentation

Utilities for managing project documentation and reports.

### 🤖 Model Tools

Tools for working with ML models and architectures.

### 🧪 Test Suite Refinement

Utilities for improving and managing test suites.

## Best Practices

- **Modular Organization**: Scripts organized by purpose and functionality
- **Clear Documentation**: Each directory has its own README
- **Consistent Naming**: Descriptive names following Python conventions
- **Proper Imports**: Each package has `__init__.py` with proper exports
- **Size Limits**: All files respect the 300-line limit for maintainability

## Usage

Each subdirectory contains specific utilities. See individual README files for detailed usage instructions.

## Maintenance

- Scripts are regularly reviewed and updated
- Obsolete files are removed promptly
- New utilities are added to appropriate categories
- Documentation is kept up-to-date
