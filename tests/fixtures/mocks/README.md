# Test Mock Fixtures - CrackSeg

This directory contains mock data and fixtures used in the test suite.

## Structure

```txt
mocks/
└── experiment_manager/
    ├── mock.experiment_manager.experiment_manager.experiment_dir/
    │   ├── 2254896334992/      # Mock experiment directory with ID
    │   ├── 2254896338784/
    │   └── ...                 # Various experiment IDs
    └── mock.experiment_manager.experiment_manager.experiment_dir.__truediv__()/
        ├── 2254896338256/      # Mock path operations
        └── ...                 # Various path mock IDs
```

## Purpose

These mock fixtures simulate experiment manager behavior for testing:

- **Experiment directories**: Mock temporary experiment folders
- **Path operations**: Mock file system path operations
- **ID generation**: Consistent numeric IDs for reproducible tests

## Usage in Tests

```python
import pytest
from unittest.mock import patch
from pathlib import Path

# Example usage in test files
@patch('src.utils.experiment.ExperimentManager')
def test_experiment_creation(mock_experiment_manager):
    # Mock experiment directory is available in fixtures
    mock_dir = Path("tests/fixtures/mocks/experiment_manager/mock.experiment_manager.experiment_manager.experiment_dir/2254896334992")
    mock_experiment_manager.experiment_dir = mock_dir
    # Test implementation...
```

## Migration Notes

- **Previous location**: `MagicMock/` (project root)
- **New location**: `tests/fixtures/mocks/` (organized with tests)
- **Structure preserved**: All directory names and IDs kept for compatibility
- **Tests updated**: Test imports may need updating if they reference old paths

## Maintenance

- Add new mock fixtures to appropriate subdirectories
- Document new mock patterns in this README
- Ensure mock data remains lightweight and focused
- Remove obsolete mock data when tests are updated

---

Reorganized: January 6, 2025 - Part of project structure cleanup
