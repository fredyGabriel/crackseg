# ExperimentTracker Usage Guide

This guide provides an overview of the ExperimentTracker component and links to detailed documentation.

## Overview

The ExperimentTracker provides comprehensive experiment metadata tracking, artifact association,
and lifecycle management for the CrackSeg project. It complements the existing ExperimentManager by
focusing on metadata collection and artifact relationships.

## Documentation Structure

### 📖 Basic Usage Guide: see `docs/guides/user-guides/usage/experiment_tracker/basic_usage.md`

- Initialization and setup
- Experiment lifecycle management
- Artifact association and retrieval
- Metadata access and management
- Best practices and testing

### 🔗 Integration Guide: see `docs/guides/user-guides/usage/legacy/experiment_tracker/experiment_tracker_integration.md`

- Integration with ExperimentManager
- Integration with ArtifactManager
- Metadata structure and error handling
- References and standards

## Quick Start

```python
from pathlib import Path
from crackseg.utils.experiment.tracker import ExperimentTracker


# Initialize tracker
tracker = ExperimentTracker(
    experiment_dir=Path("outputs/experiments/my_experiment"),
    experiment_id="my_experiment",
    experiment_name="My Experiment",
)

# Start experiment
tracker.start_experiment()

# Add artifacts
tracker.add_artifact(
    artifact_id="model-best",
    artifact_type="checkpoint",
    file_path="path/to/model.pth",
)

# Complete experiment
tracker.complete_experiment()
```

## Testing

```bash
# Run all ExperimentTracker tests
python -m pytest tests/unit/utils/test_experiment_tracker*.py -v

# Run specific test modules
python -m pytest tests/unit/utils/test_experiment_metadata.py -v
python -m pytest tests/unit/utils/test_experiment_tracker_lifecycle.py -v
python -m pytest tests/unit/utils/test_experiment_tracker_artifacts.py -v
```

## Quality Gates

```bash
# Format code
black src/crackseg/utils/experiment/tracker.py
black src/crackseg/utils/experiment/tracker/
black src/crackseg/utils/experiment/metadata.py

# Lint code
python -m ruff check src/crackseg/utils/experiment/tracker.py --fix
python -m ruff check src/crackseg/utils/experiment/tracker/ --fix
python -m ruff check src/crackseg/utils/experiment/metadata.py --fix

# Type check
basedpyright src/crackseg/utils/experiment/tracker.py
basedpyright src/crackseg/utils/experiment/tracker/
basedpyright src/crackseg/utils/experiment/metadata.py
```

## References

- Source Code: see `src/crackseg/utils/experiment/tracker.py` and related modules
- Metadata: see `src/crackseg/utils/experiment/metadata.py`
- Submodules: see `src/crackseg/utils/experiment/tracker/`
- Unit Tests: see `tests/unit/utils/`
- Project Standards: see `.cursor/rules/coding-standards.mdc`
