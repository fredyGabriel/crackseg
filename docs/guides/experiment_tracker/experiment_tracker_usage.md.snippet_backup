# ExperimentTracker Usage Guide

This guide provides an overview of the ExperimentTracker component and references to detailed documentation.

## Overview

The ExperimentTracker provides comprehensive experiment metadata tracking, artifact association,
and lifecycle management for the CrackSeg project. It complements the existing ExperimentManager by
focusing on metadata collection and artifact relationships.

## Documentation Structure

### 📖 [Basic Usage Guide](experiment_tracker_basic_usage.md)

- Initialization and setup
- Experiment lifecycle management
- Artifact association
- Metadata access and management
- Best practices
- Testing and quality gates

### 🔗 [Integration Guide](experiment_tracker_integration.md)

- Integration with ExperimentManager
- Integration with ArtifactManager
- Metadata structure details
- Error handling patterns
- References to related components

## Quick Start

```python
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from crackseg.utils.experiment.tracker import ExperimentTracker

# Initialize tracker
tracker = ExperimentTracker(
    experiment_dir=Path("outputs/experiments/my_experiment"),
    experiment_id="my_experiment",
    experiment_name="my_experiment",
    config=config,
)

# Start experiment
tracker.start_experiment()

# Update progress
tracker.update_training_progress(
    current_epoch=5,
    total_epochs=100,
    best_metrics={"iou": 0.85},
)

# Add artifacts
tracker.add_artifact(
    artifact_id="model-best",
    artifact_type="checkpoint",
    file_path="path/to/model.pth",
)

# Complete experiment
tracker.complete_experiment()
```

## Key Features

- **Comprehensive Metadata**: Environment, system, Git, and training information
- **Artifact Association**: Track checkpoints, metrics, and visualizations
- **Lifecycle Management**: Start, complete, fail, and abort experiments
- **Configuration Tracking**: Hash-based reproducibility
- **Error Handling**: Graceful handling of file corruption and save failures

## Testing

```bash
# Run all ExperimentTracker tests
python -m pytest tests/unit/utils/test_experiment_tracker.py -v
python -m pytest tests/unit/utils/test_experiment_metadata.py -v
```

## Quality Gates

```bash
# Format and lint
black src/crackseg/utils/experiment/
python -m ruff check src/crackseg/utils/experiment/ --fix

# Type check
basedpyright src/crackseg/utils/experiment/
```

## References

- **Source Code**: [tracker.py](src/crackseg/utils/experiment/tracker.py)
- **Metadata**: [metadata.py](src/crackseg/utils/experiment/metadata.py)
- **Unit Tests**: [test_experiment_tracker.py](tests/unit/utils/test_experiment_tracker.py)
- **Project Standards**: [coding-standards.mdc](.cursor/rules/coding-standards.mdc)
