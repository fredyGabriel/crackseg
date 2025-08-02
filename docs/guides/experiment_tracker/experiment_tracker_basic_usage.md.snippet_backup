# ExperimentTracker Basic Usage

This guide demonstrates basic usage of the ExperimentTracker component for experiment
metadata tracking and artifact association.

## Overview

The ExperimentTracker provides detailed experiment metadata tracking, artifact association, and
lifecycle management. It complements the existing ExperimentManager by focusing on metadata
collection and artifact relationships.

## Basic Usage

### Initialization

```python
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from crackseg.utils.experiment.tracker import ExperimentTracker

# Create configuration
config_dict = {
    "training": {
        "epochs": 100,
        "batch_size": 16,
        "learning_rate": 0.001,
    },
    "model": {
        "_target_": "src.model.UNet",
    },
}
config = OmegaConf.create(config_dict)

# Initialize tracker
tracker = ExperimentTracker(
    experiment_dir=Path("outputs/experiments/20241201-120000-my_experiment"),
    experiment_id="20241201-120000-my_experiment",
    experiment_name="my_experiment",
    config=config,
    auto_save=True,
)
```

### Experiment Lifecycle Management

```python
# Start experiment
tracker.start_experiment()

# Update training progress
tracker.update_training_progress(
    current_epoch=5,
    total_epochs=100,
    best_metrics={"iou": 0.85, "dice": 0.90},
    training_time_seconds=3600.0,
)

# Complete experiment
tracker.complete_experiment()

# Or handle failures
try:
    # Training code...
    tracker.complete_experiment()
except Exception as e:
    tracker.fail_experiment(f"Training failed: {e}")
```

### Artifact Association

```python
# Associate different types of artifacts
tracker.add_artifact(
    artifact_id="model-best",
    artifact_type="checkpoint",
    file_path="outputs/experiments/exp-001/checkpoints/model_best.pth",
    description="Best model checkpoint based on validation IoU",
    tags=["best", "model", "checkpoint"],
)

tracker.add_artifact(
    artifact_id="metrics-final",
    artifact_type="metrics",
    file_path="outputs/experiments/exp-001/metrics/final_metrics.json",
    description="Final training and validation metrics",
    tags=["metrics", "final"],
)

tracker.add_artifact(
    artifact_id="viz-training",
    artifact_type="visualization",
    file_path="outputs/experiments/exp-001/results/visualizations/training_curves.png",
    description="Training loss and metric curves",
    tags=["visualization", "training"],
)
```

### Metadata Access and Management

```python
# Get experiment metadata
metadata = tracker.get_metadata()
print(f"Experiment: {metadata.experiment_name}")
print(f"Status: {metadata.status}")
print(f"Best IoU: {metadata.best_metrics.get('iou', 'N/A')}")

# Get metadata as dictionary
metadata_dict = tracker.get_metadata_dict()

# Update description
tracker.update_description("Improved U-Net with attention mechanism")

# Manage tags
tracker.add_tags(["attention", "improved"])
tracker.remove_tags(["old_tag"])

# Get experiment summary
summary = tracker.get_experiment_summary()
print(f"Artifact count: {summary['artifact_count']}")
print(f"Training time: {summary['training_time_seconds']} seconds")
```

### Artifact Retrieval

```python
# Get artifacts by type
checkpoints = tracker.get_artifacts_by_type("checkpoint")
metrics = tracker.get_artifacts_by_type("metrics")
visualizations = tracker.get_artifacts_by_type("visualization")

print(f"Checkpoints: {checkpoints}")
print(f"Metrics files: {metrics}")
print(f"Visualizations: {visualizations}")
```

## Best Practices

### 1. Initialize Early

```python
# Initialize tracker at the beginning of your experiment
tracker = ExperimentTracker(...)
tracker.start_experiment()
```

### 2. Update Progress Regularly

```python
# Update training progress in your training loop
for epoch in range(num_epochs):
    # Training code...
    tracker.update_training_progress(
        current_epoch=epoch,
        total_epochs=num_epochs,
        best_metrics=current_best_metrics,
    )
```

### 3. Associate Artifacts Immediately

```python
# Associate artifacts as soon as they're created
checkpoint_path = save_checkpoint(model, epoch)
tracker.add_artifact(
    artifact_id=f"checkpoint-{epoch}",
    artifact_type="checkpoint",
    file_path=str(checkpoint_path),
)
```

### 4. Use Descriptive Tags

```python
# Use meaningful tags for better organization
tracker.add_tags(["attention", "improved", "final"])
```

### 5. Handle Failures Gracefully

```python
try:
    # Training code...
    tracker.complete_experiment()
except Exception as e:
    tracker.fail_experiment(f"Training failed: {e}")
    raise
```

## Testing

The ExperimentTracker includes comprehensive unit tests covering all functionality:

```bash
# Run ExperimentTracker tests
python -m pytest tests/unit/utils/test_experiment_tracker.py -v

# Run metadata tests
python -m pytest tests/unit/utils/test_experiment_metadata.py -v
```

## Quality Gates

The ExperimentTracker code passes all quality gates:

```bash
# Format code
black src/crackseg/utils/experiment/tracker.py
black src/crackseg/utils/experiment/metadata.py

# Lint code
python -m ruff check src/crackseg/utils/experiment/tracker.py --fix
python -m ruff check src/crackseg/utils/experiment/metadata.py --fix

# Type check
basedpyright src/crackseg/utils/experiment/tracker.py
basedpyright src/crackseg/utils/experiment/metadata.py
```
