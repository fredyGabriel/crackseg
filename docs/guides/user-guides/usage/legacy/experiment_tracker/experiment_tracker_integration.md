# ExperimentTracker Integration Guide

This guide demonstrates how to integrate the ExperimentTracker with existing components
in the CrackSeg project.

## Integration with Existing Components

### With ExperimentManager

```python
from crackseg.utils.experiment import ExperimentManager, ExperimentTracker


# Create experiment manager
exp_manager = ExperimentManager(
    base_dir="outputs",
    experiment_name="my_experiment",
    config=config,
)

# Create tracker using experiment manager's directory
tracker = ExperimentTracker(
    experiment_dir=exp_manager.experiment_dir,
    experiment_id=exp_manager.experiment_id,
    experiment_name=exp_manager.experiment_name,
    config=config,
)

# Use both components together
exp_manager.save_config(config)
tracker.start_experiment()

# Training loop...
for epoch in range(100):
    # Training code...

    # Update progress
    tracker.update_training_progress(
        current_epoch=epoch,
        total_epochs=100,
        best_metrics=current_metrics,
    )

    # Save artifacts
    if epoch % 10 == 0:
        checkpoint_path = exp_manager.get_path("checkpoints") / f"epoch_{epoch}.pth"
        # Save checkpoint...
        tracker.add_artifact(
            artifact_id=f"checkpoint-epoch-{epoch}",
            artifact_type="checkpoint",
            file_path=str(checkpoint_path),
        )

tracker.complete_experiment()
```

### With ArtifactManager

```python
from crackseg.utils.artifact_manager import ArtifactManager


# Initialize artifact manager
artifact_manager = ArtifactManager(
    base_path="outputs",
    experiment_name="my_experiment",
)

# Connect with experiment tracker
tracker = ExperimentTracker(
    experiment_dir=artifact_manager.experiment_path,
    experiment_id="my_experiment",
    experiment_name="my_experiment",
)

# Use together for comprehensive artifact management
artifact_path, metadata = artifact_manager.save_model(
    model, "best_model.pth", "Best model based on validation IoU"
)

tracker.add_artifact(
    artifact_id="best-model",
    artifact_type="model",
    file_path=artifact_path,
    description=metadata.description,
    tags=metadata.tags,
)
```

## Metadata Structure

The ExperimentTracker collects comprehensive metadata:

### Basic Information

- `experiment_id`: Unique experiment identifier
- `experiment_name`: Human-readable experiment name
- `status`: Experiment status (created, running, completed, failed, aborted)
- `description`: Experiment description
- `tags`: List of tags for categorization

### Configuration Metadata

- `config_hash`: SHA-256 hash of configuration for reproducibility
- `config_summary`: Extracted key configuration parameters

### Environment Metadata

- `python_version`: Python version
- `pytorch_version`: PyTorch version
- `platform`: Platform information
- `cuda_available`: CUDA availability
- `cuda_version`: CUDA version (if available)

### Training Metadata

- `total_epochs`: Total number of training epochs
- `current_epoch`: Current training epoch
- `best_metrics`: Best metrics achieved during training
- `training_time_seconds`: Total training time

### Artifact Associations

- `artifact_ids`: List of all artifact IDs
- `checkpoint_paths`: List of checkpoint file paths
- `metric_files`: List of metric file paths
- `visualization_files`: List of visualization file paths

### Git Metadata

- `git_commit`: Git commit hash
- `git_branch`: Git branch name
- `git_dirty`: Whether working directory has uncommitted changes

### System Metadata

- `hostname`: System hostname
- `username`: Username
- `memory_gb`: Available memory in GB
- `gpu_info`: GPU information dictionary

### Timestamps

- `created_at`: Creation timestamp
- `started_at`: Experiment start timestamp
- `completed_at`: Experiment completion timestamp
- `updated_at`: Last update timestamp

## Error Handling

The ExperimentTracker includes robust error handling:

```python
# Graceful handling of corrupted metadata files
tracker = ExperimentTracker(
    experiment_dir=Path("outputs/experiments/corrupted_exp"),
    experiment_id="corrupted_exp",
    experiment_name="corrupted_experiment",
)
# Will use initial metadata if file is corrupted

# Graceful handling of save failures
tracker.update_description("New description")
# Will continue working even if save fails
```

## References

- ExperimentManager: see `src/crackseg/utils/experiment/manager.py`
- ArtifactManager: see `src/crackseg/utils/artifact_manager/core.py`
- Unit Tests: see `tests/unit/utils/`
- Project Standards: see `.cursor/rules/coding-standards.mdc`
