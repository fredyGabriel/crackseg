# Experiment Data Saving Guide

This guide explains how to use the standardized experiment data saving system to ensure
compatibility with the `evaluation/` and `reporting/` modules.

## Overview

The experiment data saving system provides a standardized way to save experiment results that are
compatible with:

- **Evaluation Module**: For analyzing predictions and generating visualizations
- **Reporting Module**: For creating comprehensive experiment reports

## Problem Solved

**Before**: Each experiment script saved data in different formats, making it incompatible with
evaluation/reporting modules.

**After**: All experiments use the same standardized saving system, ensuring full compatibility.

## Structure Generated

The system creates the following directory structure:

```bash
artifacts/experiments/timestamp-experiment_name/
├── metrics/
│   ├── complete_summary.json     # Complete experiment summary
│   ├── metrics.jsonl            # Per-epoch metrics (JSONL format)
│   └── validation_metrics.json  # Validation metrics
├── logs/
│   └── training.log             # Training logs
├── checkpoints/
│   ├── best_model.pth          # Best model checkpoint
│   └── epoch_*.pth             # Epoch checkpoints
├── final_results.yaml           # Final results (backward compatibility)
└── best_model.pth              # Best model (root level)
```

## Usage

### Basic Usage

```python
from crackseg.utils.experiment_saver import save_experiment_data

# At the end of your experiment
saved_files = save_experiment_data(
    experiment_dir=Path("artifacts/experiments/my_experiment"),
    experiment_config=config,
    final_metrics=final_metrics,
    best_epoch=best_epoch,
    training_time=training_time,
    train_losses=train_losses,
    val_losses=val_losses,
    val_ious=val_ious,
    val_f1s=val_f1s,
    val_precisions=val_precisions,
    val_recalls=val_recalls,
    val_dices=val_dices,
    val_accuracies=val_accuracies,
    best_metrics=best_metrics,
    log_file="my_experiment.log",
)
```

### Integration in Trainer Class

```python
class MyTrainer:
    def __init__(self, config):
        self.config = config
        self.start_time = time.time()

        # Track metrics during training
        self.train_losses = {}
        self.val_losses = {}
        self.val_ious = {}
        # ... other metrics

    def train_epoch(self, epoch):
        # Your training logic here
        # Store metrics
        self.train_losses[epoch] = train_loss
        self.val_losses[epoch] = val_loss
        # ... store other metrics

    def save_experiment_data(self, experiment_dir, final_metrics):
        """Save experiment data using the standardized saver."""
        from crackseg.utils.experiment_saver import save_experiment_data

        training_time = time.time() - self.start_time

        return save_experiment_data(
            experiment_dir=experiment_dir,
            experiment_config=self.config,
            final_metrics=final_metrics,
            best_epoch=self.best_epoch,
            training_time=training_time,
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            val_ious=self.val_ious,
            val_f1s=self.val_f1s,
            val_precisions=self.val_precisions,
            val_recalls=self.val_recalls,
            val_dices=self.val_dices,
            val_accuracies=self.val_accuracies,
            best_metrics=self.best_metrics,
            log_file="my_experiment.log",
        )
```

## Files Generated

### 1. `metrics/complete_summary.json`

Complete experiment summary with metadata:

```json
{
  "experiment_name": "swinv2_hybrid_360x360_experiment",
  "project_name": "crack-segmentation",
  "model_type": "crackseg.model.SwinV2CnnAsppUNet",
  "dataset": "data/unified",
  "training_epochs": 100,
  "batch_size": 4,
  "learning_rate": 1e-4,
  "final_metrics": {
    "val_precision": 0.85,
    "val_recall": 0.82,
    "val_f1": 0.83,
    "val_iou": 0.71,
    "val_dice": 0.83,
    "val_accuracy": 0.95,
    "val_loss": 0.25
  },
  "best_epoch": 85,
  "total_training_time": 3600.5,
  "timestamp": "2024-01-15T10:30:00",
  "hardware_info": {
    "device": "cuda",
    "cuda_available": true,
    "gpu_name": "NVIDIA GeForce RTX 3070 Ti"
  }
}
```

### 2. `metrics/metrics.jsonl`

Per-epoch metrics in JSONL format (one JSON object per line):

```jsonl
{"epoch": 1, "train_loss": 0.65, "val_loss": 0.45, "val_iou": 0.58, ...}
{"epoch": 2, "train_loss": 0.62, "val_loss": 0.42, "val_iou": 0.61, ...}
{"epoch": 3, "train_loss": 0.59, "val_loss": 0.39, "val_iou": 0.64, ...}
```

### 3. `metrics/validation_metrics.json`

Validation metrics with final and best epoch data:

```json
{
  "final_validation": {
    "val_precision": 0.85,
    "val_recall": 0.82,
    "val_f1": 0.83,
    "val_iou": 0.71,
    "val_dice": 0.83,
    "val_accuracy": 0.95,
    "val_loss": 0.25
  },
  "best_validation": {
    "epoch": 85,
    "precision": 0.86,
    "recall": 0.83,
    "f1": 0.84,
    "iou": 0.72,
    "dice": 0.84,
    "accuracy": 0.96,
    "loss": 0.23
  }
}
```

### 4. `logs/training.log`

Copy of the training log file for debugging and analysis.

### 5. `final_results.yaml`

Final results in YAML format for backward compatibility:

```yaml
val_precision: 0.85
val_recall: 0.82
val_f1: 0.83
val_iou: 0.71
val_dice: 0.83
val_accuracy: 0.95
val_loss: 0.25
```

## Compatibility with Modules

### Evaluation Module

The evaluation module expects:

- `metrics/complete_summary.json` - For experiment metadata
- `metrics/metrics.jsonl` - For per-epoch analysis
- `best_model.pth` - For loading the trained model

### Reporting Module

The reporting module expects:

- `metrics/complete_summary.json` - For experiment summary
- `metrics/validation_metrics.json` - For performance analysis
- `logs/training.log` - For training analysis
- `final_results.yaml` - For backward compatibility

## Migration Guide

### For Existing Experiments

1. **Add the import**:

   ```python
   from crackseg.utils.experiment_saver import save_experiment_data
   ```

2. **Track metrics during training**:

   ```python
   # In your training loop
   self.train_losses[epoch] = train_loss
   self.val_losses[epoch] = val_loss
   self.val_ious[epoch] = val_iou
   # ... track other metrics
   ```

3. **Replace existing save logic**:

   ```python
   # Old way
   with open("final_results.yaml", "w") as f:
       yaml.dump(final_metrics, f)

   # New way
   saved_files = save_experiment_data(
       experiment_dir=Path(config.experiment.output_dir),
       experiment_config=config,
       final_metrics=final_metrics,
       # ... other parameters
   )
   ```

### For New Experiments

1. **Use the standardized trainer pattern** (see example above)
2. **Track all metrics during training**
3. **Call `save_experiment_data()` at the end**

## Benefits

1. **Standardization**: All experiments save data in the same format
2. **Compatibility**: Works with evaluation/ and reporting/ modules
3. **Completeness**: Saves all necessary data for analysis
4. **Backward Compatibility**: Maintains existing YAML format
5. **Extensibility**: Easy to add new metrics or data types

## Example Implementation

See `scripts/examples/experiment_saver_example.py` for complete examples of how to integrate the
experiment saver into your experiments.

## Troubleshooting

### Common Issues

1. **Missing metrics**: Ensure you track all required metrics during training
2. **Wrong file paths**: Use `Path` objects for cross-platform compatibility
3. **Import errors**: Make sure the project root is in the Python path

### Debugging

The saver provides detailed logging. Check the console output for:

- ✅ Success messages for each file saved
- ⚠️ Warnings for missing files
- ❌ Errors for failed operations

## Future Enhancements

- Support for additional metrics (e.g., learning rate, gradient norms)
- Automatic metric tracking integration
- Support for distributed training metrics
- Integration with experiment tracking systems (MLflow, Weights & Biases)
