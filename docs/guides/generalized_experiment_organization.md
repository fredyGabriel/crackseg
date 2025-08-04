# Generalized Experiment Output Organization

## Overview

The CrackSeg project now implements a **generalized experiment output organization system** that
automatically organizes all experiment outputs (metrics, configurations, checkpoints) into
timestamped folders with the format `timestamp-experiment_name`. This behavior is
**automatic and requires no manual configuration** in individual experiment scripts.

## Key Features

### ✅ **Automatic Organization**

- **No manual configuration required** in experiment scripts
- **Automatic timestamp generation** for unique experiment identification
- **Consistent folder structure** across all experiments
- **No duplication** in global folders

### ✅ **Standardized Output Structure**

```txt
artifacts/experiments/
├── 20250804-154407-swinv2_hybrid_experiment/
│   ├── checkpoints/
│   │   └── checkpoint_last.pth
│   ├── metrics/
│   │   ├── complete_summary.json
│   │   ├── summary.json
│   │   └── validation_metrics.jsonl
│   ├── configurations/
│   │   └── 20250804-154407-swinv2_hybrid_experiment/
│   │       ├── training_config.yaml
│   │       ├── config_epoch_0001.yaml
│   │       ├── config_epoch_0002.yaml
│   │       └── config_epoch_0003.yaml
│   ├── results/
│   ├── logs/
│   └── experiment_info.json
└── [other experiments...]
```

### ✅ **Automatic Detection**

The system automatically detects experiment information from configuration:

```yaml
# In experiment config
experiment:
  name: "swinv2_hybrid_experiment"
  output_dir: "artifacts/experiments/${now:%Y%m%d-%H%M%S}-swinv2_hybrid_experiment"
```

## Implementation Details

### Trainer Auto-Detection

The `Trainer` class now includes an `_auto_detect_experiment_info()` method that:

1. **Extracts experiment information** from configuration
2. **Generates timestamp** automatically
3. **Creates ExperimentManager** with correct parameters
4. **Updates checkpoint directory** to use experiment-specific path
5. **Configures MetricsManager** and StandardizedConfigStorage correctly

### Key Code Changes

#### Trainer Modification

```python
def _auto_detect_experiment_info(self):
    """Auto-detect experiment information from configuration."""
    from datetime import datetime
    from crackseg.utils.experiment.manager import ExperimentManager

    # Extract experiment information from configuration
    experiment_name = "default_experiment"
    base_dir = "artifacts"

    # Try to get experiment name from configuration
    if hasattr(self.full_cfg, 'experiment') and self.full_cfg.experiment:
        experiment_config = self.full_cfg.experiment
        if hasattr(experiment_config, 'name'):
            experiment_name = experiment_config.name

    # Create timestamp for the experiment
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_id = f"{timestamp}-{experiment_name}"

    # Create ExperimentManager with the correct information
    self.experiment_manager = ExperimentManager(
        base_dir=base_dir,
        experiment_name=experiment_name,
        config=self.full_cfg,
        create_dirs=True,
        timestamp=timestamp,
    )

    # Update checkpoint directory to use the experiment-specific directory
    self.checkpoint_dir = self.experiment_manager.get_path("checkpoints")
```

### Simplified Experiment Scripts

Experiment scripts no longer need to manually configure the experiment manager:

```python
# OLD: Manual configuration required
modified_cfg = OmegaConf.create({
    "checkpoint_dir": experiment_output_dir,
    "experiment": {
        "name": experiment_name,
        "output_dir": experiment_output_dir,
    },
    # ... complex manual configuration
})

# NEW: Automatic detection
modified_cfg = OmegaConf.create({
    "checkpoint_dir": experiment_output_dir,
    "experiment": {
        "name": experiment_name,
        "output_dir": experiment_output_dir,
    },
    "training": experiment_config.training,
    "data": experiment_config.data,
    "model": experiment_config.model,
    "evaluation": experiment_config.evaluation,
})

trainer = Trainer(
    components=training_components,
    cfg=modified_cfg,
)
```

## Benefits

### 🎯 **Consistency**

- All experiments follow the same output organization pattern
- No more inconsistent folder structures
- Standardized naming conventions

### 🎯 **Simplicity**

- No manual configuration required in experiment scripts
- Reduced boilerplate code
- Automatic timestamp generation

### 🎯 **Reliability**

- No more duplicate outputs in global folders
- Automatic experiment isolation
- Proper experiment identification

### 🎯 **Maintainability**

- Centralized logic in Trainer class
- Easy to modify behavior globally
- Clear separation of concerns

## Usage Examples

### Basic Experiment Script

```python
@hydra.main(
    version_base=None,
    config_path="../../../configs",
    config_name="experiments/my_experiment/my_experiment_config",
)
def main(cfg: DictConfig) -> None:
    # Create training components
    components = create_training_components(cfg)

    # Create configuration with experiment info
    modified_cfg = OmegaConf.create({
        "checkpoint_dir": f"artifacts/experiments/{timestamp}-{experiment_name}",
        "experiment": {
            "name": experiment_name,
            "output_dir": f"artifacts/experiments/{timestamp}-{experiment_name}",
        },
        "training": cfg.training,
        "data": cfg.data,
        "model": cfg.model,
        "evaluation": cfg.evaluation,
    })

    # Initialize trainer - automatic organization happens
    trainer = Trainer(
        components=training_components,
        cfg=modified_cfg,
    )

    # Train - all outputs automatically organized
    trainer.train()
```

### Configuration Example

```yaml
# configs/experiments/my_experiment/my_experiment_config.yaml
defaults:
  - base
  - training: default
  - data: default
  - model: default
  - evaluation: default

experiment:
  name: "my_experiment"
  output_dir: "artifacts/experiments/${now:%Y%m%d-%H%M%S}-my_experiment"

# ... rest of configuration
```

## Migration Guide

### For Existing Experiments

1. **Remove manual experiment manager configuration** from experiment scripts
2. **Simplify configuration** to only include experiment info at top level
3. **Remove manual directory creation** - handled automatically
4. **Remove manual metrics/config paths** - handled automatically

### Before (Manual Configuration)

```python
# Complex manual configuration
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_name = experiment_config.experiment.name
experiment_output_dir = f"artifacts/experiments/{timestamp}-{experiment_name}"

# Manual directory creation
experiment_path = Path(experiment_output_dir)
experiment_path.mkdir(parents=True, exist_ok=True)

# Manual subdirectory creation
metrics_dir = experiment_path / "metrics"
config_dir = experiment_path / "configurations"
metrics_dir.mkdir(exist_ok=True)
config_dir.mkdir(exist_ok=True)

# Complex configuration with manual paths
modified_cfg = OmegaConf.create({
    "checkpoint_dir": experiment_output_dir,
    "experiment": {
        "name": experiment_name,
        "output_dir": experiment_output_dir,
    },
    "experiment_id": full_experiment_id,
    "experiment_dir": experiment_output_dir,
    "metrics_dir": str(metrics_dir),
    "config_dir": str(config_dir),
    # ... rest of configuration
})

# Manual experiment manager patching
if hasattr(trainer, 'experiment_manager') and trainer.experiment_manager:
    trainer.experiment_manager.experiment_id = full_experiment_id
    trainer.experiment_manager.experiment_dir = Path(experiment_output_dir)
```

### After (Automatic Configuration)

```python
# Simple configuration
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_name = experiment_config.experiment.name
experiment_output_dir = f"artifacts/experiments/{timestamp}-{experiment_name}"

# Create the experiment directory structure
experiment_path = Path(experiment_output_dir)
experiment_path.mkdir(parents=True, exist_ok=True)

# Simple configuration
modified_cfg = OmegaConf.create({
    "checkpoint_dir": experiment_output_dir,
    "experiment": {
        "name": experiment_name,
        "output_dir": experiment_output_dir,
    },
    "training": experiment_config.training,
    "data": experiment_config.data,
    "model": experiment_config.model,
    "evaluation": experiment_config.evaluation,
})

# Automatic organization
trainer = Trainer(
    components=training_components,
    cfg=modified_cfg,
)
```

## Verification

To verify that the generalized organization is working correctly:

1. **Check experiment folder structure**:

   ```bash
   ls artifacts/experiments/
   ```

2. **Verify metrics location**:

   ```bash
   ls artifacts/experiments/[timestamp]-[experiment_name]/metrics/
   ```

3. **Verify configurations location**:

   ```bash
   ls artifacts/experiments/[timestamp]-[experiment_name]/configurations/
   ```

4. **Check for no duplication**:

   ```bash
   # Should be empty or contain only old experiments
   ls artifacts/experiments/metrics/
   ls artifacts/experiments/configurations/
   ```

## Troubleshooting

### Common Issues

1. **Metrics still in global folder**: Check that `experiment.name` is correctly set in configuration
2. **Configurations still in global folder**: Verify that `experiment.output_dir` is properly formatted
3. **Checkpoints not in experiment folder**: Ensure `checkpoint_dir` points to experiment-specific directory

### Debug Steps

1. **Check Trainer logs** for auto-detection messages:

   ```txt
   Auto-detected experiment: [timestamp]-[experiment_name]
   Experiment directory: artifacts\experiments\[timestamp]-[experiment_name]
   ```

2. **Verify configuration structure**:

   ```python
   print(cfg.experiment.name)  # Should be set
   print(cfg.experiment.output_dir)  # Should be set
   ```

3. **Check experiment manager**:

   ```python
   print(trainer.experiment_manager.experiment_id)
   print(trainer.experiment_manager.experiment_dir)
   ```

## Future Enhancements

- **Automatic experiment registry** for easier experiment discovery
- **Experiment comparison tools** for analyzing multiple experiments
- **Automatic cleanup** of old experiments
- **Integration with experiment tracking tools** (MLflow, Weights & Biases)

---

**Note**: This generalized organization system ensures that all experiments follow the same output
structure automatically, eliminating the need for manual configuration in individual experiment scripts.
