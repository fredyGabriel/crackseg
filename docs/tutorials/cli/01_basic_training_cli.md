# Tutorial 1: Basic Training Workflow (CLI Only)

This tutorial guides you through running a basic training experiment using the command line
interface (CLI) only. No GUI required.

## Prerequisites

- You have successfully installed the project and its dependencies. See
  [CLEAN_INSTALLATION.md](../../guides/workflows/CLEAN_INSTALLATION.md).
- You have activated the `crackseg` conda environment.
- **CRITICAL**: You have installed the `crackseg` package in development mode.

### Installing the Package (Required)

Before running any training, you must install the `crackseg` package:

#### Option A: Using Conda Environment (Recommended)

```bash
# From the project root directory
conda activate crackseg
pip install -e . --no-deps
```

This installs the package without dependencies, using the conda packages already installed.

#### Option B: Using Pip (Alternative)

```bash
# From the project root directory
conda activate crackseg
pip install -e .
```

This installs the package with all pip dependencies.

**Note**: If you encounter OpenCV installation issues, use Option A (conda) as OpenCV is already
installed via conda.

This makes the `crackseg` package available for imports and ensures all components work correctly.

### Verifying Installation

Verify that the package is installed correctly:

```bash
conda activate crackseg
python -c "import crackseg; print('✅ CrackSeg package imported successfully')"
```

## Step 1: Explore Available Configurations

First, let's see what configurations are available:

```bash
conda activate crackseg
dir configs/
```

You should see files like `base.yaml`, `basic_verification.yaml`, etc.

## Step 2: Examine the Base Configuration

Let's look at the base configuration to understand what we're working with:

```bash
conda activate crackseg
cat configs/base.yaml
```

This shows the default training configuration including:

- Model architecture
- Training parameters
- Data settings
- Loss function
- Optimizer settings

## Step 3: Run Basic Training

**IMPORTANT**: Use `run.py` instead of `src/main.py` for proper execution:

```bash
conda activate crackseg
python run.py --config-name base
```

**Note**: The tutorial previously used `src/main.py`, but `run.py` is the correct entry point
that handles PYTHONPATH configuration and provides better error handling. We now use `base.yaml`
which has been corrected and works properly.

### What Happens During Training

The training process will:

1. **Load Configuration**: Hydra loads the specified configuration file
2. **Setup Environment**: Creates output directories, sets up logging
3. **Load Data**: Prepares training and validation datasets
4. **Initialize Model**: Creates the neural network architecture
5. **Start Training Loop**:
   - Forward pass through the model
   - Calculate loss
   - Backward pass (gradient computation)
   - Update model parameters
   - Log metrics
6. **Validation**: Periodically evaluate on validation set
7. **Checkpointing**: Save model checkpoints

### Monitoring Training Progress

During training, you'll see output like:

```bash
[2024-01-15 10:30:15] INFO - Starting training...
[2024-01-15 10:30:16] INFO - Epoch 1/100 - Loss: 0.2345 - IoU: 0.1234
[2024-01-15 10:30:45] INFO - Validation - Loss: 0.2123 - IoU: 0.1456
[2024-01-15 10:31:15] INFO - Saved checkpoint to artifacts/experiments/checkpoints/epoch_1.pt
```

## Step 4: Monitor Training in Real-Time

### View Live Logs

To monitor training progress in real-time:

```bash
conda activate crackseg
# In a separate terminal window
tail -f artifacts/experiments/training.log
```

### Check Training Status

To see if training is still running:

```bash
conda activate crackseg
ps aux | grep python
```

## Step 5: View Results

Once training completes, examine the results:

### Check Output Directory Structure

```bash
conda activate crackseg
dir artifacts/experiments/
```

You should see a timestamped experiment directory (e.g., `20250723-003829-default/`) containing:

- `checkpoints/` - Model checkpoints (checkpoint_last.pth, model_best.pth.tar)
- `logs/` - Training logs
- `configurations/` - Configuration snapshots
- `metrics/` - Evaluation metrics (complete_summary.json)
- `results/` - Evaluation results

### View Final Metrics

```bash
conda activate crackseg
Get-Content artifacts/experiments/20250723-003829-default/metrics/complete_summary.json | ConvertFrom-Json | ConvertTo-Json -Depth 3
```

**Note**: Replace the timestamp in the path with your actual experiment timestamp.

### Check Training Logs

```bash
conda activate crackseg
Get-Content artifacts/experiments/20250723-003829-default/logs/training.log | Select-Object -Last 20
```

**Note**: Replace the timestamp in the path with your actual experiment timestamp.

## Step 6: Analyze Training Results

### View Training Curves

If you have matplotlib installed, you can plot training curves:

```bash
conda activate crackseg
python -c "
import json
import matplotlib.pyplot as plt
import numpy as np

# Load training history
with open('artifacts/experiments/20250723-003829-default/metrics/complete_summary.json', 'r') as f:
    history = json.load(f)

# Plot loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot IoU
plt.subplot(1, 2, 2)
plt.plot(history['train_iou'], label='Train IoU')
plt.plot(history['val_iou'], label='Val IoU')
plt.title('Training and Validation IoU')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()

plt.tight_layout()
plt.savefig('artifacts/experiments/20250723-003829-default/training_curves.png')
print('✅ Training curves saved to training_curves.png')
"
```

## Advanced CLI Options

### Override Configuration Parameters

You can override any configuration parameter from the command line:

```bash
conda activate crackseg
# Change learning rate
python run.py --config-name base training.learning_rate=0.001

# Change batch size
python run.py --config-name base data.dataloader.batch_size=16

# Change number of epochs
python run.py --config-name base training.epochs=50

# Multiple overrides
python run.py --config-name base training.learning_rate=0.001 data.dataloader.batch_size=16 training.epochs=50
```

### Specify Output Directory

```bash
conda activate crackseg
python run.py --config-name basic_verification hydra.run.dir=artifacts/experiments/my_custom_run
```

### Enable Debug Mode

```bash
conda activate crackseg
python run.py --config-name basic_verification hydra.verbose=true
```

## Troubleshooting

### Common Issues

1. **Import Error**: No module named 'crackseg'

    - Solution: Run `pip install -e . --no-deps` from the project root (for conda environments)
    - Alternative: Run `pip install -e .` (for pip environments)

2. **OpenCV Installation Error**
    - Solution: Use `pip install -e . --no-deps` since OpenCV is already installed via conda

3. **Configuration Error**: Could not find 'hydra/default'

    - Solution: Use `basic_verification.yaml` instead of `base.yaml` for initial testing
    - The `base.yaml` configuration has dependencies on missing Hydra configurations

4. **DataLoader Error**: Transform config missing for split: train

    - Solution: Ensure your configuration includes transform settings for all splits (train, val, test)
    - See the `basic_verification.yaml` configuration for the correct format

5. **DataLoader Error**: DataLoader.**init**() got an unexpected keyword argument 'data_root'

    - Solution: Separate data configuration from dataloader configuration
    - Data parameters (data_root, train_split, etc.) should be under `data:`
    - DataLoader parameters (batch_size, num_workers, etc.) should be under `dataloader:`

6. **CUDA Out of Memory**

    - Solution: Reduce batch size: `python run.py --config-name basic_verification dataloader.batch_size=4`

7. **Training Hangs**

    - Solution: Check if GPU is available: `nvidia-smi`
    - Alternative: Force CPU: `python run.py --config-name basic_verification training.device=cpu`

### Configuration Structure

For a working configuration, ensure you have this structure:

```yaml
# Data configuration
data:
  data_root: data/
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  image_size: [256, 256]
  seed: 42
  in_memory_cache: false
  transform:
    train:
      - name: Resize
        params:
          height: 256
          width: 256
      - name: Normalize
        params:
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
      - name: ToTensorV2
        params: {}
    val:
      # Similar structure for validation
    test:
      # Similar structure for test

# DataLoader configuration (separate from data)
dataloader:
  batch_size: 4
  num_workers: 2
  shuffle: true
  pin_memory: true
  prefetch_factor: 2

# Model configuration
model:
  _target_: crackseg.model.core.unet.BaseUNet
  # ... model parameters

# Training configuration
training:
  epochs: 2
  learning_rate: 0.001
  # ... training parameters
```

### Quality Gates

After making any changes, run the quality gates:

```bash
conda activate crackseg
black .
python -m ruff . --fix
basedpyright .
```

## What's Next?

In the next tutorial, you will learn how to create custom experiment configurations using CLI
commands and YAML files.

## Quick Reference Commands

```bash
# Basic training (CORRECTED - use run.py)
conda activate crackseg
python run.py --config-name base

# With overrides
conda activate crackseg
python run.py --config-name base training.learning_rate=0.001

# Monitor logs
conda activate crackseg
Get-Content artifacts/experiments/[TIMESTAMP]-default/logs/training.log -Wait

# Check results
conda activate crackseg
dir artifacts/experiments/
```

## Summary of Corrections Made

1. **Entry Point**: Changed from `src/main.py` to `run.py` for proper execution
2. **Configuration**: Fixed `base.yaml` to work correctly by resolving all dependency issues
3. **Module References**: Corrected all `crackseg.model` references to `crackseg.model` throughout
  the codebase
4. **Optimizer Setup**: Fixed trainer to pass `model.parameters()` instead of the model object
5. **Configuration Structure**: Separated data and dataloader configurations to prevent parameter conflicts
6. **Output Paths**: Updated all output paths to use the correct experiment directory structure
7. **PowerShell Commands**: Updated all commands to use PowerShell syntax (`dir`, `Get-Content`, etc.)
8. **Troubleshooting**: Added specific solutions for common configuration errors
9. **Examples**: Updated all command examples to use the correct entry point and configuration

### Key Fixes Applied to base.yaml

- **Hydra Configuration**: Removed problematic `hydra: default` dependency and added direct Hydra config
- **Variable Interpolation**: Fixed `${random_seed}` to `${seed}` in data configuration
- **Transform Configuration**: Added `data/transform: augmentations` to defaults
- **DataLoader Separation**: Added `data/dataloader: default` to separate DataLoader parameters
- **Module References**: Updated all `_target_` references to use `crackseg.` prefix
