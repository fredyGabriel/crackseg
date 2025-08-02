# Tutorial 2: Creating Custom Experiments (CLI Only)

This tutorial explains how to create, modify, and run custom experiment configurations using the
command line interface (CLI) only. No GUI required.

## Prerequisites

- You have completed [Tutorial 1: Basic Training Workflow (CLI)](01_basic_training_cli.md).
- **CRITICAL**: You have installed the `crackseg` package (`pip install -e . --no-deps`).
- You have verified the installation works (`python -c "import crackseg"`).

## Step 1: Understand Configuration Structure

First, let's examine the configuration hierarchy:

```bash
conda activate crackseg
dir configs/
```

The configuration system uses Hydra's composition pattern. Let's look at the base configuration:

```bash
conda activate crackseg
cat configs/base.yaml
```

Notice the `defaults` section - this is where Hydra composes configurations from multiple files.

## Step 2: Create a Custom Configuration Directory

Create a directory for your custom configurations:

```bash
conda activate crackseg
mkdir configs/experiments/tutorial_02
```

## Step 3: Create Your First Custom Configuration

Let's create a custom configuration that modifies the learning rate. **IMPORTANT**: Use
`basic_verification` as the base instead of `base` to avoid Hydra dependency issues:

```bash
conda activate crackseg
cat > configs/experiments/tutorial_02/high_lr_experiment.yaml << 'EOF'
# Custom experiment with higher learning rate
defaults:
  - basic_verification
  - _self_

# Override training parameters
training:
  learning_rate: 0.001  # Increased from default 0.0001
  epochs: 50   # Reduced for faster experimentation

# Override dataloader parameters
dataloader:
  batch_size: 8  # Smaller batch size for higher learning rate
EOF
```

## Step 4: Run Your Custom Experiment

**IMPORTANT**: Use `run.py` instead of `src/main.py` for proper execution:

```bash
conda activate crackseg
python run.py --config-name high_lr_experiment
```

## Step 5: Create Multiple Experiment Variations

Let's create several variations to compare:

### Experiment A: Low Learning Rate

```bash
conda activate crackseg
cat > configs/experiments/tutorial_02/low_lr_experiment.yaml << 'EOF'
defaults:
  - basic_verification
  - _self_

training:
  learning_rate: 0.00001  # Very low learning rate
  epochs: 100    # More epochs for slow learning

dataloader:
  batch_size: 16  # Larger batch size
EOF
```

### Experiment B: Different Loss Function

```bash
conda activate crackseg
cat > configs/experiments/tutorial_02/focal_loss_experiment.yaml << 'EOF'
defaults:
  - basic_verification
  - _self_

training:
  loss:
    _target_: crackseg.training.losses.focal.FocalLoss
    alpha: 0.25
    gamma: 2.0
  learning_rate: 0.0001
  epochs: 75

dataloader:
  batch_size: 12
EOF
```

### Experiment C: Different Model Architecture

```bash
conda activate crackseg
cat > configs/experiments/tutorial_02/swin_unet_experiment.yaml << 'EOF'
defaults:
  - basic_verification
  - _self_

model:
  _target_: crackseg.model.core.unet.BaseUNet
  encoder:
    _target_: crackseg.model.encoder.SwinTransformerEncoder
    img_size: 256
    patch_size: 4
    in_chans: 3
    embed_dim: 96
    depths: [2, 2, 6, 2]
    num_heads: [3, 6, 12, 24]
    window_size: 7
    mlp_ratio: 4.0
    qkv_bias: true
    drop_rate: 0.0
    attn_drop_rate: 0.0
    drop_path_rate: 0.1
    norm_layer: nn.LayerNorm
    patch_norm: true
  bottleneck:
    _target_: crackseg.model.bottleneck.aspp_bottleneck.ASPPModule
    in_channels: 768  # Swin-T output channels
    output_channels: 256
    dilation_rates: [1, 6, 12, 18]
  decoder:
    _target_: crackseg.model.decoder.cnn_decoder.CNNDecoder
    in_channels: 256
    skip_channels_list: [384, 192, 96, 48]  # Swin-T skip connections
    out_channels: 1
    depth: 4

training:
  learning_rate: 0.0001
  epochs: 100

dataloader:
  batch_size: 8  # Smaller batch for larger model
EOF
```

## Step 6: Run All Experiments

Run each experiment using `run.py`:

```bash
conda activate crackseg
# Run high learning rate experiment
python run.py --config-name high_lr_experiment

# Run low learning rate experiment
python run.py --config-name low_lr_experiment

# Run focal loss experiment
python run.py --config-name focal_loss_experiment

# Run Swin-UNet experiment
python run.py --config-name swin_unet_experiment
```

## Step 7: Compare Results

After all experiments complete, compare the results:

### List All Experiment Outputs

```bash
conda activate crackseg
dir artifacts/outputs/
```

### Compare Final Metrics

```bash
conda activate crackseg
echo "=== High LR Experiment ==="
cat artifacts/outputs/high_lr_experiment/metrics/final_metrics.json

echo "=== Low LR Experiment ==="
cat artifacts/outputs/low_lr_experiment/metrics/final_metrics.json

echo "=== Focal Loss Experiment ==="
cat artifacts/outputs/focal_loss_experiment/metrics/final_metrics.json

echo "=== Swin-UNet Experiment ==="
cat artifacts/outputs/swin_unet_experiment/metrics/final_metrics.json
```

### Compare Results Using Built-in Tools

The project provides several tools for comparing experiment results:

#### Option 1: Simple Text Comparison

```bash
conda activate crackseg
python scripts/experiments/tutorial_02/tutorial_02_compare.py
```

#### Option 2: Advanced Visualization

```bash
conda activate crackseg
python scripts/experiments/tutorial_02/tutorial_02_visualize.py
```

This will create detailed visualizations including:

- Training curves comparison
- Performance radar charts
- Detailed analysis tables
- CSV export of results

#### Option 3: Generic Visualizer (Advanced)

For more control, use the generic experiment visualizer:

```bash
conda activate crackseg
python scripts/experiments/experiment_visualizer.py \
  --experiments high_lr_experiment,low_lr_experiment,focal_loss_experiment,swin_unet_experiment \
  --output-dir docs/reports/tutorial_02_analysis \
  --title "Tutorial 02: Custom Experiments Analysis"
```

#### Option 4: Auto-Discovery

Automatically find and analyze recent experiments:

```bash
conda activate crackseg
python scripts/experiments/experiment_visualizer.py --auto-find --max-experiments 5
```

## Step 8: Advanced Configuration Techniques

### Using Hydra Overrides

You can override any parameter directly from the command line:

```bash
conda activate crackseg
# Override learning rate
python run.py --config-name basic_verification training.learning_rate=0.0005

# Override multiple parameters
python run.py --config-name basic_verification training.learning_rate=0.0005 training.epochs=75 dataloader.batch_size=12

# Override nested parameters
python run.py --config-name basic_verification model.encoder.init_features=32 training.optimizer.weight_decay=0.01
```

### Creating Configuration Templates

Create reusable configuration templates:

```bash
conda activate crackseg
mkdir configs/experiments/tutorial_02/templates
cat > configs/experiments/tutorial_02/templates/fast_experiment.yaml << 'EOF'
# Template for fast experimentation
defaults:
  - basic_verification
  - _self_

training:
  epochs: 10  # Quick training
  save_frequency: 5  # Save every 5 epochs

dataloader:
  batch_size: 16  # Larger batches for speed

# Disable some features for speed
evaluation:
  save_predictions: false
  compute_metrics: true
EOF
```

### Using Configuration Groups

Create specialized configuration groups:

```bash
conda activate crackseg
mkdir configs/experiments/tutorial_02/optimizers
mkdir configs/experiments/tutorial_02/models

# Create optimizer configurations
cat > configs/experiments/tutorial_02/optimizers/adam.yaml << 'EOF'
_target_: torch.optim.Adam
lr: 0.001
weight_decay: 0.0001
EOF

cat > configs/experiments/tutorial_02/optimizers/sgd.yaml << 'EOF'
_target_: torch.optim.SGD
lr: 0.01
momentum: 0.9
weight_decay: 0.0005
EOF

# Create model configurations
cat > configs/experiments/tutorial_02/models/unet_resnet50.yaml << 'EOF'
_target_: crackseg.model.core.unet.BaseUNet
encoder:
  _target_: crackseg.model.encoder.resnet.ResNetEncoder
  backbone: resnet50
decoder:
  _target_: crackseg.model.decoder.cnn_decoder.CNNDecoder
  in_channels: 2048
  skip_channels_list: [1024, 512, 256, 64]
  out_channels: 1
  depth: 4
EOF
```

Now use these in experiments:

```bash
conda activate crackseg
python run.py --config-name basic_verification training.optimizer=adam model=unet_resnet50
```

## Step 9: Batch Experiment Execution

The project provides a built-in batch execution script for Tutorial 02 experiments:

### Using the Built-in Batch Script

```bash
conda activate crackseg
.\scripts\experiments\tutorial_02\tutorial_02_batch.ps1
```

This script will automatically run all Tutorial 02 experiments in sequence.

### Manual Batch Execution

If you prefer to run experiments manually or create custom batch scripts:

```bash
conda activate crackseg
# Run high learning rate experiment
python run.py --config-name high_lr_experiment

# Run low learning rate experiment
python run.py --config-name low_lr_experiment

# Run focal loss experiment
python run.py --config-name focal_loss_experiment

# Run Swin-UNet experiment
python run.py --config-name swin_unet_experiment
```

### Creating Custom Batch Scripts

You can create your own batch scripts following this pattern:

```powershell
# Example: custom_batch.ps1
$experiments = @(
    "high_lr_experiment",
    "low_lr_experiment",
    "focal_loss_experiment",
    "swin_unet_experiment"
)

conda activate crackseg

foreach ($exp in $experiments) {
    Write-Host "Running experiment: $exp"
    python run.py --config-name "$exp"

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ $exp completed successfully" -ForegroundColor Green
    } else {
        Write-Host " $exp failed" -ForegroundColor Red
    }

    Write-Host "---"
}

Write-Host "All experiments completed!"
```

## Step 10: Experiment Management

### Clean Up Old Experiments

```bash
conda activate crackseg
# List all experiment outputs
dir artifacts/outputs/

# Remove specific experiment
Remove-Item -Recurse -Force artifacts/outputs/high_lr_experiment

# Remove all experiments (be careful!)
# Remove-Item -Recurse -Force artifacts/outputs/*
```

### Archive Successful Experiments

```bash
conda activate crackseg
# Create archive directory
mkdir experiment_archives

# Archive successful experiment (PowerShell)
Compress-Archive -Path artifacts/outputs/high_lr_experiment/ -DestinationPath experiment_archives/high_lr_experiment_$(Get-Date -Format 'yyyyMMdd').zip
```

## Troubleshooting

### Common Issues

1. Configuration Not Found

    - Ensure the `generated_configs/` directory exists
    - Check file permissions and YAML syntax
    - Use `basic_verification` as base instead of `base`

2. Import Errors

    - Verify package installation: `python -c "import crackseg"`
    - Reinstall if needed: `pip install -e . --no-deps`

3. Experiment Fails to Start

    - Check YAML syntax: `python -c "import yaml; yaml.safe_load(open('generated_configs/my_exp.yaml'))"`
    - Verify all referenced components exist
    - Use `run.py` instead of `src/main.py`

4. DataLoader Configuration Errors

    - Separate data and dataloader configurations
    - Ensure transform configurations are complete for all splits

5. Out of Memory

    - Reduce batch size in dataloader configuration
    - Use smaller model architecture

### Quality Gates

After creating custom configurations, verify code quality:

```bash
conda activate crackseg
black .
python -m ruff . --fix
basedpyright .
```

## Experiment Analysis Tools

The project provides several tools for analyzing experiment results:

### Built-in Analysis Scripts

- **`scripts/experiments/tutorial_02/tutorial_02_compare.py`**: Simple text-based comparison
- **`scripts/experiments/tutorial_02/tutorial_02_visualize.py`**: Advanced visualization wrapper
- **`scripts/experiments/tutorial_02/tutorial_02_batch.ps1`**: Batch execution script

### Generic Analysis Tool

- **`scripts/experiments/experiment_visualizer.py`**: Reusable tool for any experiment set

### Output Structure

Analysis tools create the following output structure:

```bash
docs/reports/tutorial_02_analysis/
├── training_curves.png          # Training curves comparison
├── performance_radar.png        # Performance radar chart
└── experiment_comparison.csv    # Tabular comparison data
```

For more details, see [scripts/experiments/README.md](scripts/experiments/README.md).

## What's Next?

You now know how to create and manage custom experiments using CLI. The next tutorial covers
extending the project with custom Python components like new loss functions or model architectures.

## Quick Reference Commands

```bash
# Create custom config
conda activate crackseg
cat > configs/experiments/tutorial_02/my_exp.yaml << 'EOF'
defaults:
  - basic_verification
  - _self_
training:
  learning_rate: 0.001
EOF

# Run custom experiment
conda activate crackseg
python run.py --config-name my_exp

# Override parameters
conda activate crackseg
python run.py --config-name basic_verification training.learning_rate=0.001

# Compare results using built-in tools
conda activate crackseg
python scripts/experiments/tutorial_02/tutorial_02_compare.py
python scripts/experiments/tutorial_02/tutorial_02_visualize.py

# Use generic visualizer
conda activate crackseg
python scripts/experiments/experiment_visualizer.py --experiments my_exp --auto-find
```

## Summary of Corrections Made

1. **Entry Point**: Changed from `src/main.py` to `run.py` for proper execution
2. **Base Configuration**: Use `basic_verification` instead of `base` to avoid Hydra dependency issues
3. **PowerShell Commands**: Updated `ls` to `dir` and `rm -rf` to `Remove-Item`
4. **Configuration Structure**: Separated data and dataloader configurations
5. **Batch Scripts**: Updated to use PowerShell syntax instead of bash
6. **Archiving**: Updated to use PowerShell's `Compress-Archive` instead of `tar`
7. **Configuration Location**: Updated from `generated_configs/` to `configs/experiments/tutorial_02/`
8. **Analysis Tools**: Updated to use built-in experiment analysis tools instead of manual scripts
9. **Visualization**: Added references to generic experiment visualizer and tutorial-specific wrappers
10. **Script Organization**: Updated references to reflect new organized script structure

## Important Note: Architecture Corrections

**⚠️ DeepLabV3+ References**: The original tutorial referenced DeepLabV3+ architecture components
that were not implemented in the current project. These have been corrected to use available components:

- **DeepLabV3Plus** → **BaseUNet** (available in `crackseg.model.core.unet`)
- **ResNetEncoder** → **SwinTransformerEncoder** (available in `crackseg.model.encoder`)
- **DeepLabV3PlusDecoder** → **CNNDecoder** (available in `crackseg.model.decoder`)

The corrected configuration uses Swin-UNet architecture which is fully implemented and tested in
the project.
