# Tutorial 02: Custom Experiment Configurations

This directory contains the experiment configurations created in
**Tutorial 02: Creating Custom Experiments (CLI Only)**.

## Reference Documentation

- **Main Tutorial**: [docs/tutorials/02_custom_experiment_cli.md](../../../docs/tutorials/02_custom_experiment_cli.md)
- **Comparison Script**: [scripts/experiments/tutorial_02_compare.py](../../../scripts/experiments/tutorial_02_compare.py)
- **Batch Execution**: [scripts/experiments/tutorial_02_batch.ps1](../../../scripts/experiments/tutorial_02_batch.ps1)

## Experiment Configurations

### 1. High Learning Rate Experiment

**File**: `high_lr_experiment.yaml`

- **Learning Rate**: 0.001 (10x higher than default)
- **Epochs**: 50
- **Batch Size**: 8
- **Purpose**: Test aggressive learning for faster convergence
- **Rationale**: Higher learning rates can speed up training but may cause instability
- **Expected Outcome**: Faster convergence, potentially lower final performance

### 2. Low Learning Rate Experiment

**File**: `low_lr_experiment.yaml`

- **Learning Rate**: 0.00001 (10x lower than default)
- **Epochs**: 100
- **Batch Size**: 16
- **Purpose**: Test conservative learning for maximum stability
- **Rationale**: Lower learning rates provide stable training but require more epochs
- **Expected Outcome**: Slow convergence, potentially better final performance

### 3. Focal Loss Experiment

**File**: `focal_loss_experiment.yaml`

- **Loss Function**: Focal Loss (alpha=0.25, gamma=2.0)
- **Learning Rate**: 0.0001 (default)
- **Epochs**: 75
- **Batch Size**: 12
- **Purpose**: Test focal loss for handling class imbalance
- **Rationale**: Crack segmentation has <5% positive pixels, requiring special handling
- **Expected Outcome**: Better handling of class imbalance, improved recall

### 4. DeepLabV3+ Experiment

**File**: `deeplabv3_experiment.yaml`

- **Model**: DeepLabV3+ with ResNet50 encoder
- **Learning Rate**: 0.0001 (default)
- **Epochs**: 100
- **Batch Size**: 8
- **Purpose**: Test different architecture for multi-scale feature extraction
- **Rationale**: Cracks vary from 1-5 pixels to large structural damage
- **Expected Outcome**: Better multi-scale detection, potentially higher IoU

## Configuration Structure Explanation

### Hydra Configuration Composition

Each experiment uses Hydra's composition pattern:

```yaml
defaults:
  - base          # Inherits from base configuration
  - _self_        # This file overrides base settings
```

### Key Parameters Explained

#### Learning Rate (training.learning_rate)

- **Default**: 0.0001
- **High**: 0.001 (10x higher) - Aggressive learning
- **Low**: 0.00001 (10x lower) - Conservative learning
- **Impact**: Controls how much weights are updated per step

#### Epochs (training.epochs)

- **Purpose**: Number of complete passes through the dataset
- **High LR**: Fewer epochs (50) - Fast convergence expected
- **Low LR**: More epochs (100) - Slow convergence expected
- **Medium**: Balanced epochs (75) - Standard training

#### Batch Size (data.dataloader.batch_size)

- **Purpose**: Number of samples processed together
- **Smaller**: 8 - For high LR or complex models (memory constraints)
- **Larger**: 16 - For low LR (better gradient estimates)
- **Medium**: 12 - Balanced approach

#### Loss Function (training.loss)

- **Default**: Binary Cross Entropy
- **Focal Loss**: Specialized for class imbalance
  - `alpha`: 0.25 - Weight for positive class
  - `gamma`: 2.0 - Focusing parameter for hard examples

#### Model Architecture (model)

- **Default**: U-Net with Swin Transformer
- **DeepLabV3+**: Advanced architecture with ASPP
  - `low_level_channels`: 256 - For edge/texture features
  - `aspp_channels`: 256 - For multi-scale context

## Usage

### Running Individual Experiments

```bash
# Activate environment
conda activate crackseg

# Run high learning rate experiment
python run.py --config-name base training.learning_rate=0.001 training.epochs=50 data.dataloader.batch_size=8

# Run low learning rate experiment
python run.py --config-name base training.learning_rate=0.00001 training.epochs=100 data.dataloader.batch_size=16

# Run medium learning rate experiment
python run.py --config-name base training.learning_rate=0.0001 training.epochs=75 data.dataloader.batch_size=12
```

### Running All Experiments

```bash
# Using the batch script
.\scripts\experiments\tutorial_02_batch.ps1
```

### Comparing Results

```bash
# Run comparison script
python scripts\experiments\tutorial_02_compare.py

# Run advanced visualization
python scripts\experiments\tutorial_02_visualize.py
```

## Experiment Results

The experiments generate outputs in:

- `src/crackseg/outputs/experiments/YYYYMMDD-HHMMSS-default/`

Each experiment includes:

- Training logs
- Model checkpoints
- Metrics (IoU, F1, Precision, Recall)
- Configuration snapshots

### Key Findings from Tutorial 02

Based on the experimental results:

1. **Medium Learning Rate (0.0001)** performed best with IoU=0.1701
2. **High Learning Rate (0.001)** converged faster but had lower performance
3. **Low Learning Rate (0.00001)** was stable but too slow
4. **Class imbalance** remains a challenge (Precision=1.0, Recall<0.5)

## Best Practices Applied

1. **Configuration Organization**: All tutorial configs in dedicated directory
2. **Naming Convention**: Clear, descriptive experiment names
3. **Documentation**: Comprehensive README with references
4. **Script Integration**: Automated comparison and batch execution
5. **Version Control**: Configurations tracked in git for reproducibility
6. **Comments**: Detailed explanations for each parameter

## Next Steps

After running these experiments:

1. Analyze the comparison results
2. Identify best performing configurations
3. Use insights for hyperparameter tuning
4. Consider extending with additional experiments
5. Apply lessons learned to production training

## Troubleshooting

If experiments fail:

1. Check conda environment activation
2. Verify GPU memory availability
3. Review configuration syntax
4. Check data path configurations

For more help, refer to the main tutorial documentation.
