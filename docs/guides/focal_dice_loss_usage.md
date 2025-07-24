# FocalDiceLoss Usage Guide

## Overview

The `FocalDiceLoss` is a specialized loss function designed specifically for pavement crack
segmentation tasks. It combines the strengths of Focal Loss and Dice Loss to handle the extreme
class imbalance (<5% positive pixels) and thin structure detection challenges inherent in crack segmentation.

## Key Features

- **Class Imbalance Handling**: Focal Loss component focuses on hard examples and rare positive pixels
- **Segmentation Optimization**: Dice Loss component directly optimizes the segmentation metric
- **Crack-Specific Tuning**: Default parameters optimized for crack detection scenarios
- **Component Monitoring**: Ability to access individual loss components for debugging
- **Flexible Configuration**: Fully configurable weights and parameters

## Architecture

The `FocalDiceLoss` internally uses the `CombinedLoss` framework to combine:

1. **Focal Loss**: Handles class imbalance with configurable alpha and gamma parameters
2. **Dice Loss**: Optimizes segmentation overlap with smoothing for numerical stability

## Usage Examples

### Basic Usage

```python
from crackseg.training.losses import FocalDiceLoss

# Use default configuration (optimized for crack segmentation)
loss_fn = FocalDiceLoss()

# Forward pass
predictions = model(images)  # Shape: (B, 1, H, W) - logits
targets = masks              # Shape: (B, 1, H, W) - binary masks
loss = loss_fn(predictions, targets)
```

### Custom Configuration

```python
from crackseg.training.losses.focal_dice_loss import FocalDiceLoss, FocalDiceLossConfig

# Custom configuration for specific dataset characteristics
config = FocalDiceLossConfig(
    focal_weight=0.7,        # Higher weight for Focal Loss
    dice_weight=0.3,         # Lower weight for Dice Loss
    focal_alpha=0.2,         # Lower alpha for more severe imbalance
    focal_gamma=2.5,         # Higher gamma for harder focus
    dice_smooth=2.0,         # Higher smoothing for stability
)

loss_fn = FocalDiceLoss(config)
```

### YAML Configuration

```yaml
# configs/training/loss/focal_dice.yaml
_target_: crackseg.training.losses.FocalDiceLoss
config:
  _target_: crackseg.training.losses.focal_dice_loss.FocalDiceLossConfig
  focal_weight: 0.6
  dice_weight: 0.4
  focal_alpha: 0.25
  focal_gamma: 2.0
  dice_smooth: 1.0
  dice_sigmoid: true
  dice_eps: 1e-6
  total_loss_weight: 1.0
```

### Training Script Usage

```python
# In your training script
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="training")
def train(cfg: DictConfig):
    # Loss function will be automatically instantiated from config
    loss_fn = hydra.utils.instantiate(cfg.training.loss)

    for batch in dataloader:
        predictions = model(batch["images"])
        loss = loss_fn(predictions, batch["masks"])
        loss.backward()
        optimizer.step()
```

## Parameter Tuning Guide

### For Different Crack Densities

| Crack Density | focal_alpha | focal_gamma | focal_weight | dice_weight |
|---------------|-------------|-------------|--------------|-------------|
| <2% (Very sparse) | 0.15 | 3.0 | 0.7 | 0.3 |
| 2-5% (Sparse) | 0.25 | 2.0 | 0.6 | 0.4 |
| 5-10% (Moderate) | 0.35 | 1.5 | 0.5 | 0.5 |
| >10% (Dense) | 0.5 | 1.0 | 0.4 | 0.6 |

### For Different Crack Types

#### Hairline Cracks (1-2 pixels wide)

```python
config = FocalDiceLossConfig(
    focal_weight=0.7,    # Emphasize Focal Loss for thin structures
    dice_weight=0.3,
    focal_alpha=0.2,     # Lower alpha for extreme imbalance
    focal_gamma=2.5,     # Higher gamma for hard examples
    dice_smooth=2.0,     # Higher smoothing for stability
)
```

#### Structural Cracks (3-5 pixels wide)

```python
config = FocalDiceLossConfig(
    focal_weight=0.5,    # Balanced approach
    dice_weight=0.5,
    focal_alpha=0.3,     # Moderate alpha
    focal_gamma=2.0,     # Standard gamma
    dice_smooth=1.0,     # Standard smoothing
)
```

## Monitoring and Debugging

### Access Component Losses

```python
loss_fn = FocalDiceLoss()

# Get individual components for monitoring
components = loss_fn.get_component_losses(predictions, targets)

print(f"Focal Loss: {components['focal_loss'].item():.4f}")
print(f"Dice Loss: {components['dice_loss'].item():.4f}")
print(f"Total Loss: {components['total_loss'].item():.4f}")
```

### Training Monitoring

```python
# In training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        predictions = model(batch["images"])

        # Get loss and components
        loss = loss_fn(predictions, batch["masks"])
        components = loss_fn.get_component_losses(predictions, batch["masks"])

        # Log individual components
        logger.log({
            "loss/total": loss.item(),
            "loss/focal": components["focal_loss"].item(),
            "loss/dice": components["dice_loss"].item(),
        })

        loss.backward()
        optimizer.step()
```

## Best Practices

### 1. Parameter Selection

- **Start with defaults**: The default configuration is optimized for typical crack datasets
- **Tune based on data**: Adjust alpha and gamma based on your specific crack density
- **Monitor components**: Use `get_component_losses()` to understand loss behavior

### 2. Training Stability

- **Gradient clipping**: Consider using gradient clipping with this loss
- **Learning rate**: May need lower learning rate due to Focal Loss sensitivity
- **Batch size**: Ensure sufficient batch size for stable Dice Loss computation

### 3. Validation Strategy

- **Component monitoring**: Track both Focal and Dice components during validation
- **Metric correlation**: Monitor correlation between loss components and final metrics
- **Early stopping**: Use validation Dice coefficient for early stopping

### 4. Hyperparameter Tuning

```python
# Grid search example
alpha_values = [0.15, 0.25, 0.35]
gamma_values = [1.5, 2.0, 2.5]
weight_ratios = [(0.7, 0.3), (0.6, 0.4), (0.5, 0.5)]

best_config = None
best_score = 0

for alpha in alpha_values:
    for gamma in gamma_values:
        for focal_w, dice_w in weight_ratios:
            config = FocalDiceLossConfig(
                focal_alpha=alpha,
                focal_gamma=gamma,
                focal_weight=focal_w,
                dice_weight=dice_w
            )

            # Train and evaluate
            score = train_and_evaluate(config)
            if score > best_score:
                best_score = score
                best_config = config
```

## Comparison with Other Losses

| Loss Function | Class Imbalance | Thin Structures | Training Stability | Implementation |
|---------------|----------------|-----------------|-------------------|----------------|
| BCE Loss | ❌ Poor | ❌ Poor | ✅ Good | ✅ Simple |
| Dice Loss | ✅ Good | ✅ Good | ⚠️ Moderate | ✅ Simple |
| Focal Loss | ✅ Excellent | ⚠️ Moderate | ⚠️ Moderate | ✅ Simple |
| BCE + Dice | ✅ Good | ✅ Good | ✅ Good | ✅ Simple |
| **Focal + Dice** | ✅ **Excellent** | ✅ **Excellent** | ✅ **Good** | ✅ **Simple** |

## Troubleshooting

### Common Issues

1. **NaN Loss Values**
   - Check `dice_eps` parameter (increase if needed)
   - Verify input normalization
   - Check for empty masks

2. **Training Instability**
   - Reduce learning rate
   - Increase `dice_smooth` parameter
   - Use gradient clipping

3. **Poor Convergence**
   - Adjust `focal_alpha` based on class balance
   - Tune `focal_gamma` for difficulty level
   - Check data quality and annotation

### Debugging Checklist

- [ ] Input shapes are correct (B, 1, H, W)
- [ ] Targets are binary (0 or 1)
- [ ] Predictions are logits (not probabilities)
- [ ] No NaN or Inf values in inputs
- [ ] Loss components are reasonable
- [ ] Gradients are flowing properly

## Integration with Hydra

The loss function integrates seamlessly with Hydra configuration system:

```yaml
# configs/training/default.yaml
training:
  loss: ${training.loss.focal_dice}
  batch_size: 4
  learning_rate: 0.0001
```

```bash
# Command line usage
python train.py training.loss=focal_dice
python train.py training.loss=focal_dice config.focal_alpha=0.3
```

## References

- **Focal Loss**: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- **Dice Loss**: [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797)
- **Crack Segmentation**: [DeepCrack: Learning Hierarchical Convolutional Features for Crack Detection](https://arxiv.org/abs/1809.07091)
