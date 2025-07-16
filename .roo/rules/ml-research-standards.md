---
description:
globs:
alwaysApply: false
---
# Machine Learning & Research Standards

These standards ensure reproducible, systematic, and professional machine learning research and development for the crack segmentation project.

## Experiment Management and Reproducibility

- **Reproducibility Requirements:**
  - Set explicit random seeds for all stochastic processes
  - Use deterministic algorithms when possible
  - Document exact environment configurations
  - Example:
    ```python
    import torch
    import numpy as np
    import random

    def set_seed(seed: int = 42) -> None:
        """Set seeds for reproducible experiments."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    ```

- **Experiment Tracking:**
  - Log all hyperparameters, metrics, and model configurations
  - Use structured logging with timestamps and experiment IDs
  - Save model checkpoints at regular intervals
  - Example:
    ```python
    experiment_config = {
        "model_type": "swin_unet",
        "encoder": "swin_transformer_tiny",
        "decoder": "cnn_decoder",
        "bottleneck": "aspp",
        "input_size": (512, 512),
        "batch_size": 4,  # Limited by 8GB VRAM
        "learning_rate": 1e-4,
        "seed": 42,
        "dataset": "crack500_augmented"
    }
    ```

## Model Architecture and Development

- **Architecture Standards:**
  - Use Abstract Base Classes for modular components (Encoder, Decoder, Bottleneck)
  - Implement consistent interfaces across architecture variants
  - Document architectural decisions and trade-offs
  - Example:
    ```python
    from abc import ABC, abstractmethod
    import torch.nn as nn

    class Encoder(ABC):
        """Abstract base class for encoder architectures."""

        @abstractmethod
        def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
            """Forward pass returning skip connection features."""
            pass

        @abstractmethod
        def get_output_channels(self) -> List[int]:
            """Return output channels for each skip connection level."""
            pass
    ```

- **Model Implementation Guidelines:**
  - Include comprehensive type annotations for all model components
  - Add docstrings explaining architecture variants and use cases
  - Implement proper input validation for tensor shapes and channels
  - Example:
    ```python
    class SwinUNet(nn.Module):
        """U-Net with Swin Transformer encoder for crack segmentation.

        Architecture: SwinV2-Tiny Encoder + ASPP Bottleneck + CNN Decoder
        Input: RGB images (3, 512, 512)
        Output: Binary segmentation masks (1, 512, 512)
        """

        def __init__(self, num_classes: int = 1, encoder_name: str = "swin_tiny") -> None:
            super().__init__()
            if encoder_name not in ["swin_tiny", "swin_small"]:
                raise ValueError(f"Unsupported encoder: {encoder_name}")
            # Implementation...
    ```

## Data Management and Pipeline Standards

- **Dataset Handling:**
  - Implement consistent data loading and preprocessing pipelines
  - Use reproducible data splits with documented strategies
  - Apply systematic data augmentation with clear documentation
  - Example:
    ```python
    class CrackDataset(Dataset):
        """Standard dataset class for crack segmentation.

        Supports: SUT, Crack500, CFD, DeepCrack, custom datasets
        Standard resolution: 512x512
        """

        def __init__(
            self,
            dataset_name: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            seed: int = 42
        ) -> None:
            self.dataset_name = dataset_name
            self.split = split
            self._validate_dataset_name()
            self._load_split_indices(seed)
    ```

- **Data Augmentation Standards:**
  - Document augmentation strategies and their impact on performance
  - Use consistent augmentation pipelines across experiments
  - Validate augmentations don't break data integrity
  - Example:
    ```python
    # Standard augmentation pipeline for crack segmentation
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    ```

## Performance Optimization and Resource Management

- **VRAM Management (RTX 3070 Ti - 8GB Constraints):**
  - Implement gradient accumulation for effective larger batch sizes
  - Use mixed precision training (AMP) for memory efficiency
  - Monitor and log GPU memory usage
  - Example:
    ```python
    # Gradient accumulation for 8GB VRAM constraint
    accumulation_steps = 4  # Effective batch size = batch_size * accumulation_steps
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision

    for i, (images, masks) in enumerate(dataloader):
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, masks) / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    ```

- **Memory Optimization:**
  - Clear GPU cache between experiments
  - Use efficient data loading with appropriate num_workers
  - Implement checkpointing for long training runs
  - Example:
    ```python
    # Memory management utilities
    def clear_gpu_memory() -> None:
        """Clear GPU memory between experiments."""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def log_gpu_memory(stage: str) -> None:
        """Log current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3  # GB
            print(f"[{stage}] GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
    ```

## Evaluation and Metrics Standards

- **Evaluation Protocol:**
  - Use consistent evaluation metrics across all experiments
  - Implement proper cross-validation strategies
  - Report confidence intervals and statistical significance
  - Example:
    ```python
    def evaluate_model(model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """Standard evaluation protocol for crack segmentation."""
        model.eval()
        total_iou = 0.0
        total_f1 = 0.0
        total_samples = 0

        with torch.no_grad():
            for images, masks in dataloader:
                outputs = model(images)
                predictions = (torch.sigmoid(outputs) > 0.5).float()

                iou = calculate_iou(predictions, masks)
                f1 = calculate_f1_score(predictions, masks)

                total_iou += iou * images.size(0)
                total_f1 += f1 * images.size(0)
                total_samples += images.size(0)

        return {
            "iou": total_iou / total_samples,
            "f1": total_f1 / total_samples
        }
    ```

- **SOTA Comparison Standards:**
  - Maintain standardized comparison tables with literature
  - Use identical evaluation protocols for fair comparison
  - Document dataset-specific results and preprocessing differences
  - Example format:
    ```python
    # SOTA comparison tracking
    results_table = {
        "model": "SwinV2-Tiny U-Net",
        "dataset": "Crack500",
        "iou": 0.847,
        "f1": 0.912,
        "params": "28.3M",
        "vram_usage": "6.2GB",
        "training_time": "4.5h",
        "reference": "Our implementation"
    }
    ```

## Research Documentation and Reporting

- **Experiment Documentation:**
  - Maintain detailed experiment logs with configurations and results
  - Document architectural decisions and their performance impact
  - Create reproducible experiment scripts
  - Example structure:
    ```python
    # experiments/swin_unet_experiment.py
    """
    Experiment: SwinV2-Tiny U-Net for Crack Segmentation

    Architecture:
    - Encoder: Swin Transformer V2 Tiny
    - Bottleneck: ASPP (Atrous Spatial Pyramid Pooling)
    - Decoder: CNN with skip connections

    Hypothesis: Swin Transformer's hierarchical attention will improve
    long-range dependency modeling for crack detection.

    Expected SOTA: IoU > 0.80 on Crack500 dataset
    """
    ```

- **Model Registry:**
  - Maintain a registry of all trained models with metadata
  - Version control model checkpoints and configurations
  - Document model lineage and experiment history
  - Example:
    ```python
    model_registry = {
        "swin_unet_v1": {
            "checkpoint_path": "checkpoints/swin_unet_v1_best.pth",
            "config_path": "configs/swin_unet_v1.yaml",
            "performance": {"iou": 0.823, "f1": 0.891},
            "dataset": "crack500_augmented",
            "training_date": "2024-01-15",
            "notes": "First implementation, baseline performance"
        }
    }
    ```

## Integration with Project Workflow

- **Code Quality Integration:**
  - Follow [coding-preferences.md](mdc:.roo/rules/coding-preferences.md) for all ML code
  - Apply type annotations to all model components and data loaders
  - Use Black, Ruff, and basedpyright for ML codebase
  - Example pre-commit for ML:
    ```bash
    # ML-specific quality checks
    black src/model/ src/training/ src/evaluation/
    ruff src/model/ src/training/ src/evaluation/ --fix
    basedpyright src/model/ src/training/ src/evaluation/
    pytest tests/test_model/ --cov=src/model
    ```

- **Testing Standards:**
  - Follow [testing-standards.md](mdc:.roo/rules/testing-standards.md) for ML components
  - Test model architectures with synthetic data
  - Validate data pipeline integrity
  - Example:
    ```python
    def test_swin_unet_forward_pass():
        """Test SwinUNet forward pass with expected input/output shapes."""
        model = SwinUNet(num_classes=1)
        batch_size, channels, height, width = 2, 3, 512, 512
        input_tensor = torch.randn(batch_size, channels, height, width)

        output = model(input_tensor)

        assert output.shape == (batch_size, 1, height, width)
        assert not torch.isnan(output).any()
    ```

## Research Collaboration and Version Control

- **Git Integration:**
  - Follow [git-standards.md](mdc:.roo/rules/git-standards.md) for research code
  - Use descriptive commit messages for experiments
  - Tag significant model versions and paper submissions
  - Example:
    ```bash
    # Research-specific commit format
    git commit -m "feat(model): Implement SwinV2-Tiny encoder with ASPP bottleneck

    - Add hierarchical attention mechanism for long-range dependencies
    - Integrate ASPP for multi-scale feature extraction
    - Achieve IoU: 0.847 on Crack500 validation set
    - Memory usage: 6.2GB VRAM with batch_size=4"
    ```

- **Reproducibility Package:**
  - Include environment.yml with exact package versions
  - Provide scripts to reproduce all reported results
  - Document hardware requirements and constraints
  - Example:
    ```yaml
    # environment.yml for exact reproducibility
    name: crackseg-research
    dependencies:
      - python=3.9
      - pytorch=2.0.1
      - torchvision=0.15.2
      - timm=0.9.2  # For Swin Transformer
      - albumentations=1.3.1
      - tensorboard=2.13.0
    ```

## References and Integration

- **Core Development Standards:**
  - **Code Quality**: [coding-preferences.md](mdc:.roo/rules/coding-preferences.md)
  - **Testing**: [testing-standards.md](mdc:.roo/rules/testing-standards.md)
  - **Development Process**: [workflow-preferences.md](mdc:.roo/rules/workflow-preferences.md)
  - **Version Control**: [git-standards.md](mdc:.roo/rules/git-standards.md)

- **Project Context:**
  - **Project Overview**: [general-context.md](mdc:.roo/guides/general-context.md)
  - **Technical Terms**: [glossary.md](mdc:.roo/guides/glossary.md)
  - **LLM Collaboration**: [working-with-llms.md](mdc:.roo/guides/working-with-llms.md)

