from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Configuration for data loading and splitting."""
    data_root: str = "data/"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    image_size: List[int] = field(default_factory=lambda: [512, 512])
    batch_size: int = 8
    num_workers: int = 4

    def __post_init__(self):
        total = self.train_split + self.val_split + self.test_split
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(
                f"Splits must sum to 1.0 (got {total})"
            )
        if any(s <= 0 for s in self.image_size):
            raise ValueError(
                f"Image size values must be positive (got {self.image_size})"
            )
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.num_workers < 0:
            raise ValueError("Num workers must be non-negative")


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    model_name: str = "unet"
    input_channels: int = 3
    output_channels: int = 1
    encoder_name: str = "resnet34"
    pretrained: bool = True

    def __post_init__(self):
        if self.input_channels <= 0 or self.output_channels <= 0:
            raise ValueError("Input and output channels must be positive")


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    epochs: int = 50
    learning_rate: float = 0.001
    optimizer: str = "adam"
    weight_decay: float = 0.0001
    scheduler: str = "step_lr"
    step_size: int = 10
    gamma: float = 0.5

    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError("Epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("Weight decay must be non-negative")
        if self.step_size <= 0:
            raise ValueError("Step size must be positive")
        if not (0 < self.gamma <= 1):
            raise ValueError("Gamma must be in (0, 1]")


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics and output."""
    metrics: List[str] = field(
        default_factory=lambda: ["iou", "f1", "accuracy", "recall"]
    )
    save_predictions: bool = True
    save_dir: str = "eval_outputs/"

    def __post_init__(self):
        if not self.metrics:
            raise ValueError(
                "At least one evaluation metric must be specified"
            )


@dataclass
class RootConfig:
    """Root configuration combining all components."""
    project_name: str = "crackseg"
    random_seed: int = 42
    output_dir: str = "outputs/"
    log_level: str = "INFO"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
