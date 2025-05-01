from dataclasses import dataclass, field
from typing import List, Optional, Any
from omegaconf import MISSING

# --- Base Schemas ---


@dataclass
class BaseLossConfig:
    _target_: str = MISSING


@dataclass
class BaseMetricConfig:
    _target_: str = MISSING
    smooth: float = 1e-6
    threshold: Optional[float] = 0.5


# --- Individual Loss Schemas ---

@dataclass
class BCELossConfig(BaseLossConfig):
    _target_: str = "src.training.losses.BCELoss"
    # Optional list[float] or path to tensor file? Let's assume list for now.
    weight: Optional[List[float]] = None


@dataclass
class DiceLossConfig(BaseLossConfig):
    _target_: str = "src.training.losses.DiceLoss"
    smooth: float = 1.0
    sigmoid: bool = True


@dataclass
class FocalLossConfig(BaseLossConfig):
    _target_: str = "src.training.losses.FocalLoss"
    alpha: float = 0.25
    gamma: float = 2.0
    sigmoid: bool = True


@dataclass
class BCEDiceLossConfig(BaseLossConfig):
    _target_: str = "src.training.losses.BCEDiceLoss"
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    smooth: float = 1.0


# --- Combined Loss Schema ---

@dataclass
class CombinedLossItemConfig:
    # Holds one loss config and its weight
    config: Any = MISSING  # Using Any to allow any BaseLossConfig subclass
    weight: float = 1.0


@dataclass
class CombinedLossConfig(BaseLossConfig):
    _target_: str = "src.training.losses.CombinedLoss"
    losses: List[CombinedLossItemConfig] = field(default_factory=list)
    # Note: Actual weights are normalized in CombinedLoss.__init__

    def __post_init__(self):
        if not self.losses:
            raise ValueError("CombinedLossConfig must have at least one loss \
defined in 'losses'.")
        if not isinstance(self.losses, list):
            # This check might be redundant due to Hydra/OmegaConf, but good
            # practice
            raise TypeError("CombinedLossConfig 'losses' field must be a list."
                            )
        total_weight = sum(item.weight for item in self.losses)
        if total_weight <= 0:
            raise ValueError("Sum of weights in CombinedLossConfig must be \
positive.")

# --- Individual Metric Schemas ---


@dataclass
class IoUScoreConfig(BaseMetricConfig):
    _target_: str = "src.training.metrics.IoUScore"


@dataclass
class PrecisionScoreConfig(BaseMetricConfig):
    _target_: str = "src.training.metrics.PrecisionScore"


@dataclass
class RecallScoreConfig(BaseMetricConfig):
    _target_: str = "src.training.metrics.RecallScore"


@dataclass
class F1ScoreConfig(BaseMetricConfig):
    _target_: str = "src.training.metrics.F1Score"


# --- Existing Schemas (Keep as they are) ---

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
    # Placeholder for loss configuration
    # Needs a default in configs/training/default.yaml
    loss: BaseLossConfig = MISSING

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
    # Replace simple list of strings with list of metric configs
    metrics: List[BaseMetricConfig] = field(default_factory=list)
    save_predictions: bool = True
    save_dir: str = "eval_outputs/"

    def __post_init__(self):
        # Validation for the list itself might be needed depending on usage
        pass


@dataclass
class RootConfig:
    """Root configuration combining all components."""
    # Hydra 1.1+ structured config approach for defaults
    defaults: List[Any] = field(default_factory=lambda: [
        {"training/loss": "bce_dice"},  # Example default loss
        "_self_"
    ])

    project_name: str = "crackseg"
    random_seed: int = 42
    output_dir: str = "outputs/"
    log_level: str = "INFO"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
