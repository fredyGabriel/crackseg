"""Configuration schema definitions."""

from dataclasses import dataclass, field

from omegaconf import MISSING

# --- Base Schemas ---


@dataclass
class BaseLossConfig:
    _target_: str = MISSING


@dataclass
class BaseMetricConfig:
    _target_: str = MISSING
    smooth: float = 1e-6
    threshold: float | None = 0.5


# --- Individual Loss Schemas ---


@dataclass
class BCELossConfig(BaseLossConfig):
    _target_: str = "src.training.losses.BCELoss"
    # Optional list[float] or path to tensor file? Let's assume list for now.
    weight: list[float] | None = None


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
    config: BaseLossConfig | dict[str, float | int | str | bool | None] = (
        MISSING
    )
    weight: float = 1.0


@dataclass
class CombinedLossConfig(BaseLossConfig):
    _target_: str = "src.training.losses.CombinedLoss"
    losses: list[CombinedLossItemConfig] = field(default_factory=list)
    # Note: Actual weights are normalized in CombinedLoss.__init__

    def __post_init__(self):
        # Validación relevante solo a CombinedLossConfig
        if not self.losses:
            raise ValueError(
                "CombinedLossConfig must have at least one loss defined in "
                "'losses'."
            )
        total_weight = sum(item.weight for item in self.losses)
        if total_weight <= 0:
            raise ValueError(
                "Sum of weights in CombinedLossConfig must be positive."
            )


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
    """Data configuration schema."""

    train_dir: str = MISSING
    val_dir: str = MISSING
    test_dir: str | None = None
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2


@dataclass
class ModelConfig:
    """Model configuration schema."""

    type: str = MISSING
    in_channels: int = 3
    out_channels: int = 1
    features: list[int] = MISSING


@dataclass
class OptimizerConfig:
    """Optimizer configuration schema."""

    type: str = MISSING
    lr: float = 0.001
    weight_decay: float = 0.0


@dataclass
class LossConfig:
    """Loss function configuration schema."""

    type: str = MISSING
    weight: list[float] | None = None
    reduction: str = "mean"


@dataclass
class MetricConfig:
    """Metric configuration schema."""

    type: str = MISSING
    threshold: float = 0.5


@dataclass
class TrainingConfig:
    """Training configuration schema."""

    epochs: int = 100
    optimizer: OptimizerConfig = MISSING
    loss: LossConfig = MISSING
    metrics: dict[str, MetricConfig] = field(default_factory=dict)
    early_stopping_patience: int = 7
    save_checkpoint_freq: int = 1
    validate_freq: int = 1
    log_freq: int = 10


@dataclass
class LoggingConfig:
    """Logging configuration schema."""

    level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "outputs"
    experiment_name: str = MISSING


@dataclass
class ConfigSchema:
    """Root configuration schema."""

    seed: int | None = None
    data: DataConfig = MISSING
    model: ModelConfig = MISSING
    training: TrainingConfig = MISSING
    logging: LoggingConfig = MISSING
    device: int | None = None
    debug: bool = False
