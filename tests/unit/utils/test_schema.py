# ruff: noqa: PLR2004
import pytest

from src.utils.config.schema import (
    BCELossConfig,
    CombinedLossConfig,
    CombinedLossItemConfig,
    ConfigSchema,
    DataConfig,
    DiceLossConfig,
    F1ScoreConfig,
    IoUScoreConfig,
    LoggingConfig,
    ModelConfig,
    PrecisionScoreConfig,
    RecallScoreConfig,
    TrainingConfig,
)


def test_data_config_instantiation():
    """Test DataConfig instantiation with required fields."""
    cfg = DataConfig(train_dir="/tmp/train", val_dir="/tmp/val")
    assert cfg.train_dir == "/tmp/train"
    assert cfg.val_dir == "/tmp/val"
    assert cfg.batch_size == 32


def test_model_config_instantiation():
    """Test ModelConfig instantiation with required fields."""
    cfg = ModelConfig(
        type="unet", in_channels=3, out_channels=1, features=[16, 32, 64]
    )
    assert cfg.type == "unet"
    assert cfg.in_channels == 3
    assert cfg.out_channels == 1
    assert cfg.features == [16, 32, 64]


def test_training_config_instantiation():
    """Test TrainingConfig instantiation with nested OptimizerConfig and
    LossConfig."""
    from src.utils.config.schema import (
        LossConfig,
        MetricConfig,
        OptimizerConfig,
    )

    opt = OptimizerConfig(type="adam", lr=0.001)
    loss = LossConfig(type="bce")
    metrics = {"iou": MetricConfig(type="iou")}
    cfg = TrainingConfig(epochs=5, optimizer=opt, loss=loss, metrics=metrics)
    assert cfg.epochs == 5
    assert cfg.optimizer.type == "adam"
    assert cfg.loss.type == "bce"
    assert "iou" in cfg.metrics


def test_logging_config_defaults():
    """Test LoggingConfig uses correct defaults."""
    cfg = LoggingConfig(experiment_name="exp1")
    assert cfg.level == "INFO"
    assert cfg.log_to_file is True
    assert cfg.log_dir == "outputs"
    assert cfg.experiment_name == "exp1"


def test_config_schema_instantiation():
    """Test ConfigSchema instantiation with all required nested configs."""
    data = DataConfig(train_dir="/tmp/train", val_dir="/tmp/val")
    model = ModelConfig(
        type="unet", in_channels=3, out_channels=1, features=[16, 32, 64]
    )
    from src.utils.config.schema import (
        LossConfig,
        MetricConfig,
        OptimizerConfig,
    )

    opt = OptimizerConfig(type="adam", lr=0.001)
    loss = LossConfig(type="bce")
    metrics = {"iou": MetricConfig(type="iou")}
    training = TrainingConfig(
        epochs=5, optimizer=opt, loss=loss, metrics=metrics
    )
    logging = LoggingConfig(experiment_name="exp1")
    cfg = ConfigSchema(
        data=data, model=model, training=training, logging=logging
    )
    assert cfg.data.train_dir == "/tmp/train"
    assert cfg.model.type == "unet"
    assert cfg.training.epochs == 5
    assert cfg.logging.experiment_name == "exp1"


def test_combined_loss_config_post_init_valid():
    """Test CombinedLossConfig __post_init__ with valid losses and weights."""
    item1 = CombinedLossItemConfig(config=BCELossConfig(), weight=0.7)
    item2 = CombinedLossItemConfig(config=DiceLossConfig(), weight=0.3)
    cfg = CombinedLossConfig(losses=[item1, item2])
    assert len(cfg.losses) == 2


def test_combined_loss_config_post_init_invalid():
    """
    Test CombinedLossConfig __post_init__ raises error for invalid config.
    """
    with pytest.raises(ValueError):
        CombinedLossConfig(losses=[])
    with pytest.raises((TypeError, AttributeError)):
        CombinedLossConfig(losses="not_a_list")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CombinedLossConfig(
            losses=[CombinedLossItemConfig(config=BCELossConfig(), weight=0)]
        )


def test_metric_configs_instantiation():
    """Test instantiation of metric config dataclasses."""
    iou = IoUScoreConfig()
    prec = PrecisionScoreConfig()
    rec = RecallScoreConfig()
    f1 = F1ScoreConfig()
    assert isinstance(iou, IoUScoreConfig)
    assert isinstance(prec, PrecisionScoreConfig)
    assert isinstance(rec, RecallScoreConfig)
    assert isinstance(f1, F1ScoreConfig)
