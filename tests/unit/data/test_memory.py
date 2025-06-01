import pytest
import torch
from torch import nn
from torch.cuda.amp import GradScaler

from src.data.memory import (
    calculate_gradient_accumulation_steps,
    enable_mixed_precision,
    estimate_batch_size,
    format_memory_stats,
    get_available_gpu_memory,
    get_gpu_memory_usage,
    memory_summary,
)


# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(
        self, input_size: int = 3, hidden_size: int = 64, num_classes: int = 1
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(hidden_size, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def test_memory_monitoring_functions():
    """Test that memory monitoring functions run without errors."""
    mem = get_available_gpu_memory()
    assert isinstance(mem, float)

    usage = get_gpu_memory_usage()
    assert isinstance(usage, dict)
    assert "allocated" in usage
    assert "reserved" in usage
    assert "total" in usage
    assert "available" in usage


def test_memory_stats_format():
    """Test formatting of memory statistics."""
    stats = {
        "allocated": 1000.0,
        "reserved": 1500.0,
        "total": 8000.0,
        "available": 7000.0,
    }
    formatted = format_memory_stats(stats)
    assert isinstance(formatted, str)
    assert "GPU Memory" in formatted
    assert "1000.0/8000.0" in formatted
    assert "Available: 7000.0" in formatted


def test_memory_summary():
    """Test memory summary function with and without model."""
    summary = memory_summary()
    assert isinstance(summary, str)

    model = SimpleModel()
    summary_with_model = memory_summary(model)
    assert isinstance(summary_with_model, str)
    assert "Model size" in summary_with_model
    assert "Parameters" in summary_with_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_estimate_batch_size():
    """Test batch size estimation."""
    model = SimpleModel()
    input_shape = (3, 224, 224)

    # Llamada adaptada a la firma real de estimate_batch_size
    bs = estimate_batch_size(model, input_shape)  # type: ignore

    assert bs >= 1
    assert isinstance(bs, int)


def test_gradient_accumulation_calculation():
    """Test calculation of gradient accumulation steps."""
    assert calculate_gradient_accumulation_steps(32, 32) == 1
    assert calculate_gradient_accumulation_steps(16, 32) == 1
    assert calculate_gradient_accumulation_steps(32, 8) == 4  # noqa: PLR2004
    assert calculate_gradient_accumulation_steps(100, 32) == 4  # noqa: PLR2004
    assert calculate_gradient_accumulation_steps(33, 32) == 2  # noqa: PLR2004


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mixed_precision():
    """Test mixed precision utility."""
    scaler = enable_mixed_precision()

    if torch.cuda.is_available():
        assert scaler is not None
        assert isinstance(scaler, GradScaler)
