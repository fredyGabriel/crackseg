"""Unit tests for the EarlyStopping utility."""

import pytest
# import numpy as np # Removed unused import

from src.utils.early_stopping import EarlyStopping

# --- Test Cases ---


def test_early_stopping_init():
    """Test initialization with default and custom values."""
    # Default init (min mode)
    stopper_min = EarlyStopping()
    assert stopper_min.patience == 7  # Corrected default patience
    assert stopper_min.mode == 'min'
    assert stopper_min.min_delta == 0.0
    assert stopper_min.counter == 0
    assert stopper_min.best_value is None
    assert stopper_min.early_stop is False

    # Custom init (max mode)
    stopper_max = EarlyStopping(patience=5, mode="max", min_delta=0.1,
                                verbose=False)
    assert stopper_max.patience == 5
    assert stopper_max.mode == 'max'
    assert stopper_max.min_delta == 0.1
    assert not stopper_max.verbose
    assert stopper_max.best_value is None


def test_early_stopping_invalid_mode():
    """Test initialization raises ValueError for invalid mode."""
    with pytest.raises(ValueError, match=r"mode 'invalid' is unknown"):
        EarlyStopping(mode="invalid")


def test_early_stopping_min_mode_improvement():
    """Test counter resets on improvement in min mode."""
    stopper = EarlyStopping(patience=3, mode="min")
    stopper(10.0)  # Initial best score
    assert stopper.counter == 0
    assert stopper.best_value == 10.0

    stopper(9.5)  # Improvement
    assert stopper.counter == 0
    assert stopper.best_value == 9.5

    stopper(9.6)  # No improvement
    assert stopper.counter == 1
    assert stopper.best_value == 9.5  # Best score remains

    stopper(9.4)  # Improvement again
    assert stopper.counter == 0
    assert stopper.best_value == 9.4


def test_early_stopping_min_mode_no_improvement_stops():
    """Test stopping occurs after patience epochs with no improvement (min)."""
    stopper = EarlyStopping(patience=3, mode="min", verbose=False)
    stopper(10.0)
    assert not stopper.early_stop
    stopper(10.1)  # No improvement
    assert stopper.counter == 1
    assert not stopper.early_stop
    stopper(10.0)  # No improvement
    assert stopper.counter == 2
    assert not stopper.early_stop
    stopper(10.0)  # No improvement - patience reached
    assert stopper.counter == 3
    assert stopper.early_stop  # Should stop now


def test_early_stopping_min_mode_min_delta():
    """Test min_delta requirement in min mode."""
    stopper = EarlyStopping(patience=3, mode="min", min_delta=0.1)
    stopper(10.0)
    assert stopper.counter == 0
    assert stopper.best_value == 10.0

    stopper(9.95)  # Improvement < min_delta
    assert stopper.counter == 1
    assert stopper.best_value == 10.0  # Best score not updated

    stopper(9.8)  # Improvement >= min_delta
    assert stopper.counter == 0
    assert stopper.best_value == 9.8


def test_early_stopping_max_mode_improvement():
    """Test counter resets on improvement in max mode."""
    stopper = EarlyStopping(patience=3, mode="max")
    stopper(0.5)   # Initial best score
    assert stopper.counter == 0
    assert stopper.best_value == 0.5

    stopper(0.6)   # Improvement
    assert stopper.counter == 0
    assert stopper.best_value == 0.6

    stopper(0.55)  # No improvement
    assert stopper.counter == 1
    assert stopper.best_value == 0.6  # Best score remains

    stopper(0.7)   # Improvement again
    assert stopper.counter == 0
    assert stopper.best_value == 0.7


def test_early_stopping_max_mode_no_improvement_stops():
    """Test stopping occurs after patience epochs with no improvement (max)."""
    stopper = EarlyStopping(patience=3, mode="max", verbose=False)
    stopper(0.5)
    assert not stopper.early_stop
    stopper(0.45)  # No improvement
    assert stopper.counter == 1
    assert not stopper.early_stop
    stopper(0.5)   # No improvement
    assert stopper.counter == 2
    assert not stopper.early_stop
    stopper(0.5)   # No improvement - patience reached
    assert stopper.counter == 3
    assert stopper.early_stop  # Should stop now


def test_early_stopping_max_mode_min_delta():
    """Test min_delta requirement in max mode."""
    # Note: min_delta is positive, but adjusted internally for max mode
    stopper = EarlyStopping(patience=3, mode="max", min_delta=0.1)
    stopper(0.5)
    assert stopper.counter == 0
    assert stopper.best_value == 0.5

    stopper(0.55)  # Improvement < min_delta
    assert stopper.counter == 1
    assert stopper.best_value == 0.5  # Best score not updated

    stopper(0.65)  # Improvement >= min_delta
    assert stopper.counter == 0
    assert stopper.best_value == 0.65


def test_early_stopping_none_metric():
    """Test step method handles None metric value gracefully."""
    stopper = EarlyStopping(patience=3)
    stopper(10.0)
    assert stopper.counter == 0
    assert not stopper.early_stop
    assert stopper.best_value == 10.0

    assert stopper(None) is False
    assert stopper.counter == 0
    assert not stopper.early_stop
    assert stopper.best_value == 10.0

    stopper(9.0)  # Should still work after None
    assert stopper.counter == 0
    assert not stopper.early_stop
    assert stopper.best_value == 9.0
