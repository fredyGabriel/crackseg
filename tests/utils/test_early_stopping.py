"""Tests for the EarlyStopping utility class."""

import pytest
import numpy as np

# Import the class to test
from src.utils.early_stopping import EarlyStopping


# --- Test Cases ---

def test_early_stopping_init():
    """Test initialization with default and custom values."""
    # Default init (min mode)
    stopper_min = EarlyStopping()
    assert stopper_min.patience == 10
    assert stopper_min.mode == "min"
    assert stopper_min.min_delta == 0
    assert stopper_min.best_score == np.inf
    assert stopper_min.counter == 0
    assert not stopper_min.early_stop

    # Custom init (max mode)
    stopper_max = EarlyStopping(
        monitor_metric="accuracy", patience=5, min_delta=0.01, mode="max"
    )
    assert stopper_max.patience == 5
    assert stopper_max.mode == "max"
    assert stopper_max.min_delta == -0.01  # Delta adjusts for mode
    assert stopper_max.best_score == -np.inf
    assert stopper_max.counter == 0
    assert not stopper_max.early_stop


def test_early_stopping_invalid_mode():
    """Test initialization raises ValueError for invalid mode."""
    with pytest.raises(ValueError):
        EarlyStopping(mode="invalid_mode")


# --- Tests for Mode 'min' ---

def test_early_stopping_min_mode_improvement():
    """Test counter resets on improvement in min mode."""
    stopper = EarlyStopping(patience=3, mode="min")
    stopper.step(10.0)  # Initial best score
    assert stopper.best_score == 10.0
    stopper.step(11.0)  # No improvement
    assert stopper.counter == 1
    stopper.step(9.0)   # Improvement
    assert stopper.best_score == 9.0
    assert stopper.counter == 0
    assert not stopper.should_stop()


def test_early_stopping_min_mode_no_improvement_stops():
    """Test stopping occurs after patience epochs with no improvement (min)."""
    stopper = EarlyStopping(patience=3, mode="min", verbose=False)
    stopper.step(10.0)
    assert not stopper.should_stop()
    stopper.step(10.1)  # No improvement 1
    assert stopper.counter == 1
    assert not stopper.should_stop()
    stopper.step(10.0)  # No improvement 2
    assert stopper.counter == 2
    assert not stopper.should_stop()
    stopper.step(10.05)  # No improvement 3 (reaches patience)
    assert stopper.counter == 3
    assert stopper.should_stop()  # Should stop now
    # Further calls should still indicate stop
    assert stopper.step(9.0)  # Improvement doesn't matter now
    assert stopper.should_stop()


def test_early_stopping_min_mode_min_delta():
    """Test min_delta requirement in min mode."""
    stopper = EarlyStopping(patience=3, mode="min", min_delta=0.1)
    stopper.step(10.0)
    stopper.step(9.95)  # Improvement less than min_delta
    assert stopper.best_score == 10.0  # Score shouldn't update
    assert stopper.counter == 1
    stopper.step(9.89)  # Improvement meets min_delta (10.0 - 0.1)
    assert stopper.best_score == 9.89
    assert stopper.counter == 0


# --- Tests for Mode 'max' ---

def test_early_stopping_max_mode_improvement():
    """Test counter resets on improvement in max mode."""
    stopper = EarlyStopping(patience=3, mode="max")
    stopper.step(0.5)   # Initial best score
    assert stopper.best_score == 0.5
    stopper.step(0.4)   # No improvement
    assert stopper.counter == 1
    stopper.step(0.6)   # Improvement
    assert stopper.best_score == 0.6
    assert stopper.counter == 0
    assert not stopper.should_stop()


def test_early_stopping_max_mode_no_improvement_stops():
    """Test stopping occurs after patience epochs with no improvement (max)."""
    stopper = EarlyStopping(patience=3, mode="max", verbose=False)
    stopper.step(0.5)
    assert not stopper.should_stop()
    stopper.step(0.45)  # No improvement 1
    assert stopper.counter == 1
    assert not stopper.should_stop()
    stopper.step(0.5)   # No improvement 2
    assert stopper.counter == 2
    assert not stopper.should_stop()
    stopper.step(0.49)  # No improvement 3 (reaches patience)
    assert stopper.counter == 3
    assert stopper.should_stop()  # Should stop now
    # Further calls should still indicate stop
    assert stopper.step(0.6)  # Improvement doesn't matter now
    assert stopper.should_stop()


def test_early_stopping_max_mode_min_delta():
    """Test min_delta requirement in max mode."""
    # Note: min_delta is positive, but adjusted internally for max mode
    stopper = EarlyStopping(patience=3, mode="max", min_delta=0.1)
    stopper.step(0.5)
    stopper.step(0.55)  # Improvement less than min_delta
    assert stopper.best_score == 0.5  # Score shouldn't update
    assert stopper.counter == 1
    stopper.step(0.61)  # Improvement meets min_delta (0.5 + 0.1)
    assert stopper.best_score == 0.61
    assert stopper.counter == 0


# --- Test Edge Cases ---

def test_early_stopping_none_metric():
    """Test step method handles None metric value gracefully."""
    stopper = EarlyStopping(patience=3)
    stopper.step(10.0)
    assert not stopper.step(None)  # Should return False and not stop
    assert stopper.counter == 0      # Counter shouldn't increment
    assert stopper.best_score == 10.0  # Best score remains
    assert not stopper.should_stop()

    # Check it doesn't stop prematurely if patience reached via Nones
    stopper.step(None)
    stopper.step(None)
    stopper.step(None)
    assert not stopper.should_stop()
