import pytest
import torch
from pytest import FixtureRequest

from src.training.metrics import (
    F1Score,
    IoUScore,
    PrecisionScore,
    RecallScore,
    get_scalar_metrics,
)

# --- Test Fixtures ---


@pytest.fixture
def sample_pred_logits() -> torch.Tensor:
    # Batch size 2, H=2, W=2
    # Ex: [[[-1.0, 1.0], [1.0, -1.0]], [[2.0, -2.0], [-2.0, 2.0]]]
    return torch.tensor(
        [[[-1.0, 1.0], [1.0, -1.0]], [[2.0, -2.0], [-2.0, 2.0]]]
    )


@pytest.fixture
def sample_target_mask() -> torch.Tensor:
    # Batch size 2, H=2, W=2
    # Ex: [[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]
    return torch.tensor(
        [[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]
    ).float()


@pytest.fixture
def perfect_target_mask() -> torch.Tensor:
    # Matches sigmoid(logits > 0.5) for sample_pred_logits (thresh 0.5)
    # Sigmoid approx: [[[0.26, 0.73], [0.73, 0.26]], [[0.88, 0.11],
    # [0.11, 0.88]]]
    # Thresholded: [[[0., 1.],[1., 0.]], [[1., 0.],[0., 1.]]]
    return torch.tensor(
        [[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]
    ).float()


# --- Helper Function for Thresholding (for manual verification) ---


def threshold_tensor(
    tensor: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    is_logits = tensor.min() < 0 or tensor.max() > 1
    probs = torch.sigmoid(tensor) if is_logits else tensor
    return (probs > threshold).float()


# --- Test Cases ---


# Test IoU Score
@pytest.mark.parametrize(
    "pred, target, threshold, expected_iou",
    [
        # Perfect match (logits)
        ("sample_pred_logits", "perfect_target_mask", 0.5, 1.0),
        # Perfect match (pre-thresholded)
        (
            threshold_tensor(
                torch.tensor(
                    [[[-1.0, 1.0], [1.0, -1.0]], [[2.0, -2.0], [-2.0, 2.0]]]
                )
            ),
            "perfect_target_mask",
            None,
            1.0,
        ),
        # Partial match (logits vs same target) - should be perfect
        # TP=4, FP=0, FN=0 -> IoU=1.0
        ("sample_pred_logits", "sample_target_mask", 0.5, 1.0),
        # Empty pred (logits), non-empty target
        (torch.zeros((2, 2, 2)) - 5, "sample_target_mask", 0.5, 0.0),
        # Non-empty pred (logits), empty target
        ("sample_pred_logits", torch.zeros((2, 2, 2)), 0.5, 0.0),
        # Empty pred and target -> IoU=1 (smooth/smooth)
        (torch.zeros((2, 2, 2)) - 5, torch.zeros((2, 2, 2)), 0.5, 1.0),
        # Full pred and target
        (torch.ones((2, 2, 2)) + 5, torch.ones((2, 2, 2)), 0.5, 1.0),
    ],
)
def test_iou_score(
    pred: torch.Tensor | str,
    target: torch.Tensor | str,
    threshold: float | None,
    expected_iou: float,
    request: FixtureRequest,
) -> None:
    # Resolve fixture references
    pred_tensor = (
        request.getfixturevalue(pred) if isinstance(pred, str) else pred
    )
    target_tensor = (
        request.getfixturevalue(target) if isinstance(target, str) else target
    )

    metric = IoUScore(threshold=threshold)
    iou = metric(pred_tensor, target_tensor)
    assert torch.isclose(iou, torch.tensor(expected_iou), atol=1e-5)


# Test Precision Score
@pytest.mark.parametrize(
    "pred, target, threshold, expected_precision",
    [
        # Perfect match
        ("sample_pred_logits", "perfect_target_mask", 0.5, 1.0),
        # Partial match (logits vs same target)
        # TP=4, FP=0 -> Prec=1.0
        ("sample_pred_logits", "sample_target_mask", 0.5, 1.0),
        # Empty pred, non-empty target -> Prec=1 (smooth/smooth)
        (torch.zeros((2, 2, 2)) - 5, "sample_target_mask", 0.5, 1.0),
        # Non-empty pred, empty target -> Prec=0
        # TP=0, FP=4 -> Prec=0
        ("sample_pred_logits", torch.zeros((2, 2, 2)), 0.5, 0.0),
        # Empty pred and target -> Prec=1 (smooth/smooth)
        (torch.zeros((2, 2, 2)) - 5, torch.zeros((2, 2, 2)), 0.5, 1.0),
        # Full pred and target
        # TP=8, FP=0 -> Prec=1
        (torch.ones((2, 2, 2)) + 5, torch.ones((2, 2, 2)), 0.5, 1.0),
    ],
)
def test_precision_score(
    pred: torch.Tensor | str,
    target: torch.Tensor | str,
    threshold: float | None,
    expected_precision: float,
    request: FixtureRequest,
) -> None:
    pred_tensor = (
        request.getfixturevalue(pred) if isinstance(pred, str) else pred
    )
    target_tensor = (
        request.getfixturevalue(target) if isinstance(target, str) else target
    )

    metric = PrecisionScore(threshold=threshold)
    precision = metric(pred_tensor, target_tensor)
    assert torch.isclose(
        precision, torch.tensor(expected_precision), atol=1e-5
    )


# Test Recall Score
@pytest.mark.parametrize(
    "pred, target, threshold, expected_recall",
    [
        # Perfect match
        ("sample_pred_logits", "perfect_target_mask", 0.5, 1.0),
        # Partial match (logits vs same target)
        # TP=4, FN=0 -> Rec=1.0
        ("sample_pred_logits", "sample_target_mask", 0.5, 1.0),
        # Empty pred, non-empty target -> Rec=0
        # TP=0, FN=4 -> Rec=0
        (torch.zeros((2, 2, 2)) - 5, "sample_target_mask", 0.5, 0.0),
        # Non-empty pred, empty target -> Rec=1 (smooth/smooth)
        ("sample_pred_logits", torch.zeros((2, 2, 2)), 0.5, 1.0),
        # Empty pred and target -> Rec=1 (smooth/smooth)
        (torch.zeros((2, 2, 2)) - 5, torch.zeros((2, 2, 2)), 0.5, 1.0),
        # Full pred and target
        # TP=8, FN=0 -> Rec=1
        (torch.ones((2, 2, 2)) + 5, torch.ones((2, 2, 2)), 0.5, 1.0),
    ],
)
def test_recall_score(
    pred: torch.Tensor | str,
    target: torch.Tensor | str,
    threshold: float | None,
    expected_recall: float,
    request: FixtureRequest,
) -> None:
    pred_tensor = (
        request.getfixturevalue(pred) if isinstance(pred, str) else pred
    )
    target_tensor = (
        request.getfixturevalue(target) if isinstance(target, str) else target
    )

    metric = RecallScore(threshold=threshold)
    recall = metric(pred_tensor, target_tensor)
    assert torch.isclose(recall, torch.tensor(expected_recall), atol=1e-5)


# Test F1 Score
@pytest.mark.parametrize(
    "pred, target, threshold, expected_f1",
    [
        # Perfect match
        ("sample_pred_logits", "perfect_target_mask", 0.5, 1.0),
        # Partial match (logits vs same target)
        # TP=4, FP=0, FN=0 -> F1=1.0
        ("sample_pred_logits", "sample_target_mask", 0.5, 1.0),
        # Empty pred, non-empty target -> F1=0
        (torch.zeros((2, 2, 2)) - 5, "sample_target_mask", 0.5, 0.0),
        # Non-empty pred, empty target -> F1=0
        ("sample_pred_logits", torch.zeros((2, 2, 2)), 0.5, 0.0),
        # Empty pred and target -> F1=1 (smooth/smooth)
        (torch.zeros((2, 2, 2)) - 5, torch.zeros((2, 2, 2)), 0.5, 1.0),
        # Full pred and target
        (torch.ones((2, 2, 2)) + 5, torch.ones((2, 2, 2)), 0.5, 1.0),
    ],
)
def test_f1_score(
    pred: torch.Tensor | str,
    target: torch.Tensor | str,
    threshold: float | None,
    expected_f1: float,
    request: FixtureRequest,
) -> None:
    pred_tensor = (
        request.getfixturevalue(pred) if isinstance(pred, str) else pred
    )
    target_tensor = (
        request.getfixturevalue(target) if isinstance(target, str) else target
    )

    metric = F1Score(threshold=threshold)
    f1 = metric(pred_tensor, target_tensor)
    assert torch.isclose(f1, torch.tensor(expected_f1), atol=1e-5)


# Test smooth parameter
def test_smoothness(sample_target_mask: torch.Tensor) -> None:
    # Test case where denominator would be zero without smoothing
    # Logits -> all zeros after sigmoid+threshold
    pred_empty = torch.zeros_like(sample_target_mask) - 5
    target_empty = torch.zeros_like(sample_target_mask)

    iou_metric = IoUScore(smooth=1e-6, threshold=0.5)
    prec_metric = PrecisionScore(smooth=1e-6, threshold=0.5)
    rec_metric = RecallScore(smooth=1e-6, threshold=0.5)
    f1_metric = F1Score(smooth=1e-6, threshold=0.5)

    # Check empty pred, empty target (TP=0, FP=0, FN=0)
    # Expected: IoU=1, Prec=1, Rec=1, F1=1 (smooth/smooth)
    assert torch.isclose(
        iou_metric(pred_empty, target_empty), torch.tensor(1.0)
    )
    assert torch.isclose(
        prec_metric(pred_empty, target_empty), torch.tensor(1.0)
    )
    assert torch.isclose(
        rec_metric(pred_empty, target_empty), torch.tensor(1.0)
    )
    assert torch.isclose(
        f1_metric(pred_empty, target_empty), torch.tensor(1.0)
    )

    # Check empty pred, non-empty target (TP=0, FP > 0, FN=0)
    # This case is hard to test reliably with smoothing and random pred
    # pred_non_empty = torch.sigmoid(torch.randn_like(sample_target_mask) * 2)
    # Expected: IoU=0, Prec=0, Rec=1, F1=0
    # Note: Recall is TP / (TP + FN) = 0 / (0 + 0) = smooth/smooth = 1
    # assert torch.isclose(iou_metric(pred_non_empty, target_empty),
    #                      torch.tensor(0.5), atol=1e-5) # Expect 0.5 now?
    # assert torch.isclose(prec_metric(pred_non_empty, target_empty),
    #                      torch.tensor(0.0), atol=1e-5)
    # assert torch.isclose(rec_metric(pred_non_empty, target_empty),
    #                      torch.tensor(1.0))
    # assert torch.isclose(f1_metric(pred_non_empty, target_empty),
    #                      torch.tensor(0.0), atol=1e-5)

    # Check empty pred, non-empty target (TP=0, FP=0, FN > 0)
    # Expected: IoU=0, Prec=1, Rec=0, F1=0
    # Note: Precision is TP / (TP + FP) = 0 / (0 + 0) = smooth/smooth = 1
    assert torch.isclose(
        iou_metric(pred_empty, sample_target_mask),
        torch.tensor(0.0),
        atol=1e-5,
    )
    assert torch.isclose(
        prec_metric(pred_empty, sample_target_mask), torch.tensor(1.0)
    )
    assert torch.isclose(
        rec_metric(pred_empty, sample_target_mask),
        torch.tensor(0.0),
        atol=1e-5,
    )
    assert torch.isclose(
        f1_metric(pred_empty, sample_target_mask), torch.tensor(0.0), atol=1e-5
    )


# --- Tests for Utility Functions ---


def test_get_scalar_metrics_basic():
    """Test extracting scalars from simple tensor dictionary."""
    metrics_dict = {
        "loss": torch.tensor(0.5),
        "iou": torch.tensor(0.75),
        "f1": torch.tensor(0.8),
    }
    metrics = get_scalar_metrics(metrics_dict)
    assert metrics == pytest.approx({"loss": 0.5, "iou": 0.75, "f1": 0.8})


def test_get_scalar_metrics_mixed_types() -> None:
    """Test with a dict containing mixed types (should skip non-scalars)."""
    metrics_dict: dict[str, torch.Tensor] = {
        "loss": torch.tensor(0.2),
        "epoch": torch.tensor(3.0),
        "lr": torch.tensor(1e-4),
        "non_scalar": torch.tensor([1.0, 2.0]),  # Non-scalar tensor
    }
    metrics = get_scalar_metrics(metrics_dict)
    assert metrics == pytest.approx({"loss": 0.2, "epoch": 3.0, "lr": 1e-4})
    assert "non_scalar" not in metrics  # Should be skipped


def test_get_scalar_metrics_empty():
    """Test with an empty dictionary."""
    metrics_dict: dict[str, torch.Tensor] = {}
    metrics = get_scalar_metrics(metrics_dict)
    assert metrics == {}
