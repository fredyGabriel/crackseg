from abc import ABC, abstractmethod

import torch


def _threshold(x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Applies a threshold to the input tensor."""
    return (x > threshold).float()


class Metric(ABC):
    """Base class for evaluation metrics."""

    def __init__(
        self,
        smooth: float = 1e-6,
        threshold: float | None = 0.5,
        # Default to 4 for backward compatibility / tests
        expected_dims_before_squeeze: int = 4,
        # Default to 3 for backward compatibility / tests
        expected_dims_after_squeeze: int = 3,
    ):
        self.smooth = smooth
        self.threshold = threshold
        self.expected_dims_before_squeeze = expected_dims_before_squeeze
        self.expected_dims_after_squeeze = expected_dims_after_squeeze

    def _validate_and_preprocess(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validates input shapes and applies thresholding."""
        if (
            pred.dim() == self.expected_dims_before_squeeze
            and pred.shape[1] == 1
        ):
            pred = pred.squeeze(1)
        if (
            target.dim() == self.expected_dims_before_squeeze
            and target.shape[1] == 1
        ):
            target = target.squeeze(1)

        assert pred.dim() == self.expected_dims_after_squeeze, (
            f"Pred tensor must have {self.expected_dims_after_squeeze} dims "
            f"(N, H, W after squeeze), got {pred.dim()}"
        )
        assert target.dim() == self.expected_dims_after_squeeze, (
            f"Target tensor must have {self.expected_dims_after_squeeze} dims "
            f"(N, H, W after squeeze), got {target.dim()}"
        )
        assert pred.shape == target.shape, "Pred and target shapes must match"

        if self.threshold is not None:
            # Apply sigmoid if input looks like logits
            is_logits = pred.min() < 0 or pred.max() > 1
            probs = torch.sigmoid(pred) if is_logits else pred
            pred = _threshold(probs, self.threshold)

        return pred, target

    @abstractmethod
    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the metric score for a batch."""
        pass

    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Allows calling the metric instance like a function."""
        return self.forward(pred, target)


class IoUScore(Metric):
    """Calculates Intersection over Union (IoU) score."""

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        pred, target = self._validate_and_preprocess(pred, target)

        intersection = torch.sum(pred * target, dim=(1, 2))
        union = (
            torch.sum(pred, dim=(1, 2))
            + torch.sum(target, dim=(1, 2))
            - intersection
        )

        iou = (intersection + self.smooth) / (union + self.smooth)
        return iou.mean()


class PrecisionScore(Metric):
    """Calculates Precision score."""

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        pred, target = self._validate_and_preprocess(pred, target)

        tp = torch.sum(pred * target, dim=(1, 2))  # True Positives
        fp = torch.sum(pred * (1 - target), dim=(1, 2))  # False Positives

        precision = (tp + self.smooth) / (tp + fp + self.smooth)
        return precision.mean()


class RecallScore(Metric):
    """Calculates Recall score."""

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        pred, target = self._validate_and_preprocess(pred, target)

        tp = torch.sum(pred * target, dim=(1, 2))  # True Positives
        fn = torch.sum((1 - pred) * target, dim=(1, 2))  # False Negatives

        recall = (tp + self.smooth) / (tp + fn + self.smooth)
        return recall.mean()


class F1Score(Metric):
    """Calculates F1 score (Dice coefficient)."""

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        pred, target = self._validate_and_preprocess(pred, target)

        tp = torch.sum(pred * target, dim=(1, 2))  # True Positives
        fp = torch.sum(pred * (1 - target), dim=(1, 2))  # False Positives
        fn = torch.sum((1 - pred) * target, dim=(1, 2))  # False Negatives

        # Calculate F1 score directly (equivalent to Dice)
        f1 = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        return f1.mean()


# --- Utility function for extracting scalar values ---
def get_scalar_metrics(
    metrics_dict: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Extracts scalar float values from a dictionary of metric tensors.

    Args:
        metrics_dict: Dictionary where keys are metric names and values
                      are scalar torch.Tensor objects.

    Returns:
        Dictionary with the same keys but float values.
    """
    scalar_metrics = {}
    for name, value in metrics_dict.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            scalar_metrics[name] = value.item()
        elif isinstance(value, float | int):
            scalar_metrics[name] = float(value)
        # else: # Optionally warn or skip non-scalar tensors
        #     print(f"Warning: Metric '{name}' is not a scalar tensor, \
        # skipping.")
    return scalar_metrics
