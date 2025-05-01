import torch
from abc import ABC, abstractmethod
from typing import Dict


def _threshold(x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Applies a threshold to the input tensor."""
    return (x > threshold).float()


class Metric(ABC):
    """Base class for evaluation metrics."""

    def __init__(self, smooth: float = 1e-6, threshold: float | None = 0.5):
        self.smooth = smooth
        self.threshold = threshold

    def _validate_and_preprocess(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validates input shapes and applies thresholding."""
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        assert pred.dim() == 3, (
            f"Pred tensor must have 3 dims (N, H, W), got {pred.dim()}"
        )
        assert target.dim() == 3, (
            f"Target tensor must have 3 dims (N, H, W), got {target.dim()}"
        )
        assert pred.shape == target.shape, "Pred and target shapes must match"

        if self.threshold is not None:
            # Apply sigmoid if input looks like logits
            is_logits = pred.min() < 0 or pred.max() > 1
            probs = torch.sigmoid(pred) if is_logits else pred
            pred = _threshold(probs, self.threshold)

        return pred, target

    @abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor
                ) -> torch.Tensor:
        """Calculates the metric score for a batch."""
        pass

    def __call__(self, pred: torch.Tensor, target: torch.Tensor
                 ) -> torch.Tensor:
        """Allows calling the metric instance like a function."""
        return self.forward(pred, target)


class IoUScore(Metric):
    """Calculates Intersection over Union (IoU) score."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor
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

    def forward(self, pred: torch.Tensor, target: torch.Tensor
                ) -> torch.Tensor:
        pred, target = self._validate_and_preprocess(pred, target)

        tp = torch.sum(pred * target, dim=(1, 2))  # True Positives
        fp = torch.sum(pred * (1 - target), dim=(1, 2))  # False Positives

        precision = (tp + self.smooth) / (tp + fp + self.smooth)
        return precision.mean()


class RecallScore(Metric):
    """Calculates Recall score."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor
                ) -> torch.Tensor:
        pred, target = self._validate_and_preprocess(pred, target)

        tp = torch.sum(pred * target, dim=(1, 2))  # True Positives
        fn = torch.sum((1 - pred) * target, dim=(1, 2))  # False Negatives

        recall = (tp + self.smooth) / (tp + fn + self.smooth)
        return recall.mean()


class F1Score(Metric):
    """Calculates F1 score (Dice coefficient)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor
                ) -> torch.Tensor:
        pred, target = self._validate_and_preprocess(pred, target)

        tp = torch.sum(pred * target, dim=(1, 2))  # True Positives
        fp = torch.sum(pred * (1 - target), dim=(1, 2))  # False Positives
        fn = torch.sum((1 - pred) * target, dim=(1, 2))  # False Negatives

        # Calculate F1 score directly (equivalent to Dice)
        f1 = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        return f1.mean()


# --- Utility function for extracting scalar values ---
def get_scalar_metrics(metrics_dict: Dict[str, torch.Tensor]) -> Dict[str,
                                                                      float]:
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
        elif isinstance(value, (float, int)):
            scalar_metrics[name] = float(value)
        # else: # Optionally warn or skip non-scalar tensors
        #     print(f"Warning: Metric '{name}' is not a scalar tensor, \
        # skipping.")
    return scalar_metrics


# Example usage (can be removed later)
if __name__ == '__main__':
    # Example Tensors (Batch size 2, Height 3, Width 3)
    pred_logits = torch.randn(2, 3, 3)  # Example logits
    pred_probs = torch.sigmoid(pred_logits)  # Example probabilities
    # Example binary target mask
    target_mask = torch.randint(0, 2, (2, 3, 3)).float()

    print(f"Target Mask:\n{target_mask}\n")

    # Instantiate metrics
    iou_metric = IoUScore(threshold=0.5)
    precision_metric = PrecisionScore(threshold=0.5)
    recall_metric = RecallScore(threshold=0.5)
    f1_metric = F1Score(threshold=0.5)

    iou_metric_no_thresh = IoUScore(threshold=None)
    precision_metric_no_thresh = PrecisionScore(threshold=None)
    recall_metric_no_thresh = RecallScore(threshold=None)
    f1_metric_no_thresh = F1Score(threshold=None)

    # --- Using Logits ---
    print("--- Metrics using Logits (threshold=0.5) ---")
    iou = iou_metric(pred_logits, target_mask)
    precision = precision_metric(pred_logits, target_mask)
    recall = recall_metric(pred_logits, target_mask)
    f1 = f1_metric(pred_logits, target_mask)
    print(f"IoU: {iou.item():.4f}")
    print(f"Precision: {precision.item():.4f}")
    print(f"Recall: {recall.item():.4f}")
    print(f"F1 Score: {f1.item():.4f}\n")

    # --- Using Probabilities ---
    print("--- Metrics using Probabilities (threshold=0.5) ---")
    iou = iou_metric(pred_probs, target_mask)
    precision = precision_metric(pred_probs, target_mask)
    recall = recall_metric(pred_probs, target_mask)
    f1 = f1_metric(pred_probs, target_mask)
    print(f"IoU: {iou.item():.4f}")
    print(f"Precision: {precision.item():.4f}")
    print(f"Recall: {recall.item():.4f}")
    print(f"F1 Score: {f1.item():.4f}\n")

    # --- Using Pre-thresholded Predictions ---
    print("--- Metrics using Pre-thresholded Probs (threshold=None) ---")
    pred_thresholded = (pred_probs > 0.5).float()
    print(f"Thresholded Prediction:\n{pred_thresholded}\n")
    iou = iou_metric_no_thresh(pred_thresholded, target_mask)
    precision = precision_metric_no_thresh(pred_thresholded, target_mask)
    recall = recall_metric_no_thresh(pred_thresholded, target_mask)
    f1 = f1_metric_no_thresh(pred_thresholded, target_mask)
    print(f"IoU: {iou.item():.4f}")
    print(f"Precision: {precision.item():.4f}")
    print(f"Recall: {recall.item():.4f}")
    print(f"F1 Score: {f1.item():.4f}\n")

    # --- Edge Case: Empty Target ---
    print("--- Edge Case: Empty Target ---")
    empty_target = torch.zeros_like(target_mask)
    iou = iou_metric(pred_probs, empty_target)
    precision = precision_metric(pred_probs, empty_target)
    recall = recall_metric(pred_probs, empty_target)
    f1 = f1_metric(pred_probs, empty_target)
    print(f"IoU: {iou.item():.4f}")
    # Precision ~1.0 if pred empty, else low/NaN handled by smooth
    print(f"Precision: {precision.item():.4f}")
    # Recall ~1.0 if pred empty, else low
    print(f"Recall: {recall.item():.4f}")
    print(f"F1 Score: {f1.item():.4f}\n")

    # --- Edge Case: Empty Prediction ---
    print("--- Edge Case: Empty Prediction ---")
    empty_pred = torch.zeros_like(pred_probs)
    iou = iou_metric_no_thresh(empty_pred, target_mask)
    # Precision ~1.0 if target empty, else low/NaN handled by smooth
    precision = precision_metric_no_thresh(empty_pred, target_mask)
    recall = recall_metric_no_thresh(empty_pred, target_mask)
    f1 = f1_metric_no_thresh(empty_pred, target_mask)
    print(f"IoU: {iou.item():.4f}")
    print(f"Precision: {precision.item():.4f}")
    # Recall ~1.0 if target empty, else low
    print(f"Recall: {recall.item():.4f}")
    print(f"F1 Score: {f1.item():.4f}")

    # --- Edge Case: Full Prediction and Target ---
    print("--- Edge Case: Full Prediction and Target ---")
    full_pred = torch.ones_like(pred_probs)
    full_target = torch.ones_like(target_mask)
    iou = iou_metric_no_thresh(full_pred, full_target)
    precision = precision_metric_no_thresh(full_pred, full_target)
    recall = recall_metric_no_thresh(full_pred, full_target)
    f1 = f1_metric_no_thresh(full_pred, full_target)
    print(f"IoU: {iou.item():.4f}")  # Expected ~1.0
    print(f"Precision: {precision.item():.4f}")  # Expected ~1.0
    print(f"Recall: {recall.item():.4f}")  # Expected ~1.0
    print(f"F1 Score: {f1.item():.4f}")  # Expected ~1.0
