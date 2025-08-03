"""Integration tests for Plotly visualization components.

This module tests the integration of Plotly-based visualization
components with the broader system.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from crackseg.evaluation.visualization import InteractivePlotlyVisualizer


class TestPlotlyVisualizationIntegration:
    """Integration tests for Plotly visualization components."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_training_data = {
            "loss": [2.0, 1.5, 1.2],
            "val_loss": [1.8, 1.4, 1.1],
            "iou": [0.3, 0.5, 0.7],
            "val_iou": [0.25, 0.45, 0.65],
        }
        self.sample_epochs = [1, 2, 3]

        self.sample_prediction_data = [
            {
                "image_path": "data/test/images/5.jpg",
                "original_image": np.random.randint(
                    0, 255, (256, 256, 3), dtype=np.uint8
                ),
                "prediction_mask": np.random.choice([True, False], (256, 256)),
                "ground_truth_mask": np.random.choice(
                    [True, False], (256, 256)
                ),
                "probability_mask": np.random.random((256, 256)),
                "confidence_map": np.random.random((256, 256)),
                "metrics": {"iou": 0.75, "f1": 0.8, "dice": 0.85},
                "iou": 0.75,
                "dice": 0.85,
            },
            {
                "image_path": "data/test/images/6.jpg",
                "original_image": np.random.randint(
                    0, 255, (256, 256, 3), dtype=np.uint8
                ),
                "prediction_mask": np.random.choice([True, False], (256, 256)),
                "ground_truth_mask": np.random.choice(
                    [True, False], (256, 256)
                ),
                "probability_mask": np.random.random((256, 256)),
                "confidence_map": np.random.random((256, 256)),
                "metrics": {"iou": 0.82, "f1": 0.85, "dice": 0.88},
                "iou": 0.82,
                "dice": 0.88,
            },
        ]

    def test_interactive_plotly_workflow(self) -> None:
        """Test complete interactive Plotly workflow."""
        # Initialize visualizer
        visualizer = InteractivePlotlyVisualizer(
            export_formats=["html", "png", "json"]
        )

        # Test training curves
        training_fig = visualizer.create_interactive_training_curves(
            metrics_data=self.sample_training_data, epochs=self.sample_epochs
        )
        assert training_fig is not None
        assert hasattr(training_fig, "data")

        # Test prediction grid
        prediction_fig = visualizer.create_interactive_prediction_grid(
            results=self.sample_prediction_data
        )
        assert prediction_fig is not None
        assert hasattr(prediction_fig, "data")

        # Test confidence map
        confidence_fig = visualizer.create_interactive_confidence_map(
            confidence_data={
                "confidence_map": self.sample_prediction_data[0][
                    "confidence_map"
                ]
            }
        )
        assert confidence_fig is not None
        assert hasattr(confidence_fig, "data")

    def test_plotly_export_integration(self) -> None:
        """Test Plotly export functionality integration."""
        visualizer = InteractivePlotlyVisualizer(
            export_formats=["html", "png", "json"]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test export functionality
            export_result = visualizer.export_visualization(
                fig=Mock(),  # Mock figure
                output_path=temp_path / "test_export",
                formats=["html", "png"],
            )

            assert export_result is not None
            assert "html" in export_result
            assert "png" in export_result

    def test_plotly_error_handling(self) -> None:
        """Test Plotly error handling integration."""
        visualizer = InteractivePlotlyVisualizer()

        # Test with invalid data
        with pytest.raises(ValueError):
            visualizer.create_interactive_training_curves(
                metrics_data={}, epochs=[]
            )

        # Test with empty results list instead of None
        with pytest.raises(ValueError):
            visualizer.create_interactive_prediction_grid(results=[])

    def test_plotly_performance_integration(self) -> None:
        """Test Plotly performance under load."""
        visualizer = InteractivePlotlyVisualizer()

        # Test with larger dataset
        large_training_data = {
            "loss": list(range(100)),
            "val_loss": list(range(100)),
            "iou": [i / 100 for i in range(100)],
            "val_iou": [i / 100 for i in range(100)],
        }
        large_epochs = list(range(100))

        # Should complete without performance issues
        fig = visualizer.create_interactive_training_curves(
            metrics_data=large_training_data, epochs=large_epochs
        )
        assert fig is not None
