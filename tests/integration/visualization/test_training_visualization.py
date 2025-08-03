"""Integration tests for training visualization components.

This module tests the integration of training visualization
components with the broader system.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from crackseg.evaluation.visualization import AdvancedTrainingVisualizer


class TestTrainingVisualizationIntegration:
    """Integration tests for training visualization components."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_training_data = {
            "loss": [2.0, 1.5, 1.2],
            "val_loss": [1.8, 1.4, 1.1],
            "iou": [0.3, 0.5, 0.7],
            "val_iou": [0.25, 0.45, 0.65],
        }
        self.sample_epochs = [1, 2, 3]

    def test_advanced_training_visualizer_workflow(self) -> None:
        """Test complete advanced training visualizer workflow."""
        # Initialize visualizer
        visualizer = AdvancedTrainingVisualizer()

        # Test training curves
        training_fig = visualizer.create_training_curves(
            training_data=self.sample_training_data
        )
        assert training_fig is not None
        assert hasattr(training_fig, "data")

        # Test loss analysis
        loss_fig = visualizer.create_loss_analysis(
            training_loss=self.sample_training_data["loss"],
            validation_loss=self.sample_training_data["val_loss"],
        )
        assert loss_fig is not None
        assert hasattr(loss_fig, "data")

        # Test metrics analysis
        metrics_fig = visualizer.create_metrics_analysis(
            metrics_data=self.sample_training_data
        )
        assert metrics_fig is not None
        assert hasattr(metrics_fig, "data")

    def test_training_export_integration(self) -> None:
        """Test training visualization export functionality."""
        visualizer = AdvancedTrainingVisualizer()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test export functionality
            export_result = visualizer.export_visualization(
                fig=Mock(),  # Mock figure
                output_path=temp_path / "training_export",
                formats=["html", "png"],
            )

            assert export_result is not None
            assert "html" in export_result
            assert "png" in export_result

    def test_training_error_handling(self) -> None:
        """Test training visualization error handling."""
        visualizer = AdvancedTrainingVisualizer()

        # Test with invalid data
        with pytest.raises(ValueError):
            visualizer.create_training_curves(training_data={})

        # Test with mismatched data lengths
        with pytest.raises(ValueError):
            visualizer.create_loss_analysis(
                training_loss=[1, 2, 3],
                validation_loss=[1, 2],  # Mismatched length
            )

    def test_training_performance_integration(self) -> None:
        """Test training visualization performance under load."""
        visualizer = AdvancedTrainingVisualizer()

        # Test with larger dataset
        large_training_data = {
            "loss": list(range(100)),
            "val_loss": list(range(100)),
            "iou": [i / 100 for i in range(100)],
            "val_iou": [i / 100 for i in range(100)],
        }

        # Should complete without performance issues
        fig = visualizer.create_training_curves(
            training_data=large_training_data
        )
        assert fig is not None

    def test_training_memory_integration(self) -> None:
        """Test training visualization memory management."""
        visualizer = AdvancedTrainingVisualizer()

        # Test memory cleanup
        initial_memory = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )

        # Create multiple figures
        for _i in range(10):
            fig = visualizer.create_training_curves(
                training_data=self.sample_training_data,
            )
            del fig  # Explicit cleanup

        # Memory should not grow excessively
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            # Allow some memory growth but not excessive
            assert (
                final_memory - initial_memory < 100 * 1024 * 1024
            )  # 100MB limit
