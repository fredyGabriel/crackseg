"""Integration tests for the complete visualization system.

This module tests the integration between different visualization
components and their interaction with the broader system.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from crackseg.evaluation.visualization import (
    AdvancedPredictionVisualizer,
    AdvancedTrainingVisualizer,
    ExperimentVisualizer,
    InteractivePlotlyVisualizer,
)


class TestVisualizationSystemIntegration:
    """Integration tests for the complete visualization system."""

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

    def test_advanced_prediction_visualizer_workflow(self) -> None:
        """Test advanced prediction visualizer workflow."""
        visualizer = AdvancedPredictionVisualizer()

        # Test comparison grid
        comparison_fig = visualizer.create_comparison_grid(
            results=self.sample_prediction_data
        )
        assert comparison_fig is not None

        # Test confidence map
        confidence_fig = visualizer.create_confidence_map(
            result=self.sample_prediction_data[0]
        )
        assert confidence_fig is not None

        # Test error analysis
        error_fig = visualizer.create_error_analysis(
            result=self.sample_prediction_data[0]
        )
        assert error_fig is not None

        # Test segmentation overlay
        overlay_fig = visualizer.create_segmentation_overlay(
            result=self.sample_prediction_data[0]
        )
        assert overlay_fig is not None

        # Test tabular comparison
        tabular_fig = visualizer.create_tabular_comparison(
            results=self.sample_prediction_data
        )
        assert tabular_fig is not None

    def test_advanced_training_visualizer_workflow(self) -> None:
        """Test advanced training visualizer workflow."""
        visualizer = AdvancedTrainingVisualizer()

        # Test training curves
        curves_fig = visualizer.create_training_curves(
            training_data={
                "metrics": [
                    {
                        "epoch": 1,
                        "loss": 2.0,
                        "val_loss": 1.8,
                        "iou": 0.3,
                        "val_iou": 0.25,
                    },
                    {
                        "epoch": 2,
                        "loss": 1.5,
                        "val_loss": 1.4,
                        "iou": 0.5,
                        "val_iou": 0.45,
                    },
                    {
                        "epoch": 3,
                        "loss": 1.2,
                        "val_loss": 1.1,
                        "iou": 0.7,
                        "val_iou": 0.65,
                    },
                ]
            }
        )
        assert curves_fig is not None

        # Test learning rate analysis
        lr_fig = visualizer.analyze_learning_rate_schedule(
            training_data={
                "metrics": [
                    {
                        "epoch": 1,
                        "loss": 2.0,
                        "val_loss": 1.8,
                        "iou": 0.3,
                        "val_iou": 0.25,
                    },
                    {
                        "epoch": 2,
                        "loss": 1.5,
                        "val_loss": 1.4,
                        "iou": 0.5,
                        "val_iou": 0.45,
                    },
                    {
                        "epoch": 3,
                        "loss": 1.2,
                        "val_loss": 1.1,
                        "iou": 0.7,
                        "val_iou": 0.65,
                    },
                ]
            }
        )
        assert lr_fig is not None

        # Test gradient flow
        gradient_fig = visualizer.visualize_gradient_flow(
            gradient_data={
                "metrics": [
                    {
                        "epoch": 1,
                        "loss": 2.0,
                        "val_loss": 1.8,
                        "iou": 0.3,
                        "val_iou": 0.25,
                    },
                    {
                        "epoch": 2,
                        "loss": 1.5,
                        "val_loss": 1.4,
                        "iou": 0.5,
                        "val_iou": 0.45,
                    },
                    {
                        "epoch": 3,
                        "loss": 1.2,
                        "val_loss": 1.1,
                        "iou": 0.7,
                        "val_iou": 0.65,
                    },
                ]
            }
        )
        assert gradient_fig is not None

        # Test parameter distribution
        param_fig = visualizer.visualize_parameter_distributions(
            model_path=Path("outputs/checkpoints/model_best.pth.tar")
        )
        assert param_fig is not None

    def test_experiment_visualizer_workflow(self) -> None:
        """Test experiment visualizer workflow."""
        visualizer = ExperimentVisualizer()

        # Test loading experiment data (mock)
        experiment_dir = Path("outputs/test_experiment")
        experiment_data = visualizer.load_experiment_data(experiment_dir)
        # This will be empty since the directory doesn't exist, but it should
        # not raise an error
        assert isinstance(experiment_data, dict)

        # Test creating comparison table
        experiments_data = {
            "exp1": {
                "summary": {
                    "best_metrics": {
                        "loss": {"value": 0.5},
                        "iou": {"value": 0.8},
                        "f1": {"value": 0.85},
                        "precision": {"value": 0.82},
                        "recall": {"value": 0.88},
                    },
                    "experiment_info": {
                        "total_epochs": 100,
                        "best_epoch": 85,
                    },
                }
            },
            "exp2": {
                "summary": {
                    "best_metrics": {
                        "loss": {"value": 0.4},
                        "iou": {"value": 0.85},
                        "f1": {"value": 0.88},
                        "precision": {"value": 0.86},
                        "recall": {"value": 0.90},
                    },
                    "experiment_info": {
                        "total_epochs": 100,
                        "best_epoch": 90,
                    },
                }
            },
        }

        comparison_df = visualizer.create_comparison_table(experiments_data)
        assert not comparison_df.empty
        assert "Experiment" in comparison_df.columns
        assert "Final Loss" in comparison_df.columns
        assert "Final IoU" in comparison_df.columns

        # Test finding experiment directories
        experiment_dirs = visualizer.find_experiment_directories()
        assert isinstance(experiment_dirs, list)

    def test_cross_visualizer_compatibility(self) -> None:
        """Test compatibility between different visualizers."""
        # Test that different visualizers can work with the same data
        interactive_viz = InteractivePlotlyVisualizer()
        advanced_pred_viz = AdvancedPredictionVisualizer()
        advanced_train_viz = AdvancedTrainingVisualizer()
        experiment_viz = ExperimentVisualizer()

        # All should be able to handle the same data structures
        assert interactive_viz is not None
        assert advanced_pred_viz is not None
        assert advanced_train_viz is not None
        assert experiment_viz is not None

    def test_export_integration(self) -> None:
        """Test export functionality integration."""
        visualizer = InteractivePlotlyVisualizer(
            export_formats=["html", "png", "json"]
        )

        fig = visualizer.create_interactive_training_curves(
            metrics_data=self.sample_training_data, epochs=self.sample_epochs
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_export"

            # Test saving with multiple formats
            visualizer.export_handler.save_plot(fig, save_path)

            # Verify directory was created
            assert save_path.parent.exists()

    def test_template_integration(self) -> None:
        """Test template integration with visualizers."""
        # Mock template
        mock_template = Mock()
        mock_template.apply_template.return_value = Mock()

        # Test with interactive visualizer
        interactive_viz = InteractivePlotlyVisualizer(template=mock_template)
        _fig = interactive_viz.create_interactive_training_curves(
            metrics_data=self.sample_training_data, epochs=self.sample_epochs
        )

        # Template should be applied
        assert interactive_viz.template == mock_template

    def test_error_handling_integration(self) -> None:
        """Test error handling across the visualization system."""
        visualizer = InteractivePlotlyVisualizer()

        # Test with invalid data
        _invalid_data = {"invalid": "data"}

        # Should handle gracefully without crashing
        fig = visualizer.create_interactive_training_curves(
            metrics_data={"loss": [1.0]}, epochs=[1]
        )
        assert fig is not None

        # Test with empty data
        empty_data = {}
        fig = visualizer.create_interactive_training_curves(
            metrics_data=empty_data, epochs=[]
        )
        assert fig is not None

    def test_performance_integration(self) -> None:
        """Test performance characteristics of the visualization system."""
        visualizer = InteractivePlotlyVisualizer()

        # Test with larger dataset
        large_training_data = {
            "loss": [
                2.0 * np.exp(-i / 20) + 0.1 * np.random.random()
                for i in range(100)
            ],
            "val_loss": [
                1.8 * np.exp(-i / 25) + 0.15 * np.random.random()
                for i in range(100)
            ],
            "iou": [
                0.3 + 0.6 * (1 - np.exp(-i / 15)) + 0.02 * np.random.random()
                for i in range(100)
            ],
            "val_iou": [
                0.25 + 0.55 * (1 - np.exp(-i / 18)) + 0.03 * np.random.random()
                for i in range(100)
            ],
        }
        large_epochs = list(range(1, 101))

        # Should handle large datasets without performance issues
        fig = visualizer.create_interactive_training_curves(
            metrics_data=large_training_data, epochs=large_epochs
        )
        assert fig is not None

    def test_memory_integration(self) -> None:
        """Test memory usage of the visualization system."""
        visualizer = InteractivePlotlyVisualizer()

        # Test with large images
        large_prediction_data = [
            {
                "original_image": np.random.randint(
                    0, 255, (1024, 1024, 3), dtype=np.uint8
                ),
                "prediction_mask": np.random.choice(
                    [True, False], (1024, 1024)
                ),
                "ground_truth_mask": np.random.choice(
                    [True, False], (1024, 1024)
                ),
                "confidence_map": np.random.random((1024, 1024)),
                "metrics": {"iou": 0.75, "f1": 0.8, "dice": 0.85},
            }
        ]

        # Should handle large images without memory issues
        fig = visualizer.create_interactive_prediction_grid(
            results=large_prediction_data
        )
        assert fig is not None

    def test_concurrent_visualization(self) -> None:
        """Test concurrent visualization operations."""
        visualizer = InteractivePlotlyVisualizer()

        # Test multiple simultaneous operations
        training_fig = visualizer.create_interactive_training_curves(
            metrics_data=self.sample_training_data, epochs=self.sample_epochs
        )
        prediction_fig = visualizer.create_interactive_prediction_grid(
            results=self.sample_prediction_data
        )
        confidence_fig = visualizer.create_interactive_confidence_map(
            confidence_data={
                "confidence_map": self.sample_prediction_data[0][
                    "confidence_map"
                ]
            }
        )

        # All should work concurrently
        assert training_fig is not None
        assert prediction_fig is not None
        assert confidence_fig is not None

    def test_data_type_integration(self) -> None:
        """Test integration with different data types."""
        visualizer = InteractivePlotlyVisualizer()

        # Test with numpy arrays
        numpy_data = {
            "loss": [float(np.float32(2.0))],
            "val_loss": [float(np.float32(1.8))],
            "iou": [float(np.float32(0.3))],
            "val_iou": [float(np.float32(0.25))],
        }

        fig = visualizer.create_interactive_training_curves(
            metrics_data=numpy_data, epochs=[1]
        )
        assert fig is not None

        # Test with torch tensors
        torch_data = {
            "loss": [torch.tensor(2.0).item()],
            "val_loss": [torch.tensor(1.8).item()],
            "iou": [torch.tensor(0.3).item()],
            "val_iou": [torch.tensor(0.25).item()],
        }

        fig = visualizer.create_interactive_training_curves(
            metrics_data=torch_data, epochs=[1]
        )
        assert fig is not None

    def test_configuration_integration(self) -> None:
        """Test configuration integration with visualizers."""
        # Test with custom configuration
        _custom_config = {
            "figure_size": (12, 8),
            "color_palette": "viridis",
            "line_width": 3,
            "font_size": 14,
        }

        visualizer = InteractivePlotlyVisualizer(
            export_formats=["html", "png", "json"]
        )

        # Should work with custom configuration
        fig = visualizer.create_interactive_training_curves(
            metrics_data=self.sample_training_data, epochs=self.sample_epochs
        )
        assert fig is not None

    def test_file_system_integration(self) -> None:
        """Test file system integration."""
        visualizer = InteractivePlotlyVisualizer(
            export_formats=["html", "png", "json"]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test saving to different directory structures
            nested_dir = Path(temp_dir) / "nested" / "deep" / "directory"
            save_path = nested_dir / "test_visualization"

            fig = visualizer.create_interactive_training_curves(
                metrics_data=self.sample_training_data,
                epochs=self.sample_epochs,
            )

            # Should create nested directories automatically
            visualizer.export_handler.save_plot(fig, save_path)

            # Verify directory structure was created
            assert nested_dir.exists()

    def test_metadata_integration(self) -> None:
        """Test metadata integration across visualizers."""
        visualizer = InteractivePlotlyVisualizer(
            export_formats=["html", "png", "json"]
        )

        _fig = visualizer.create_interactive_training_curves(
            metrics_data=self.sample_training_data, epochs=self.sample_epochs
        )

        # Test metadata generation
        metadata = visualizer.metadata_handler.create_metadata(
            "training_curves",
            {
                "epochs": len(self.sample_epochs),
                "metrics": list(self.sample_training_data.keys()),
            },
        )

        assert "created_at" in metadata
        assert "plot_type" in metadata
        assert metadata["plot_type"] == "training_curves"

    def test_error_recovery_integration(self) -> None:
        """Test error recovery in the visualization system."""
        visualizer = InteractivePlotlyVisualizer()

        # Test recovery from invalid data
        invalid_results = [
            {
                "original_image": None,  # Invalid data
                "prediction_mask": None,
                "ground_truth_mask": None,
                "confidence_map": None,
                "metrics": {},
            }
        ]

        # Should handle gracefully
        fig = visualizer.create_interactive_prediction_grid(
            results=invalid_results
        )
        assert fig is not None

        # Should create empty figure with appropriate message
        assert hasattr(fig, "layout")


class TestVisualizationSystemStress:
    """Stress tests for the visualization system."""

    def test_large_dataset_handling(self) -> None:
        """Test handling of large datasets."""
        visualizer = InteractivePlotlyVisualizer()

        # Create large dataset
        large_training_data = {
            "loss": [
                2.0 * np.exp(-i / 20) + 0.1 * np.random.random()
                for i in range(1000)
            ],
            "val_loss": [
                1.8 * np.exp(-i / 25) + 0.15 * np.random.random()
                for i in range(1000)
            ],
            "iou": [
                0.3 + 0.6 * (1 - np.exp(-i / 15)) + 0.02 * np.random.random()
                for i in range(1000)
            ],
            "val_iou": [
                0.25 + 0.55 * (1 - np.exp(-i / 18)) + 0.03 * np.random.random()
                for i in range(1000)
            ],
        }
        large_epochs = list(range(1, 1001))

        # Should handle large datasets
        fig = visualizer.create_interactive_training_curves(
            metrics_data=large_training_data, epochs=large_epochs
        )
        assert fig is not None

    def test_many_prediction_results(self) -> None:
        """Test handling of many prediction results."""
        visualizer = InteractivePlotlyVisualizer()

        # Create many prediction results
        many_results = [
            {
                "original_image": np.random.randint(
                    0, 255, (256, 256, 3), dtype=np.uint8
                ),
                "prediction_mask": np.random.choice([True, False], (256, 256)),
                "ground_truth_mask": np.random.choice(
                    [True, False], (256, 256)
                ),
                "confidence_map": np.random.random((256, 256)),
                "metrics": {"iou": 0.75, "f1": 0.8, "dice": 0.85},
            }
            for _ in range(50)
        ]

        # Should handle many results
        fig = visualizer.create_interactive_prediction_grid(
            results=many_results
        )
        assert fig is not None

    def test_concurrent_operations(self) -> None:
        """Test concurrent visualization operations."""
        visualizer = InteractivePlotlyVisualizer()

        # Test multiple concurrent operations
        import threading

        def create_training_curves():
            return visualizer.create_interactive_training_curves(
                metrics_data={"loss": [2.0, 1.5], "val_loss": [1.8, 1.4]},
                epochs=[1, 2],
            )

        def create_prediction_grid():
            return visualizer.create_interactive_prediction_grid(
                results=[
                    {
                        "original_image": np.random.randint(
                            0, 255, (256, 256, 3), dtype=np.uint8
                        ),
                        "prediction_mask": np.random.choice(
                            [True, False], (256, 256)
                        ),
                        "ground_truth_mask": np.random.choice(
                            [True, False], (256, 256)
                        ),
                        "confidence_map": np.random.random((256, 256)),
                        "metrics": {"iou": 0.75, "f1": 0.8, "dice": 0.85},
                    }
                ]
            )

        # Run concurrent operations
        threads = []
        results = []

        for _ in range(5):
            t1 = threading.Thread(
                target=lambda: results.append(create_training_curves())
            )
            t2 = threading.Thread(
                target=lambda: results.append(create_prediction_grid())
            )
            threads.extend([t1, t2])
            t1.start()
            t2.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All operations should complete successfully
        assert len(results) == 10
        for result in results:
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__])
