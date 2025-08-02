# Visualization Testing Guide

Comprehensive testing guide for the CrackSeg visualization system.

## Table of Contents

1. [Overview](#overview)
2. [Testing Strategy](#testing-strategy)
3. [Unit Testing](#unit-testing)
4. [Integration Testing](#integration-testing)
5. [Test Data Management](#test-data-management)
6. [Mocking Strategies](#mocking-strategies)
7. [Quality Gates](#quality-gates)
8. [Performance Testing](#performance-testing)
9. [Error Handling Testing](#error-handling-testing)
10. [Best Practices](#best-practices)

## Overview

This guide covers comprehensive testing strategies for the CrackSeg visualization system, ensuring
reliability, performance, and maintainability of all visualization components.

## Testing Strategy

### Core Principles

1. **Production Code Drives Test Expectations**: Tests adapt to the actual implementation
2. **Comprehensive Coverage**: Unit tests for components, integration tests for workflows
3. **Type Safety**: All tests include proper type annotations
4. **Quality Gates**: All test code must pass ruff, black, and basedpyright
5. **Realistic Data**: Use realistic test data that mimics production scenarios

### Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions and workflows
- **Performance Tests**: Test with large datasets and memory usage
- **Error Handling Tests**: Test graceful failure scenarios

## Unit Testing

### InteractivePlotlyVisualizer Testing

```python
import pytest
from unittest.mock import Mock, patch
import numpy as np
from pathlib import Path
from crackseg.evaluation.visualization import InteractivePlotlyVisualizer


class TestInteractivePlotlyVisualizer:
    """Test InteractivePlotlyVisualizer functionality."""

    @pytest.fixture
    def visualizer(self) -> InteractivePlotlyVisualizer:
        """Provide visualizer instance."""
        return InteractivePlotlyVisualizer(
            export_formats=["html", "png", "pdf"]
        )

    @pytest.fixture
    def sample_training_data(self) -> dict[str, list[float]]:
        """Provide sample training data."""
        return {
            "loss": [2.0, 1.5, 1.2, 0.9, 0.7],
            "val_loss": [1.8, 1.4, 1.1, 0.8, 0.6],
            "iou": [0.3, 0.5, 0.7, 0.8, 0.85],
            "val_iou": [0.25, 0.45, 0.65, 0.75, 0.8],
        }

    @pytest.fixture
    def sample_epochs(self) -> list[int]:
        """Provide sample epochs."""
        return [1, 2, 3, 4, 5]

    def test_create_interactive_training_curves(
        self,
        visualizer: InteractivePlotlyVisualizer,
        sample_training_data: dict[str, list[float]],
        sample_epochs: list[int]
    ) -> None:
        """Test creating interactive training curves."""
        fig = visualizer.create_interactive_training_curves(
            metrics_data=sample_training_data,
            epochs=sample_epochs
        )

        assert fig is not None
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
        assert len(fig.data) == len(sample_training_data)

    def test_create_interactive_prediction_grid(
        self,
        visualizer: InteractivePlotlyVisualizer
    ) -> None:
        """Test creating interactive prediction grid."""
        sample_results = [
            {
                "original_image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
                "prediction_mask": np.random.choice([True, False], (256, 256)),
                "ground_truth_mask": np.random.choice([True, False], (256, 256)),
                "confidence_map": np.random.random((256, 256)),
                "metrics": {"iou": 0.75, "f1": 0.8, "dice": 0.85},
            }
        ]

        fig = visualizer.create_interactive_prediction_grid(
            results=sample_results,
            max_images=4
        )

        assert fig is not None
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')

    def test_create_interactive_confidence_map(
        self,
        visualizer: InteractivePlotlyVisualizer
    ) -> None:
        """Test creating interactive confidence map."""
        confidence_data = {
            "confidence_map": np.random.random((256, 256)),
            "original_image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            "prediction_mask": np.random.choice([True, False], (256, 256)),
        }

        fig = visualizer.create_interactive_confidence_map(
            confidence_data=confidence_data
        )

        assert fig is not None
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')

    def test_export_functionality(
        self,
        visualizer: InteractivePlotlyVisualizer,
        sample_training_data: dict[str, list[float]],
        sample_epochs: list[int],
        tmp_path: Path
    ) -> None:
        """Test export functionality."""
        fig = visualizer.create_interactive_training_curves(
            metrics_data=sample_training_data,
            epochs=sample_epochs
        )

        save_path = tmp_path / "test_export"

        # Test HTML export
        html_path = visualizer.export_handler.save_plot(fig, save_path, format="html")
        assert html_path.exists()

        # Test PNG export
        png_path = visualizer.export_handler.save_plot(fig, save_path, format="png")
        assert png_path.exists()

    def test_metadata_functionality(
        self,
        visualizer: InteractivePlotlyVisualizer,
        sample_training_data: dict[str, list[float]],
        sample_epochs: list[int]
    ) -> None:
        """Test metadata functionality."""
        fig = visualizer.create_interactive_training_curves(
            metrics_data=sample_training_data,
            epochs=sample_epochs
        )

        metadata = visualizer.metadata_handler.create_metadata(
            plot_type="training_curves",
            data_info={
                "epochs": len(sample_epochs),
                "metrics": list(sample_training_data.keys()),
            }
        )

        assert "created_at" in metadata
        assert "plot_type" in metadata
        assert metadata["plot_type"] == "training_curves"
```

### ExportHandler Testing

```python
class TestExportHandler:
    """Test ExportHandler functionality."""

    @pytest.fixture
    def export_handler(self) -> ExportHandler:
        """Provide export handler instance."""
        return ExportHandler()

    @pytest.fixture
    def mock_figure(self) -> Mock:
        """Provide mock figure."""
        mock_fig = Mock()
        mock_fig.data = [Mock(), Mock()]
        mock_fig.layout = Mock()
        mock_fig.layout.title = Mock()
        mock_fig.layout.title.text = "Test Figure"

        # Mock size
        mock_size = Mock()
        mock_size.tolist.return_value = [8.0, 6.0]
        mock_fig.get_size_inches.return_value = mock_size
        mock_fig.dpi = 100

        return mock_fig

    def test_save_plot_html(
        self,
        export_handler: ExportHandler,
        mock_figure: Mock,
        tmp_path: Path
    ) -> None:
        """Test saving plot as HTML."""
        save_path = tmp_path / "test_plot"

        result_path = export_handler.save_plot(
            mock_figure,
            save_path,
            format="html"
        )

        assert result_path.exists()
        assert result_path.suffix == ".html"

    def test_save_plot_png(
        self,
        export_handler: ExportHandler,
        mock_figure: Mock,
        tmp_path: Path
    ) -> None:
        """Test saving plot as PNG."""
        save_path = tmp_path / "test_plot"

        result_path = export_handler.save_plot(
            mock_figure,
            save_path,
            format="png"
        )

        assert result_path.exists()
        assert result_path.suffix == ".png"

    def test_matplotlib_to_json(
        self,
        export_handler: ExportHandler,
        mock_figure: Mock
    ) -> None:
        """Test converting matplotlib figure to JSON."""
        result = export_handler._matplotlib_to_json(mock_figure)

        assert result["type"] == "matplotlib_figure"
        assert result["axes_count"] == 2
        assert result["figure_size"] == [8.0, 6.0]

    def test_unsupported_format(
        self,
        export_handler: ExportHandler,
        mock_figure: Mock,
        tmp_path: Path
    ) -> None:
        """Test handling unsupported format."""
        save_path = tmp_path / "test_plot"

        with pytest.raises(ValueError, match="Unsupported format"):
            export_handler.save_plot(mock_figure, save_path, format="unsupported")
```

### MetadataHandler Testing

```python
class TestMetadataHandler:
    """Test MetadataHandler functionality."""

    @pytest.fixture
    def metadata_handler(self) -> MetadataHandler:
        """Provide metadata handler instance."""
        return MetadataHandler()

    @pytest.fixture
    def mock_figure(self) -> Mock:
        """Provide mock figure with metadata."""
        mock_fig = Mock()
        mock_fig.layout = Mock()
        mock_fig.layout.annotations = []
        mock_fig.data = [Mock(), Mock()]

        return mock_fig

    def test_create_metadata(
        self,
        metadata_handler: MetadataHandler
    ) -> None:
        """Test creating metadata."""
        metadata = metadata_handler.create_metadata(
            plot_type="training_curves",
            data_info={"epochs": 100, "metrics": ["loss", "iou"]},
            custom_metadata={"experiment_id": "exp_001"}
        )

        assert "created_at" in metadata
        assert "plot_type" in metadata
        assert metadata["plot_type"] == "training_curves"
        assert "experiment_id" in metadata["custom_metadata"]

    def test_embed_metadata_in_figure(
        self,
        metadata_handler: MetadataHandler,
        mock_figure: Mock
    ) -> None:
        """Test embedding metadata in figure."""
        metadata = {"experiment_id": "exp_001", "model": "swin_v2"}

        result_fig = metadata_handler.embed_metadata_in_figure(
            mock_figure,
            metadata
        )

        assert result_fig is not None
        assert hasattr(result_fig.layout, 'annotations')

    def test_extract_metadata_from_figure(
        self,
        metadata_handler: MetadataHandler,
        mock_figure: Mock
    ) -> None:
        """Test extracting metadata from figure."""
        # First embed metadata
        metadata = {"experiment_id": "exp_001"}
        fig_with_metadata = metadata_handler.embed_metadata_in_figure(
            mock_figure,
            metadata
        )

        # Then extract it
        extracted_metadata = metadata_handler.extract_metadata_from_figure(
            fig_with_metadata
        )

        assert extracted_metadata is not None
        assert "experiment_id" in extracted_metadata

    def test_validate_metadata(
        self,
        metadata_handler: MetadataHandler
    ) -> None:
        """Test metadata validation."""
        valid_metadata = {
            "created_at": "2024-01-15T10:30:00",
            "plot_type": "training_curves",
            "data_info": {"epochs": 100}
        }

        is_valid = metadata_handler.validate_metadata(valid_metadata)
        assert is_valid is True

        invalid_metadata = {"invalid": "data"}
        is_valid = metadata_handler.validate_metadata(invalid_metadata)
        assert is_valid is False
```

## Integration Testing

### Cross-Component Integration

```python
class TestVisualizationSystemIntegration:
    """Test integration between visualization components."""

    @pytest.fixture
    def sample_prediction_data(self) -> list[dict[str, Any]]:
        """Provide sample prediction data."""
        return [
            {
                "image_path": "data/test/images/5.jpg",
                "original_image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
                "prediction_mask": np.random.choice([True, False], (256, 256)),
                "ground_truth_mask": np.random.choice([True, False], (256, 256)),
                "probability_mask": np.random.random((256, 256)),
                "confidence_map": np.random.random((256, 256)),
                "metrics": {"iou": 0.75, "f1": 0.8, "dice": 0.85},
                "iou": 0.75,
                "dice": 0.85,
            },
            {
                "image_path": "data/test/images/6.jpg",
                "original_image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
                "prediction_mask": np.random.choice([True, False], (256, 256)),
                "ground_truth_mask": np.random.choice([True, False], (256, 256)),
                "probability_mask": np.random.random((256, 256)),
                "confidence_map": np.random.random((256, 256)),
                "metrics": {"iou": 0.82, "f1": 0.85, "dice": 0.88},
                "iou": 0.82,
                "dice": 0.88,
            },
        ]

    @pytest.fixture
    def sample_training_data(self) -> dict[str, list[float]]:
        """Provide sample training data."""
        return {
            "loss": [2.0, 1.5, 1.2, 0.9, 0.7],
            "val_loss": [1.8, 1.4, 1.1, 0.8, 0.6],
            "iou": [0.3, 0.5, 0.7, 0.8, 0.85],
            "val_iou": [0.25, 0.45, 0.65, 0.75, 0.8],
        }

    @pytest.fixture
    def sample_epochs(self) -> list[int]:
        """Provide sample epochs."""
        return [1, 2, 3, 4, 5]

    def test_interactive_plotly_visualizer_workflow(
        self,
        sample_training_data: dict[str, list[float]],
        sample_epochs: list[int],
        sample_prediction_data: list[dict[str, Any]]
    ) -> None:
        """Test complete InteractivePlotlyVisualizer workflow."""
        visualizer = InteractivePlotlyVisualizer()

        # Test training curves
        training_fig = visualizer.create_interactive_training_curves(
            metrics_data=sample_training_data,
            epochs=sample_epochs
        )
        assert training_fig is not None

        # Test prediction grid
        prediction_fig = visualizer.create_interactive_prediction_grid(
            results=sample_prediction_data
        )
        assert prediction_fig is not None

        # Test confidence map
        confidence_fig = visualizer.create_interactive_confidence_map(
            confidence_data={"confidence_map": sample_prediction_data[0]["confidence_map"]}
        )
        assert confidence_fig is not None

    def test_advanced_prediction_visualizer_workflow(
        self,
        sample_prediction_data: list[dict[str, Any]]
    ) -> None:
        """Test AdvancedPredictionVisualizer workflow."""
        visualizer = AdvancedPredictionVisualizer()

        # Test comparison grid
        comparison_fig = visualizer.create_comparison_grid(
            results=sample_prediction_data
        )
        assert comparison_fig is not None

        # Test confidence map
        confidence_fig = visualizer.create_confidence_map(
            result=sample_prediction_data[0]
        )
        assert confidence_fig is not None

    def test_advanced_training_visualizer_workflow(self) -> None:
        """Test AdvancedTrainingVisualizer workflow."""
        visualizer = AdvancedTrainingVisualizer()

        training_data = {
            "metrics": [
                {"epoch": 1, "loss": 2.0, "val_loss": 1.8, "iou": 0.3, "val_iou": 0.25},
                {"epoch": 2, "loss": 1.5, "val_loss": 1.4, "iou": 0.5, "val_iou": 0.45},
                {"epoch": 3, "loss": 1.2, "val_loss": 1.1, "iou": 0.7, "val_iou": 0.65},
            ]
        }

        # Test training curves
        curves_fig = visualizer.create_training_curves(training_data=training_data)
        assert curves_fig is not None

        # Test learning rate analysis
        lr_fig = visualizer.analyze_learning_rate_schedule(training_data=training_data)
        assert lr_fig is not None

    def test_experiment_visualizer_workflow(self) -> None:
        """Test ExperimentVisualizer workflow."""
        visualizer = ExperimentVisualizer()

        # Test loading experiment data
        experiment_dir = Path("outputs/test_experiment")
        experiment_data = visualizer.load_experiment_data(experiment_dir)
        assert isinstance(experiment_data, dict)

        # Test comparison table
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
                    "experiment_info": {"total_epochs": 100, "best_epoch": 85},
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
                    "experiment_info": {"total_epochs": 100, "best_epoch": 90},
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
```

## Test Data Management

### Realistic Test Data Generation

```python
def generate_realistic_training_data(epochs: int = 100) -> dict[str, list[float]]:
    """Generate realistic training data for testing."""
    import numpy as np

    # Simulate realistic training curves
    base_loss = 2.0
    base_val_loss = 1.8
    base_iou = 0.3
    base_val_iou = 0.25

    loss_curve = [base_loss * np.exp(-i/50) + 0.1 * np.random.random() for i in range(epochs)]
    val_loss_curve = [base_val_loss * np.exp(-i/60) + 0.15 * np.random.random() for i in range(epochs)]
    iou_curve = [base_iou + 0.6 * (1 - np.exp(-i/40)) + 0.02 * np.random.random() for i in range(epochs)]
    val_iou_curve = [base_val_iou + 0.55 * (1 - np.exp(-i/50)) + 0.03 * np.random.random() for i in range(epochs)]

    return {
        "loss": loss_curve,
        "val_loss": val_loss_curve,
        "iou": iou_curve,
        "val_iou": val_iou_curve,
    }

def generate_realistic_prediction_data(num_samples: int = 10) -> list[dict[str, Any]]:
    """Generate realistic prediction data for testing."""
    import numpy as np

    results = []
    for i in range(num_samples):
        # Create realistic image data
        image_size = (512, 512, 3)
        original_image = np.random.randint(0, 255, image_size, dtype=np.uint8)

        # Create realistic prediction mask (sparse cracks)
        prediction_mask = np.zeros((512, 512), dtype=bool)
        # Add some crack-like patterns
        for _ in range(5):
            start_x = np.random.randint(0, 512)
            start_y = np.random.randint(0, 512)
            length = np.random.randint(50, 200)
            angle = np.random.uniform(0, 2 * np.pi)

            for j in range(length):
                x = int(start_x + j * np.cos(angle))
                y = int(start_y + j * np.sin(angle))
                if 0 <= x < 512 and 0 <= y < 512:
                    prediction_mask[x, y] = True

        # Create ground truth (similar but with some differences)
        ground_truth_mask = prediction_mask.copy()
        # Add some noise
        noise_indices = np.random.choice([True, False], prediction_mask.shape, p=[0.05, 0.95])
        ground_truth_mask[noise_indices] = ~ground_truth_mask[noise_indices]

        # Create confidence map
        confidence_map = np.random.random((512, 512))
        confidence_map[prediction_mask] = np.random.uniform(0.7, 1.0, prediction_mask.sum())
        confidence_map[~prediction_mask] = np.random.uniform(0.0, 0.3, (~prediction_mask).sum())

        # Calculate realistic metrics
        intersection = np.logical_and(prediction_mask, ground_truth_mask).sum()
        union = np.logical_or(prediction_mask, ground_truth_mask).sum()
        iou = intersection / union if union > 0 else 0.0

        precision = intersection / prediction_mask.sum() if prediction_mask.sum() > 0 else 0.0
        recall = intersection / ground_truth_mask.sum() if ground_truth_mask.sum() > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append({
            "image_path": f"data/test/images/sample_{i}.jpg",
            "original_image": original_image,
            "prediction_mask": prediction_mask,
            "ground_truth_mask": ground_truth_mask,
            "probability_mask": confidence_map,
            "confidence_map": confidence_map,
            "metrics": {
                "iou": float(iou),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "dice": float(2 * intersection / (prediction_mask.sum() + ground_truth_mask.sum())),
            },
            "iou": float(iou),
            "dice": float(2 * intersection / (prediction_mask.sum() + ground_truth_mask.sum())),
        })

    return results
```

## Mocking Strategies

### Effective Mocking for Visualization Testing

```python
@pytest.fixture
def mock_plotly_figure() -> Mock:
    """Create a mock Plotly figure."""
    mock_fig = Mock()
    mock_fig.data = [Mock(), Mock()]
    mock_fig.layout = Mock()
    mock_fig.layout.title = Mock()
    mock_fig.layout.title.text = "Test Figure"
    mock_fig.layout.annotations = []

    # Mock methods
    mock_fig.show = Mock()
    mock_fig.write_html = Mock()
    mock_fig.write_image = Mock()
    mock_fig.to_json = Mock(return_value='{"type": "Figure"}')

    return mock_fig

@pytest.fixture
def mock_numpy_array() -> Mock:
    """Create a mock NumPy array."""
    mock_array = Mock()
    mock_array.shape = (256, 256, 3)
    mock_array.dtype = np.uint8
    mock_array.__array__ = Mock(return_value=np.random.randint(0, 255, (256, 256, 3)))

    return mock_array

@pytest.fixture
def mock_path() -> Mock:
    """Create a mock Path object."""
    mock_path = Mock()
    mock_path.exists.return_value = True
    mock_path.with_suffix.return_value = mock_path
    mock_path.__str__ = Mock(return_value="/tmp/test_path")

    return mock_path

def test_with_mocks(
    mock_plotly_figure: Mock,
    mock_numpy_array: Mock,
    mock_path: Mock
) -> None:
    """Test using mocked objects."""
    with patch('plotly.graph_objects.Figure', return_value=mock_plotly_figure):
        with patch('pathlib.Path', return_value=mock_path):
            visualizer = InteractivePlotlyVisualizer()

            # Test with mocked data
            training_data = {"loss": [1.0, 0.8, 0.6]}
            epochs = [1, 2, 3]

            fig = visualizer.create_interactive_training_curves(
                metrics_data=training_data,
                epochs=epochs
            )

            assert fig is not None
            mock_plotly_figure.show.assert_not_called()  # Should not be called automatically
```

## Quality Gates

### Pre-Commit Quality Checks

```bash
# Run quality gates on test files
python -m ruff check tests/ --fix
black tests/
basedpyright tests/

# Run quality gates on visualization source files
python -m ruff check src/crackseg/evaluation/visualization/ --fix
black src/crackseg/evaluation/visualization/
basedpyright src/crackseg/evaluation/visualization/
```

### Quality Gate Configuration

```python
# pyproject.toml quality gate settings
[tool.ruff]
target-version = "py312"
line-length = 88
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
    "UP", # pyupgrade
]

[tool.ruff.per-file-ignores]
"tests/**/*.py" = [
    "E501",  # line too long (allow in tests)
    "B011",  # do not use assert False
]

[tool.basedpyright]
pythonVersion = "3.12"
typeCheckingMode = "strict"
```

## Performance Testing

### Large Dataset Performance Tests

```python
import time
import psutil
import gc

class TestVisualizationPerformance:
    """Test visualization performance with large datasets."""

    def test_large_training_data_performance(self) -> None:
        """Test performance with large training dataset."""
        visualizer = InteractivePlotlyVisualizer()

        # Generate large dataset (1000+ epochs)
        large_training_data = generate_realistic_training_data(epochs=1000)
        large_epochs = list(range(1, 1001))

        # Measure memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Measure time
        start_time = time.time()

        fig = visualizer.create_interactive_training_curves(
            metrics_data=large_training_data,
            epochs=large_epochs
        )

        end_time = time.time()
        creation_time = end_time - start_time

        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before

        # Assertions
        assert creation_time < 10.0  # Should complete within 10 seconds
        assert memory_used < 500.0   # Should use less than 500MB additional memory
        assert fig is not None

        # Cleanup
        del fig
        gc.collect()

    def test_many_predictions_performance(self) -> None:
        """Test performance with many prediction results."""
        visualizer = InteractivePlotlyVisualizer()

        # Generate many prediction results
        many_results = generate_realistic_prediction_data(num_samples=100)

        # Measure performance
        start_time = time.time()

        fig = visualizer.create_interactive_prediction_grid(
            results=many_results,
            max_images=50  # Limit for performance
        )

        end_time = time.time()
        creation_time = end_time - start_time

        # Assertions
        assert creation_time < 15.0  # Should complete within 15 seconds
        assert fig is not None

        # Cleanup
        del fig
        gc.collect()
```

## Error Handling Testing

### Comprehensive Error Handling Tests

```python
class TestVisualizationErrorHandling:
    """Test error handling in visualization components."""

    def test_invalid_training_data(self) -> None:
        """Test handling of invalid training data."""
        visualizer = InteractivePlotlyVisualizer()

        # Test with empty data
        with pytest.raises(ValueError, match="Empty metrics data"):
            visualizer.create_interactive_training_curves(
                metrics_data={},
                epochs=[]
            )

        # Test with mismatched data lengths
        with pytest.raises(ValueError, match="Mismatched data lengths"):
            visualizer.create_interactive_training_curves(
                metrics_data={"loss": [1.0, 2.0]},
                epochs=[1, 2, 3]  # Mismatched length
            )

        # Test with invalid data types
        with pytest.raises(TypeError, match="Invalid data type"):
            visualizer.create_interactive_training_curves(
                metrics_data={"loss": "invalid"},
                epochs=[1, 2, 3]
            )

    def test_invalid_prediction_data(self) -> None:
        """Test handling of invalid prediction data."""
        visualizer = InteractivePlotlyVisualizer()

        # Test with missing required keys
        invalid_results = [{"original_image": np.random.randint(0, 255, (256, 256, 3))}]

        with pytest.raises(KeyError, match="Missing required key"):
            visualizer.create_interactive_prediction_grid(results=invalid_results)

        # Test with invalid image dimensions
        invalid_results = [{
            "original_image": np.random.randint(0, 255, (256, 256)),  # Missing channel dimension
            "prediction_mask": np.random.choice([True, False], (256, 256)),
            "ground_truth_mask": np.random.choice([True, False], (256, 256)),
            "confidence_map": np.random.random((256, 256)),
            "metrics": {"iou": 0.75},
        }]

        with pytest.raises(ValueError, match="Invalid image dimensions"):
            visualizer.create_interactive_prediction_grid(results=invalid_results)

    def test_export_errors(self) -> None:
        """Test export error handling."""
        export_handler = ExportHandler()
        mock_fig = Mock()

        # Test unsupported format
        with pytest.raises(ValueError, match="Unsupported format"):
            export_handler.save_plot(mock_fig, Path("/tmp/test"), format="unsupported")

        # Test file system errors
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                export_handler.save_plot(mock_fig, Path("/nonexistent/path"), format="html")

    def test_metadata_errors(self) -> None:
        """Test metadata error handling."""
        metadata_handler = MetadataHandler()

        # Test invalid metadata
        with pytest.raises(ValueError, match="Invalid metadata"):
            metadata_handler.create_metadata(
                plot_type="",  # Empty plot type
                data_info={}
            )

        # Test invalid figure
        with pytest.raises(TypeError, match="Invalid figure object"):
            metadata_handler.embed_metadata_in_figure(None, {"test": "data"})
```

## Best Practices

### Testing Best Practices Summary

1. **Use realistic test data** that mimics production scenarios
2. **Test both success and failure paths** comprehensively
3. **Use type annotations** in all test code
4. **Follow quality gates** (ruff, black, basedpyright)
5. **Mock external dependencies** appropriately
6. **Test performance** with large datasets
7. **Test error handling** for all edge cases
8. **Use fixtures** for reusable test data
9. **Test integration** between components
10. **Document test strategies** and patterns

### Test Organization

```bash
tests/
├── unit/
│   ├── test_interactive_plotly.py
│   ├── test_advanced_prediction_viz.py
│   ├── test_advanced_training_viz.py
│   └── test_experiment_viz.py
├── integration/
│   ├── test_visualization_integration.py
│   └── test_visualization_workflows.py
├── performance/
│   ├── test_large_dataset_performance.py
│   └── test_memory_usage.py
└── fixtures/
    ├── training_data.py
    └── prediction_data.py
```

This comprehensive testing guide ensures the CrackSeg visualization system is thoroughly tested,
reliable, and maintainable.
