"""Unit tests for AdvancedTrainingVisualizer.

Tests the advanced training visualization system including training
curves, learning rate analysis, gradient flow visualization, and
parameter distributions. Also tests integration with ArtifactManager
for proper artifact tracking.
"""

import json
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from matplotlib.figure import Figure

from crackseg.evaluation.visualization.advanced_training_viz import (
    AdvancedTrainingVisualizer,
)
from crackseg.utils.artifact_manager import (
    ArtifactManager,
    ArtifactManagerConfig,
)


class SimpleModel(nn.Module):
    """Simple test model for parameter distribution testing."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 1)
        self.conv = nn.Conv2d(3, 16, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestAdvancedTrainingVisualizer:
    """Test AdvancedTrainingVisualizer functionality."""

    @pytest.fixture
    def temp_dir(self) -> Generator[Path, None, None]:
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def artifact_manager(self, temp_dir: Path) -> ArtifactManager:
        """Create ArtifactManager instance for testing."""
        config = ArtifactManagerConfig(
            base_path=str(temp_dir), experiment_name="test_experiment"
        )
        return ArtifactManager(config)

    @pytest.fixture
    def visualizer(self) -> AdvancedTrainingVisualizer:
        """Create AdvancedTrainingVisualizer instance for testing."""
        return AdvancedTrainingVisualizer(interactive=False)

    @pytest.fixture
    def visualizer_with_artifacts(
        self, artifact_manager: ArtifactManager
    ) -> AdvancedTrainingVisualizer:
        """Create AdvancedTrainingVisualizer with ArtifactManager."""
        return AdvancedTrainingVisualizer(
            interactive=False, artifact_manager=artifact_manager
        )

    @pytest.fixture
    def sample_training_data(self) -> dict:
        """Create sample training data for testing."""
        return {
            "metrics": [
                {"epoch": 1, "loss": 0.5, "accuracy": 0.8},
                {"epoch": 2, "loss": 0.4, "accuracy": 0.85},
                {"epoch": 3, "loss": 0.3, "accuracy": 0.9},
            ],
            "summary": {
                "experiment_info": {"total_epochs": 3, "best_epoch": 3},
                "best_metrics": {
                    "loss": {"value": 0.3},
                    "accuracy": {"value": 0.9},
                },
            },
            "config": {"training": {"learning_rate": 0.001}},
        }

    @pytest.fixture
    def sample_gradient_data(self) -> dict:
        """Create sample gradient data for testing."""
        return {
            "gradients": [
                {
                    "epoch": 1,
                    "layer1": 0.1,
                    "layer2": 0.05,
                    "layer3": 0.02,
                },
                {
                    "epoch": 2,
                    "layer1": 0.08,
                    "layer2": 0.04,
                    "layer3": 0.015,
                },
            ]
        }

    def test_initialization(
        self, visualizer: AdvancedTrainingVisualizer
    ) -> None:
        """Test AdvancedTrainingVisualizer initialization."""
        assert visualizer.interactive is False
        assert "figure_size" in visualizer.style_config
        assert "color_palette" in visualizer.style_config
        assert visualizer.artifact_manager is None

    def test_initialization_with_artifacts(
        self, visualizer_with_artifacts: AdvancedTrainingVisualizer
    ) -> None:
        """Test initialization with ArtifactManager."""
        assert visualizer_with_artifacts.artifact_manager is not None
        assert isinstance(
            visualizer_with_artifacts.artifact_manager, ArtifactManager
        )

    def test_connect_artifact_manager(
        self,
        visualizer: AdvancedTrainingVisualizer,
        artifact_manager: ArtifactManager,
    ) -> None:
        """Test connecting with ArtifactManager."""
        assert visualizer.artifact_manager is None
        visualizer.connect_artifact_manager(artifact_manager)
        assert visualizer.artifact_manager is artifact_manager

    def test_load_training_data_success(
        self, visualizer: AdvancedTrainingVisualizer, temp_dir: Path
    ) -> None:
        """Test loading training data successfully."""
        # Create test data files
        metrics_file = temp_dir / "metrics" / "metrics.jsonl"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_file, "w") as f:
            f.write('{"epoch": 1, "loss": 0.5}\n')
            f.write('{"epoch": 2, "loss": 0.4}\n')

        summary_file = temp_dir / "metrics" / "complete_summary.json"
        with open(summary_file, "w") as f:
            json.dump({"experiment_info": {"total_epochs": 2}}, f)

        data = visualizer.load_training_data(temp_dir)
        assert "metrics" in data
        assert len(data["metrics"]) == 2
        assert "summary" in data

    def test_load_training_data_missing_files(
        self, visualizer: AdvancedTrainingVisualizer, temp_dir: Path
    ) -> None:
        """Test loading training data with missing files."""
        data = visualizer.load_training_data(temp_dir)
        assert data == {}

    def test_create_training_curves_success(
        self,
        visualizer: AdvancedTrainingVisualizer,
        sample_training_data: dict,
    ) -> None:
        """Test creating training curves successfully."""
        fig = visualizer.create_training_curves(sample_training_data)
        assert isinstance(fig, Figure)
        assert hasattr(fig, "savefig")

    def test_create_training_curves_no_metrics(
        self, visualizer: AdvancedTrainingVisualizer
    ) -> None:
        """Test creating training curves with no metrics."""
        data = {"metrics": []}
        fig = visualizer.create_training_curves(data)
        assert isinstance(fig, Figure)

    def test_create_training_curves_specific_metrics(
        self,
        visualizer: AdvancedTrainingVisualizer,
        sample_training_data: dict,
    ) -> None:
        """Test creating training curves with specific metrics."""
        fig = visualizer.create_training_curves(
            sample_training_data, metrics=["loss"]
        )
        assert isinstance(fig, Figure)

    def test_analyze_learning_rate_schedule_success(
        self,
        visualizer: AdvancedTrainingVisualizer,
        sample_training_data: dict,
    ) -> None:
        """Test learning rate schedule analysis successfully."""
        fig = visualizer.analyze_learning_rate_schedule(sample_training_data)
        assert isinstance(fig, Figure)

    def test_analyze_learning_rate_schedule_no_lr_data(
        self, visualizer: AdvancedTrainingVisualizer
    ) -> None:
        """Test learning rate analysis with no LR data."""
        data = {"metrics": [{"epoch": 1, "loss": 0.5}]}
        fig = visualizer.analyze_learning_rate_schedule(data)
        assert isinstance(fig, Figure)

    def test_visualize_gradient_flow_success(
        self,
        visualizer: AdvancedTrainingVisualizer,
        sample_gradient_data: dict,
    ) -> None:
        """Test gradient flow visualization successfully."""
        fig = visualizer.visualize_gradient_flow(sample_gradient_data)
        assert isinstance(fig, Figure)

    def test_visualize_gradient_flow_no_gradient_data(
        self, visualizer: AdvancedTrainingVisualizer
    ) -> None:
        """Test gradient flow visualization with no gradient data."""
        data = {"metrics": [{"epoch": 1, "loss": 0.5}]}
        fig = visualizer.visualize_gradient_flow(data)
        assert isinstance(fig, Figure)

    def test_visualize_parameter_distributions_success(
        self, visualizer: AdvancedTrainingVisualizer, temp_dir: Path
    ) -> None:
        """Test parameter distributions visualization successfully."""
        # Create a simple model and save checkpoint
        model = SimpleModel()
        checkpoint = {"model_state_dict": model.state_dict()}
        checkpoint_path = temp_dir / "model.pth"
        torch.save(checkpoint, checkpoint_path)

        fig = visualizer.visualize_parameter_distributions(checkpoint_path)
        assert isinstance(fig, Figure)

    def test_visualize_parameter_distributions_invalid_checkpoint(
        self, visualizer: AdvancedTrainingVisualizer, temp_dir: Path
    ) -> None:
        """Test parameter distributions with invalid checkpoint."""
        # Create invalid checkpoint
        checkpoint_path = temp_dir / "invalid_model.pth"
        torch.save({"invalid": "data"}, checkpoint_path)

        fig = visualizer.visualize_parameter_distributions(checkpoint_path)
        assert isinstance(fig, Figure)

    def test_extract_parameter_statistics(
        self, visualizer: AdvancedTrainingVisualizer
    ) -> None:
        """Test parameter statistics extraction."""
        model = SimpleModel()
        model_state = model.state_dict()
        stats = visualizer._extract_parameter_statistics(model_state)
        assert isinstance(stats, dict)
        assert len(stats) > 0
        assert "mean" in list(stats.values())[0]
        assert "std" in list(stats.values())[0]

    def test_create_empty_plot(
        self, visualizer: AdvancedTrainingVisualizer
    ) -> None:
        """Test creating empty plot."""
        fig = visualizer._create_empty_plot("Test Title")
        assert isinstance(fig, Figure)
        assert hasattr(fig, "savefig")

    def test_create_comprehensive_report(
        self, visualizer: AdvancedTrainingVisualizer, temp_dir: Path
    ) -> None:
        """Test creating comprehensive report."""
        output_dir = temp_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create minimal experiment structure
        experiment_dir = temp_dir / "experiment"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        (experiment_dir / "metrics").mkdir(exist_ok=True)

        with open(experiment_dir / "metrics" / "metrics.jsonl", "w") as f:
            f.write('{"epoch": 1, "loss": 0.5}\n')

        report_files = visualizer.create_comprehensive_report(
            experiment_dir, output_dir
        )
        assert isinstance(report_files, dict)

    def test_interactive_mode(self) -> None:
        """Test interactive mode with Plotly."""
        try:
            visualizer = AdvancedTrainingVisualizer(interactive=True)
            assert visualizer.interactive is True
        except ImportError:
            pytest.skip("Plotly not available")

    def test_custom_style_config(self) -> None:
        """Test custom style configuration."""
        custom_style = {
            "figure_size": (10, 6),
            "dpi": 300,
            "color_palette": "plasma",
            "grid_alpha": 0.3,
            "line_width": 2,
            "font_size": 14,
            "title_font_size": 16,
            "legend_font_size": 10,
        }
        visualizer = AdvancedTrainingVisualizer(style_config=custom_style)
        assert visualizer.style_config["figure_size"] == (10, 6)
        assert visualizer.style_config["color_palette"] == "plasma"

    def test_save_visualization_with_artifacts(
        self, visualizer_with_artifacts: AdvancedTrainingVisualizer
    ) -> None:
        """Test saving visualization with ArtifactManager."""
        fig = Figure()
        result = visualizer_with_artifacts._save_visualization_with_artifacts(
            fig, "test_plot.png", "Test visualization"
        )
        # Should return None in test environment due to temp file issues
        # but the method should not raise exceptions
        assert result is None or isinstance(result, tuple)

    def test_save_visualization_without_artifacts(
        self, visualizer: AdvancedTrainingVisualizer
    ) -> None:
        """Test saving visualization without ArtifactManager."""
        fig = Figure()
        result = visualizer._save_visualization_with_artifacts(
            fig, "test_plot.png", "Test visualization"
        )
        assert result is None

    def test_error_handling_invalid_data(
        self, visualizer: AdvancedTrainingVisualizer
    ) -> None:
        """Test error handling with invalid data."""
        # Test with None data
        fig = visualizer.create_training_curves({})
        assert isinstance(fig, Figure)

        # Test with malformed data
        fig = visualizer.analyze_learning_rate_schedule({"invalid": "data"})
        assert isinstance(fig, Figure)

    def test_metric_auto_detection(
        self,
        visualizer: AdvancedTrainingVisualizer,
        sample_training_data: dict,
    ) -> None:
        """Test automatic metric detection."""
        fig = visualizer.create_training_curves(sample_training_data)
        assert isinstance(fig, Figure)

    def test_matplotlib_integration(
        self,
        visualizer: AdvancedTrainingVisualizer,
        sample_training_data: dict,
    ) -> None:
        """Test matplotlib integration for static plots."""
        fig = visualizer.create_training_curves(sample_training_data)
        assert isinstance(fig, Figure)
        assert hasattr(fig, "savefig")
        assert hasattr(fig, "get_axes")

    def test_seaborn_integration(
        self, visualizer: AdvancedTrainingVisualizer
    ) -> None:
        """Test seaborn integration for styling."""
        # Test that seaborn styling is applied
        assert visualizer.style_config["color_palette"] == "viridis"

    def test_pytorch_integration(
        self, visualizer: AdvancedTrainingVisualizer, temp_dir: Path
    ) -> None:
        """Test PyTorch integration for model loading."""
        model = SimpleModel()
        checkpoint = {"model_state_dict": model.state_dict()}
        checkpoint_path = temp_dir / "model.pth"
        torch.save(checkpoint, checkpoint_path)

        fig = visualizer.visualize_parameter_distributions(checkpoint_path)
        assert isinstance(fig, Figure)

    def test_plotly_integration(self) -> None:
        """Test Plotly integration for interactive plots."""
        try:
            visualizer = AdvancedTrainingVisualizer(interactive=True)
            sample_data = {
                "metrics": [{"epoch": 1, "loss": 0.5, "accuracy": 0.8}]
            }

            fig = visualizer.create_training_curves(sample_data)

            # Should return a Plotly figure
            assert hasattr(fig, "to_html")

        except ImportError:
            pytest.skip("Plotly not available")
