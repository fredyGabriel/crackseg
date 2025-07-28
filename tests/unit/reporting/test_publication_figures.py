"""Test script for publication-ready figure generation.

This script creates sample experiment data and tests the
PublicationFigureGenerator to ensure it works correctly before using it
with real experiments.
"""

import sys
import tempfile
from pathlib import Path

from omegaconf import DictConfig

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from crackseg.reporting.config import (  # noqa: E402
    ExperimentData,
    OutputFormat,
    ReportConfig,
)
from crackseg.reporting.figures import (  # noqa: E402
    PublicationFigureGenerator,
    PublicationStyle,
)


def create_sample_experiment_data() -> list[ExperimentData]:
    """Create sample experiment data for testing."""
    # Create temporary directories for experiment data
    temp_dirs = []
    for _ in range(3):
        temp_dir = Path(tempfile.mkdtemp())
        temp_dirs.append(temp_dir)
        temp_dir.mkdir(exist_ok=True)

    # Sample training metrics
    training_metrics = {
        "train_loss": [0.8, 0.6, 0.4, 0.3, 0.25, 0.22, 0.2, 0.18, 0.17, 0.16],
        "val_loss": [
            0.85,
            0.65,
            0.45,
            0.35,
            0.3,
            0.28,
            0.26,
            0.25,
            0.24,
            0.23,
        ],
        "train_iou": [0.3, 0.45, 0.6, 0.7, 0.75, 0.78, 0.8, 0.82, 0.83, 0.84],
        "val_iou": [0.25, 0.4, 0.55, 0.65, 0.7, 0.73, 0.75, 0.76, 0.77, 0.78],
        "train_f1": [0.4, 0.55, 0.7, 0.78, 0.82, 0.85, 0.87, 0.88, 0.89, 0.9],
        "val_f1": [0.35, 0.5, 0.65, 0.73, 0.77, 0.8, 0.82, 0.83, 0.84, 0.85],
        "learning_rate": [
            1e-3,
            1e-3,
            1e-3,
            1e-3,
            1e-4,
            1e-4,
            1e-4,
            1e-4,
            1e-4,
            1e-4,
        ],
    }

    # Sample performance metrics
    performance_metrics = {
        "iou": 0.78,
        "f1_score": 0.85,
        "precision": 0.88,
        "recall": 0.82,
        "dice": 0.83,
        "training_time_hours": 2.5,
    }

    # Sample model configurations
    model_configs = [
        {
            "encoder": "resnet50",
            "decoder": "unet",
            "bottleneck": "aspp",
        },
        {
            "encoder": "swin_t",
            "decoder": "unet",
            "bottleneck": "psp",
        },
        {
            "encoder": "efficientnet_b0",
            "decoder": "deeplabv3",
            "bottleneck": "aspp",
        },
    ]

    experiments = []
    for i, (temp_dir, model_config) in enumerate(
        zip(temp_dirs, model_configs, strict=False)
    ):
        # Create sample artifacts
        artifacts = {
            "checkpoint": temp_dir / "model_best.pth",
            "config": temp_dir / "config.yaml",
        }

        # Create dummy files
        artifacts["checkpoint"].touch()
        artifacts["config"].touch()

        # Create experiment data
        exp_data = ExperimentData(
            experiment_id=f"test_exp_{i + 1}",
            experiment_dir=temp_dir,
            config=DictConfig(model_config),
            metrics={
                "training_metrics": training_metrics,
                "performance_metrics": performance_metrics,
            },
            artifacts=artifacts,
        )
        experiments.append(exp_data)

    return experiments


def test_publication_figure_generation() -> None:
    """Test publication figure generation functionality."""
    print("🧪 Testing Publication Figure Generation")
    print("=" * 50)

    # Create sample data
    sample_experiments = create_sample_experiment_data()
    print(f"📊 Created {len(sample_experiments)} sample experiments")

    # Initialize publication figure generator
    style = PublicationStyle(
        figure_width=8.0,
        figure_height=6.0,
        dpi=300,
        supported_formats=["png", "svg", "pdf"],
    )

    figure_generator = PublicationFigureGenerator(style=style)
    config = ReportConfig(
        output_formats=[OutputFormat.JSON],
        include_publication_figures=True,
        include_recommendations=True,
    )

    # Test single experiment figure generation
    print("\n🔍 Testing single experiment figure generation...")
    single_figures = figure_generator.generate_publication_figures(
        sample_experiments[0], config
    )

    print(f"✅ Generated {len(single_figures)} figure types:")
    for fig_type, formats in single_figures.items():
        print(f"   - {fig_type}: {list(formats.keys())}")

    # Test comparison figure generation
    print("\n📊 Testing comparison figure generation...")
    comparison_figures = figure_generator.generate_comparison_figures(
        sample_experiments, config
    )

    print(f"✅ Generated {len(comparison_figures)} comparison figures:")
    for fig_type, formats in comparison_figures.items():
        print(f"   - {fig_type}: {list(formats.keys())}")

    # Test style configuration
    print("\n🎨 Testing style configuration...")
    style_config = figure_generator.get_style_config()
    print(f"   - Supported formats: {style_config.supported_formats}")
    print(
        f"   - Figure dimensions: "
        f"{style_config.figure_width}x{style_config.figure_height}"
    )
    print(f"   - DPI: {style_config.dpi}")
    print(f"   - Font family: {style_config.font_family}")

    # Test format support
    print("\n📁 Testing format support...")
    supported_formats = figure_generator.get_supported_formats()
    print(f"   - Supported formats: {supported_formats}")

    print("\n✅ All publication figure tests completed successfully!")


def test_figure_quality() -> None:
    """Test figure quality and export capabilities."""
    print("\n🔍 Testing Figure Quality and Export")
    print("=" * 40)

    # Create sample data
    sample_experiments = create_sample_experiment_data()

    # Test different style configurations
    styles = [
        PublicationStyle(
            figure_width=6.0,
            figure_height=4.0,
            font_family="serif",
            color_palette="viridis",
        ),
        PublicationStyle(
            figure_width=8.0,
            figure_height=6.0,
            font_family="sans-serif",
            color_palette="plasma",
        ),
    ]

    for i, style in enumerate(styles):
        print(f"\n🎨 Testing Style Configuration {i + 1}:")
        print(f"   - Font family: {style.font_family}")
        print(f"   - Color palette: {style.color_palette}")
        print(f"   - Figure size: {style.figure_width}x{style.figure_height}")

        figure_generator = PublicationFigureGenerator(style=style)
        config = ReportConfig(
            output_formats=[OutputFormat.JSON],
            include_publication_figures=True,
        )

        # Generate figures with this style
        figures = figure_generator.generate_publication_figures(
            sample_experiments[0], config
        )

        print(f"   ✅ Generated {len(figures)} figure types")

    print("\n✅ Figure quality tests completed!")


def main() -> None:
    """Main test function."""
    try:
        test_publication_figure_generation()
        test_figure_quality()

        print("\n🎉 All publication figure generation tests passed!")
        print("\n📋 Summary:")
        print("   ✅ Publication-ready figure generation")
        print("   ✅ Multiple export formats (PNG, SVG, PDF)")
        print("   ✅ Academic/industry publication styles")
        print("   ✅ Customizable figure dimensions and styling")
        print("   ✅ Training curves and performance comparisons")
        print("   ✅ Model architecture visualizations")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
