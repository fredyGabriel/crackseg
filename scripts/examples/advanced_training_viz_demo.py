"""Demo script for AdvancedTrainingVisualizer.

This script demonstrates the usage of the AdvancedTrainingVisualizer
with integration to the ArtifactManager system for proper artifact tracking.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn

from crackseg.evaluation.visualization.advanced_training_viz import (
    AdvancedTrainingVisualizer,
)
from crackseg.utils.artifact_manager import (
    ArtifactManager,
    ArtifactManagerConfig,
)


class DemoModel(nn.Module):
    """Simple demo model for testing visualizations."""

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 1)
        self.conv = nn.Conv2d(3, 16, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.linear1(x))


def create_sample_training_data() -> dict:
    """Create sample training data for demonstration."""
    metrics_data = []
    for epoch in range(1, 21):
        # Simulate realistic training curves
        train_loss = (
            0.5 * torch.exp(-torch.tensor(epoch / 20.0))
            + 0.1 * torch.randn(1) * 0.02
        )
        val_loss = train_loss + 0.05 * torch.randn(1)
        accuracy = (
            0.8
            + 0.15 * (1 - torch.exp(-torch.tensor(epoch / 15.0)))
            + 0.02 * torch.randn(1)
        )
        iou = (
            0.6
            + 0.3 * (1 - torch.exp(-torch.tensor(epoch / 12.0)))
            + 0.03 * torch.randn(1)
        )

        metrics_data.append(
            {
                "epoch": epoch,
                "loss": float(train_loss),
                "val_loss": float(val_loss),
                "accuracy": float(accuracy),
                "iou": float(iou),
                "lr": 0.001 * (0.9 ** (epoch - 1)),  # Learning rate decay
            }
        )

    summary_data = {
        "experiment_info": {
            "total_epochs": 20,
            "best_epoch": 18,
            "training_time": "00:45:30",
        },
        "best_metrics": {
            "loss": {"value": 0.12, "epoch": 18},
            "val_loss": {"value": 0.15, "epoch": 18},
            "accuracy": {"value": 0.94, "epoch": 18},
            "iou": {"value": 0.89, "epoch": 18},
        },
        "final_metrics": {
            "loss": 0.12,
            "val_loss": 0.15,
            "accuracy": 0.94,
            "iou": 0.89,
        },
    }

    config_data = {
        "model": {
            "architecture": "unet",
            "encoder": "resnet34",
            "decoder": "unet_decoder",
        },
        "training": {
            "learning_rate": 0.001,
            "batch_size": 16,
            "epochs": 20,
            "optimizer": "adam",
            "scheduler": "step_lr",
        },
        "data": {
            "train_size": 1000,
            "val_size": 200,
            "test_size": 100,
        },
    }

    return {
        "metrics": metrics_data,
        "summary": summary_data,
        "config": config_data,
    }


def create_sample_gradient_data() -> dict:
    """Create sample gradient data for demonstration."""
    gradient_data = []
    layer_names = [
        "encoder.conv1",
        "encoder.conv2",
        "encoder.conv3",
        "decoder.conv1",
        "decoder.conv2",
        "decoder.conv3",
    ]

    for epoch in range(1, 21):
        epoch_gradients = {}
        for i, layer in enumerate(layer_names):
            # Simulate realistic gradient norms
            base_norm = 0.1 * torch.exp(-torch.tensor(epoch / 25.0))
            noise = 0.02 * torch.randn(1)
            epoch_gradients[layer] = float(base_norm + noise + 0.01 * i)

        gradient_data.append({"epoch": epoch, **epoch_gradients})

    return {"gradients": gradient_data}


def create_sample_model_checkpoint() -> dict:
    """Create a sample model checkpoint for demonstration."""
    model = DemoModel()

    # Create optimizer state
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.step()  # Generate some optimizer state

    checkpoint = {
        "epoch": 20,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": 0.12,
        "accuracy": 0.94,
        "iou": 0.89,
    }

    return checkpoint


def demonstrate_visualizer() -> None:
    """Demonstrate the AdvancedTrainingVisualizer functionality."""
    print("🚀 AdvancedTrainingVisualizer Demo")
    print("=" * 50)

    # Use project directories instead of temporary directories
    project_root = Path(__file__).parent.parent.parent
    demo_dir = project_root / "outputs" / "demo_experiment"
    demo_dir.mkdir(parents=True, exist_ok=True)

    # Initialize ArtifactManager with project directory
    artifact_config = ArtifactManagerConfig(
        base_path=str(demo_dir), experiment_name="demo_experiment"
    )
    artifact_manager = ArtifactManager(artifact_config)

    # Initialize visualizer with custom styling
    custom_style = {
        "figure_size": (14, 10),
        "dpi": 300,
        "color_palette": "plasma",
        "grid_alpha": 0.3,
        "line_width": 2,
        "font_size": 12,
        "title_font_size": 16,
        "legend_font_size": 10,
    }

    visualizer = AdvancedTrainingVisualizer(
        style_config=custom_style,
        interactive=True,
        artifact_manager=artifact_manager,
    )

    print("✅ Visualizer initialized with ArtifactManager integration")

    # Create sample data
    training_data = create_sample_training_data()

    # Create experiment directory structure within project
    experiment_dir = demo_dir / "experiment_data"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "metrics").mkdir(exist_ok=True)

    # Save sample data files
    with open(experiment_dir / "metrics" / "metrics.jsonl", "w") as f:
        for entry in training_data["metrics"]:
            f.write(json.dumps(entry) + "\n")

    with open(experiment_dir / "metrics" / "complete_summary.json", "w") as f:
        json.dump(training_data["summary"], f, indent=2)

    with open(experiment_dir / "config.yaml", "w") as f:
        json.dump(training_data["config"], f, indent=2)

    # Create model checkpoint
    checkpoint = create_sample_model_checkpoint()
    checkpoint_path = experiment_dir / "model_best.pth"
    torch.save(checkpoint, checkpoint_path)

    print("✅ Sample data created and saved")

    # Load training data
    loaded_data = visualizer.load_training_data(experiment_dir)
    print(
        f"✅ Training data loaded: "
        f"{len(loaded_data.get('metrics', []))} epochs"
    )

    # Create visualizations within project structure
    output_dir = demo_dir / "visualizations"
    output_dir.mkdir(exist_ok=True)

    print("\n📊 Creating visualizations...")

    # 1. Training curves
    print("  - Training curves...")
    visualizer.create_training_curves(
        loaded_data, save_path=output_dir / "training_curves.png"
    )
    print("    ✅ Training curves created")

    # 2. Learning rate analysis
    print("  - Learning rate analysis...")
    visualizer.analyze_learning_rate_schedule(
        loaded_data, save_path=output_dir / "learning_rate_analysis.png"
    )
    print("    ✅ Learning rate analysis created")

    # 3. Parameter distributions
    print("  - Parameter distributions...")
    visualizer.visualize_parameter_distributions(
        checkpoint_path,
        save_path=output_dir / "parameter_distributions.png",
    )
    print("    ✅ Parameter distributions created")

    # 4. Comprehensive report
    print("  - Comprehensive report...")
    visualizer.create_comprehensive_report(
        experiment_dir, output_dir, include_gradients=False
    )
    print("    ✅ Comprehensive report created")

    # List generated files
    print(f"\n📁 Generated files in {output_dir}:")
    for file_path in output_dir.iterdir():
        if file_path.is_file():
            print(f"  - {file_path.name}")

    # Show artifact manager integration
    print("\n📦 ArtifactManager metadata:")
    for artifact in artifact_manager.metadata:
        print(f"  - {artifact.artifact_type}: {artifact.description}")

    print("\n✅ Demo completed successfully!")
    print(f"📂 All files saved in project directory: {demo_dir}")
    print(f"📂 Relative to project root: {demo_dir.relative_to(project_root)}")


def main() -> None:
    """Main demo function."""
    try:
        demonstrate_visualizer()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
