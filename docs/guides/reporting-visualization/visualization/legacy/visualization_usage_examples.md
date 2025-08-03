# Visualization Usage Examples

Comprehensive examples for using the CrackSeg visualization system.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Interactive Plotly Examples](#interactive-plotly-examples)
3. [Advanced Prediction Examples](#advanced-prediction-examples)
4. [Advanced Training Examples](#advanced-training-examples)
5. [Experiment Visualization Examples](#experiment-visualization-examples)
6. [Export and Metadata Examples](#export-and-metadata-examples)
7. [Custom Templates Examples](#custom-templates-examples)
8. [Error Handling Examples](#error-handling-examples)
9. [Performance Optimization Examples](#performance-optimization-examples)

## Quick Start

### Basic Training Curves

```python
from pathlib import Path
import numpy as np
from crackseg.evaluation.visualization import InteractivePlotlyVisualizer


# Create sample training data
training_data = {
    "loss": [2.0, 1.5, 1.2, 0.9, 0.7],
    "val_loss": [1.8, 1.4, 1.1, 0.8, 0.6],
    "iou": [0.3, 0.5, 0.7, 0.8, 0.85],
    "val_iou": [0.25, 0.45, 0.65, 0.75, 0.8],
}
epochs = [1, 2, 3, 4, 5]

# Initialize visualizer
visualizer = InteractivePlotlyVisualizer(
    export_formats=["html", "png", "pdf"]
)

# Create interactive training curves
fig = visualizer.create_interactive_training_curves(
    metrics_data=training_data,
    epochs=epochs,
    title="Training Progress",
    save_path=Path("outputs/training_curves")
)

# Display the figure
fig.show()
```

### Basic Prediction Grid

```python
import numpy as np
from crackseg.evaluation.visualization import InteractivePlotlyVisualizer


# Create sample prediction data
prediction_data = [
    {
        "original_image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        "prediction_mask": np.random.choice([True, False], (256, 256)),
        "ground_truth_mask": np.random.choice([True, False], (256, 256)),
        "confidence_map": np.random.random((256, 256)),
        "metrics": {"iou": 0.75, "f1": 0.8, "dice": 0.85},
    },
    {
        "original_image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        "prediction_mask": np.random.choice([True, False], (256, 256)),
        "ground_truth_mask": np.random.choice([True, False], (256, 256)),
        "confidence_map": np.random.random((256, 256)),
        "metrics": {"iou": 0.82, "f1": 0.85, "dice": 0.88},
    },
]

# Create prediction grid
fig = visualizer.create_interactive_prediction_grid(
    results=prediction_data,
    max_images=6,
    show_metrics=True,
    show_confidence=True,
    save_path=Path("outputs/prediction_grid")
)

fig.show()
```

## Interactive Plotly Examples

### Real-Time Training Dashboard

```python
from crackseg.evaluation.visualization import InteractivePlotlyVisualizer
import time


# Initialize visualizer with real-time capabilities
visualizer = InteractivePlotlyVisualizer(
    responsive=True,
    export_formats=["html", "json"]
)

# Simulate real-time training updates
for epoch in range(1, 11):
    # Generate training metrics
    training_data = {
        "loss": [2.0 * np.exp(-i/10) + 0.1 * np.random.random() for i in range(epoch)],
        "val_loss": [1.8 * np.exp(-i/12) + 0.15 * np.random.random() for i in range(epoch)],
        "iou": [0.3 + 0.6 * (1 - np.exp(-i/8)) + 0.02 * np.random.random() for i in range(epoch)],
        "val_iou": [0.25 + 0.55 * (1 - np.exp(-i/10)) + 0.03 * np.random.random() for i in range(epoch)],
    }
    epochs = list(range(1, epoch + 1))

    # Create updated visualization
    fig = visualizer.create_interactive_training_curves(
        metrics_data=training_data,
        epochs=epochs,
        title=f"Training Progress - Epoch {epoch}"
    )

    # Save with timestamp
    fig.write_html(f"outputs/training_epoch_{epoch}.html")

    print(f"Epoch {epoch} completed")
    time.sleep(1)  # Simulate training time
```

### Confidence Map Visualization

```python
import numpy as np
from crackseg.evaluation.visualization import InteractivePlotlyVisualizer


# Create confidence data
confidence_data = {
    "confidence_map": np.random.random((512, 512)),
    "original_image": np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
    "prediction_mask": np.random.choice([True, False], (512, 512)),
}

# Create confidence map
fig = visualizer.create_interactive_confidence_map(
    confidence_data=confidence_data,
    save_path=Path("outputs/confidence_map")
)

fig.show()
```

## Advanced Prediction Examples

### Comprehensive Prediction Analysis

```python
from crackseg.evaluation.visualization import AdvancedPredictionVisualizer
import numpy as np
from pathlib import Path


# Initialize advanced prediction visualizer
visualizer = AdvancedPredictionVisualizer()

# Create comprehensive prediction results
prediction_results = [
    {
        "image_path": "data/test/images/sample_1.jpg",
        "original_image": np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
        "prediction_mask": np.random.choice([True, False], (512, 512)),
        "ground_truth_mask": np.random.choice([True, False], (512, 512)),
        "probability_mask": np.random.random((512, 512)),
        "confidence_map": np.random.random((512, 512)),
        "metrics": {
            "iou": 0.75,
            "f1": 0.8,
            "dice": 0.85,
            "precision": 0.82,
            "recall": 0.78,
        },
    },
    {
        "image_path": "data/test/images/sample_2.jpg",
        "original_image": np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
        "prediction_mask": np.random.choice([True, False], (512, 512)),
        "ground_truth_mask": np.random.choice([True, False], (512, 512)),
        "probability_mask": np.random.random((512, 512)),
        "confidence_map": np.random.random((512, 512)),
        "metrics": {
            "iou": 0.82,
            "f1": 0.85,
            "dice": 0.88,
            "precision": 0.86,
            "recall": 0.84,
        },
    },
]

# Create comparison grid
comparison_fig = visualizer.create_comparison_grid(
    results=prediction_results,
    max_images=4,
    show_metrics=True,
    show_confidence=True,
    save_path=Path("outputs/prediction_comparison")
)

# Create confidence map for first result
confidence_fig = visualizer.create_confidence_map(
    result=prediction_results[0],
    show_original=True,
    show_contours=True,
    save_path=Path("outputs/confidence_analysis")
)

# Create error analysis
error_fig = visualizer.create_error_analysis(
    result=prediction_results[0],
    save_path=Path("outputs/error_analysis")
)

# Create segmentation overlay
overlay_fig = visualizer.create_segmentation_overlay(
    result=prediction_results[0],
    show_confidence=True,
    save_path=Path("outputs/segmentation_overlay")
)

# Create tabular comparison
tabular_fig = visualizer.create_tabular_comparison(
    results=prediction_results,
    save_path=Path("outputs/tabular_comparison")
)
```

## Advanced Training Examples

### Comprehensive Training Analysis

```python
from crackseg.evaluation.visualization import AdvancedTrainingVisualizer
from pathlib import Path
import numpy as np


# Initialize advanced training visualizer
visualizer = AdvancedTrainingVisualizer(interactive=True)

# Create comprehensive training data
training_data = {
    "metrics": [
        {
            "epoch": 1,
            "loss": 2.0,
            "val_loss": 1.8,
            "iou": 0.3,
            "val_iou": 0.25,
            "learning_rate": 0.001,
            "gradient_norm": 1.2,
        },
        {
            "epoch": 2,
            "loss": 1.5,
            "val_loss": 1.4,
            "iou": 0.5,
            "val_iou": 0.45,
            "learning_rate": 0.001,
            "gradient_norm": 0.9,
        },
        {
            "epoch": 3,
            "loss": 1.2,
            "val_loss": 1.1,
            "iou": 0.7,
            "val_iou": 0.65,
            "learning_rate": 0.0005,
            "gradient_norm": 0.7,
        },
    ]
}

# Create training curves
curves_fig = visualizer.create_training_curves(
    training_data=training_data,
    metrics=["loss", "val_loss", "iou", "val_iou"],
    save_path=Path("outputs/training_curves_advanced")
)

# Analyze learning rate schedule
lr_fig = visualizer.analyze_learning_rate_schedule(
    training_data=training_data,
    save_path=Path("outputs/learning_rate_analysis")
)

# Visualize parameter distributions
param_fig = visualizer.visualize_parameter_distributions(
    model_path=Path("outputs/checkpoints/model_best.pth.tar"),
    save_path=Path("outputs/parameter_distributions")
)

# Visualize gradient flow
gradient_data = {
    "metrics": training_data["metrics"],
    "layer_gradients": {
        "encoder": [1.2, 0.9, 0.7],
        "decoder": [0.8, 0.6, 0.5],
        "classifier": [0.5, 0.4, 0.3],
    }
}

gradient_fig = visualizer.visualize_gradient_flow(
    gradient_data=gradient_data,
    save_path=Path("outputs/gradient_flow")
)
```

## Experiment Visualization Examples

### Experiment Comparison and Analysis

```python
from crackseg.evaluation.visualization import ExperimentVisualizer
from pathlib import Path
import pandas as pd


# Initialize experiment visualizer
visualizer = ExperimentVisualizer()

# Load experiment data
experiment_dir = Path("outputs/experiments/experiment_001")
experiment_data = visualizer.load_experiment_data(experiment_dir)

# Create comparison data for multiple experiments
experiments_data = {
    "baseline": {
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
    "improved_model": {
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
    "data_augmentation": {
        "summary": {
            "best_metrics": {
                "loss": {"value": 0.35},
                "iou": {"value": 0.87},
                "f1": {"value": 0.90},
                "precision": {"value": 0.88},
                "recall": {"value": 0.92},
            },
            "experiment_info": {
                "total_epochs": 100,
                "best_epoch": 95,
            },
        }
    },
}

# Create comparison table
comparison_df = visualizer.create_comparison_table(experiments_data)
print("Experiment Comparison:")
print(comparison_df)

# Find experiment directories
experiment_dirs = visualizer.find_experiment_directories()
print(f"Found {len(experiment_dirs)} experiment directories")
```

## Export and Metadata Examples

### Multi-Format Export with Metadata

```python
from crackseg.evaluation.visualization import InteractivePlotlyVisualizer
from pathlib import Path
import json


# Initialize visualizer with multiple export formats
visualizer = InteractivePlotlyVisualizer(
    export_formats=["html", "png", "pdf", "svg", "json"]
)

# Create sample data
training_data = {
    "loss": [2.0, 1.5, 1.2, 0.9, 0.7],
    "val_loss": [1.8, 1.4, 1.1, 0.8, 0.6],
    "iou": [0.3, 0.5, 0.7, 0.8, 0.85],
    "val_iou": [0.25, 0.45, 0.65, 0.75, 0.8],
}
epochs = [1, 2, 3, 4, 5]

# Create visualization
fig = visualizer.create_interactive_training_curves(
    metrics_data=training_data,
    epochs=epochs,
    title="Training Progress with Metadata"
)

# Create metadata
metadata = visualizer.metadata_handler.create_metadata(
    plot_type="training_curves",
    data_info={
        "epochs": len(epochs),
        "metrics": list(training_data.keys()),
        "experiment_id": "exp_001",
        "model_version": "v1.0",
    },
    custom_metadata={
        "author": "CrackSeg Team",
        "description": "Training curves for crack segmentation model",
        "tags": ["training", "crack-segmentation", "deep-learning"],
    }
)

# Save with metadata
save_path = Path("outputs/training_with_metadata")
visualizer.export_handler.save_plot(fig, save_path, metadata=metadata)

# Save metadata separately
metadata_path = save_path.with_suffix(".json")
visualizer.metadata_handler.save_metadata(metadata, metadata_path)

print(f"Visualization saved with metadata to {save_path}")
print(f"Metadata saved to {metadata_path}")
```

### Metadata Extraction and Analysis

```python
from crackseg.evaluation.visualization import InteractivePlotlyVisualizer


# Initialize visualizer
visualizer = InteractivePlotlyVisualizer()

# Create figure with embedded metadata
fig = visualizer.create_interactive_training_curves(
    metrics_data={"loss": [1.0, 0.8, 0.6]},
    epochs=[1, 2, 3]
)

# Embed metadata in figure
metadata = {
    "experiment_id": "exp_002",
    "model_config": "swin_v2_tiny",
    "dataset": "crack_dataset_v1",
    "training_date": "2024-01-15",
}

fig_with_metadata = visualizer.metadata_handler.embed_metadata_in_figure(fig, metadata)

# Extract metadata from figure
extracted_metadata = visualizer.metadata_handler.extract_metadata_from_figure(fig_with_metadata)
print("Extracted metadata:", extracted_metadata)

# Validate metadata
is_valid = visualizer.metadata_handler.validate_metadata(extracted_metadata)
print(f"Metadata is valid: {is_valid}")
```

## Custom Templates Examples

### Creating Custom Training Template

```python
from crackseg.evaluation.visualization.templates import BaseVisualizationTemplate

from crackseg.evaluation.visualization import InteractivePlotlyVisualizer

import plotly.graph_objects as go

class CustomTrainingTemplate(BaseVisualizationTemplate):
    """Custom template for professional training visualizations."""

    def __init__(self) -> None:
        """Initialize custom template."""
        super().__init__({
            "figure_size": (14, 10),
            "color_palette": "plasma",
            "line_width": 3,
            "font_size": 14,
            "dpi": 150,
            "grid_alpha": 0.2,
            "title_font_size": 18,
            "legend_font_size": 12,
            "background_color": "#ffffff",
            "grid_color": "#e0e0e0",
            "text_color": "#2c3e50",
            "primary_color": "#3498db",
            "secondary_color": "#e74c3c",
            "success_color": "#27ae60",
        })

    def apply_training_curves_style(self, fig: go.Figure) -> go.Figure:
        """Apply custom style to training curves."""
        fig.update_layout(
            plot_bgcolor=self.config["background_color"],
            paper_bgcolor=self.config["background_color"],
            font=dict(
                size=self.config["font_size"],
                color=self.config["text_color"]
            ),
            title=dict(
                font=dict(size=self.config["title_font_size"])
            ),
            legend=dict(
                font=dict(size=self.config["legend_font_size"])
            ),
            xaxis=dict(
                gridcolor=self.config["grid_color"],
                gridwidth=1,
                showgrid=True,
            ),
            yaxis=dict(
                gridcolor=self.config["grid_color"],
                gridwidth=1,
                showgrid=True,
            ),
        )

        # Update traces with custom colors
        colors = [self.config["primary_color"], self.config["secondary_color"],
                 self.config["success_color"], "#f39c12"]

        for i, trace in enumerate(fig.data):
            trace.line.width = self.config["line_width"]
            trace.line.color = colors[i % len(colors)]
            trace.marker.size = 8
            trace.marker.color = colors[i % len(colors)]

        return fig

# Use custom template
custom_template = CustomTrainingTemplate()
visualizer = InteractivePlotlyVisualizer(
    template=custom_template,
    export_formats=["html", "png", "pdf"]
)

# Create visualization with custom styling
training_data = {
    "loss": [2.0, 1.5, 1.2, 0.9, 0.7],
    "val_loss": [1.8, 1.4, 1.1, 0.8, 0.6],
    "iou": [0.3, 0.5, 0.7, 0.8, 0.85],
    "val_iou": [0.25, 0.45, 0.65, 0.75, 0.8],
}
epochs = [1, 2, 3, 4, 5]

fig = visualizer.create_interactive_training_curves(
    metrics_data=training_data,
    epochs=epochs,
    title="Professional Training Progress",
    save_path=Path("outputs/custom_training")
)

fig.show()
```

## Error Handling Examples

### Robust Error Handling

```python
from crackseg.evaluation.visualization import InteractivePlotlyVisualizer
from pathlib import Path
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize visualizer with error handling
visualizer = InteractivePlotlyVisualizer()

def safe_visualization_creation(data, epochs, title):
    """Safely create visualization with error handling."""
    try:
        fig = visualizer.create_interactive_training_curves(
            metrics_data=data,
            epochs=epochs,
            title=title
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        # Create empty figure as fallback
        import plotly.graph_objects as go
        fallback_fig = go.Figure()
        fallback_fig.add_annotation(
            text=f"Error creating visualization: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        return fallback_fig

# Test with invalid data
invalid_data = {"invalid": "data"}
empty_epochs = []

fig = safe_visualization_creation(invalid_data, empty_epochs, "Invalid Data Test")
fig.show()

# Test with valid data
valid_data = {
    "loss": [2.0, 1.5, 1.2],
    "val_loss": [1.8, 1.4, 1.1],
}
valid_epochs = [1, 2, 3]

fig = safe_visualization_creation(valid_data, valid_epochs, "Valid Data Test")
fig.show()
```

## Performance Optimization Examples

### Large Dataset Handling

```python
from crackseg.evaluation.visualization import InteractivePlotlyVisualizer
import numpy as np
from pathlib import Path


# Initialize visualizer optimized for large datasets
visualizer = InteractivePlotlyVisualizer(
    responsive=True,
    export_formats=["html", "json"]  # JSON is faster for large datasets
)

# Create large training dataset (1000+ epochs)
large_training_data = {
    "loss": [2.0 * np.exp(-i/100) + 0.1 * np.random.random() for i in range(1000)],
    "val_loss": [1.8 * np.exp(-i/120) + 0.15 * np.random.random() for i in range(1000)],
    "iou": [0.3 + 0.6 * (1 - np.exp(-i/80)) + 0.02 * np.random.random() for i in range(1000)],
    "val_iou": [0.25 + 0.55 * (1 - np.exp(-i/100)) + 0.03 * np.random.random() for i in range(1000)],
}
large_epochs = list(range(1, 1001))

# Create visualization with performance optimizations
fig = visualizer.create_interactive_training_curves(
    metrics_data=large_training_data,
    epochs=large_epochs,
    title="Large Dataset Training Progress",
    save_path=Path("outputs/large_dataset_training")
)

print("Large dataset visualization created successfully")
fig.show()

# Create many prediction results
many_results = [
    {
        "original_image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        "prediction_mask": np.random.choice([True, False], (256, 256)),
        "ground_truth_mask": np.random.choice([True, False], (256, 256)),
        "confidence_map": np.random.random((256, 256)),
        "metrics": {"iou": 0.75, "f1": 0.8, "dice": 0.85},
    }
    for _ in range(50)  # 50 prediction results
]

# Create prediction grid with performance optimizations
fig = visualizer.create_interactive_prediction_grid(
    results=many_results,
    max_images=20,  # Limit for performance
    show_metrics=True,
    show_confidence=False,  # Disable for performance
    save_path=Path("outputs/many_predictions")
)

print("Many predictions visualization created successfully")
fig.show()
```

### Memory Management

```python
import gc
from crackseg.evaluation.visualization import InteractivePlotlyVisualizer
import numpy as np


def create_memory_efficient_visualizations():
    """Create visualizations with proper memory management."""
    visualizer = InteractivePlotlyVisualizer()

    # Create multiple visualizations with memory cleanup
    for i in range(10):
        # Generate data
        training_data = {
            "loss": [2.0 * np.exp(-j/20) + 0.1 * np.random.random() for j in range(100)],
            "val_loss": [1.8 * np.exp(-j/25) + 0.15 * np.random.random() for j in range(100)],
        }
        epochs = list(range(1, 101))

        # Create visualization
        fig = visualizer.create_interactive_training_curves(
            metrics_data=training_data,
            epochs=epochs,
            title=f"Training Progress {i+1}"
        )

        # Save visualization
        fig.write_html(f"outputs/training_{i+1}.html")

        # Clean up memory
        del fig
        gc.collect()

        print(f"Created visualization {i+1}/10")

# Run memory-efficient visualization creation
create_memory_efficient_visualizations()
print("All visualizations created with memory management")
```

## Best Practices Summary

1. **Use appropriate export formats** for your use case
2. **Include metadata** for reproducibility
3. **Validate data** before visualization
4. **Use templates** for consistent styling
5. **Handle errors gracefully** in production code
6. **Monitor memory usage** with large datasets
7. **Test visualizations** with sample data first
8. **Use type annotations** for better code quality
9. **Follow quality gates** (ruff, black, basedpyright)
10. **Document custom templates** and configurations

This comprehensive guide provides practical examples for all aspects of the CrackSeg visualization
system, from basic usage to advanced customization and performance optimization.
