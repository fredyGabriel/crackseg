# Visualization API Reference

Comprehensive API documentation for the CrackSeg visualization system.

## Table of Contents

1. [Interactive Plotly Visualizer](#interactive-plotly-visualizer)
2. [Advanced Prediction Visualizer](#advanced-prediction-visualizer)
3. [Advanced Training Visualizer](#advanced-training-visualizer)
4. [Experiment Visualizer](#experiment-visualizer)
5. [Export Handlers](#export-handlers)
6. [Metadata Handlers](#metadata-handlers)
7. [Usage Examples](#usage-examples)
8. [Configuration Guide](#configuration-guide)

## Interactive Plotly Visualizer

The `InteractivePlotlyVisualizer` provides interactive Plotly visualizations with real-time
capabilities, zoom, hover information, and export functionality.

### Class: InteractivePlotlyVisualizer

```python
class InteractivePlotlyVisualizer:
    """Interactive Plotly visualizer with advanced features."""

    def __init__(
        self,
        template: BaseVisualizationTemplate | None = None,
        responsive: bool = True,
        export_formats: list[str] | None = None,
    ) -> None:
        """Initialize the interactive Plotly visualizer.

        Args:
            template: Optional template for consistent styling.
            responsive: Whether to make plots responsive.
            export_formats: List of export formats (html, png, pdf, svg, jpg, json).
        """
```

### Methods

#### create_interactive_training_curves

```python
def create_interactive_training_curves(
    self,
    training_data: dict[str, Any],
    metrics: list[str] | None = None,
    real_time: bool = False,
    save_path: Path | None = None,
) -> PlotlyFigure:
    """Create interactive training curves with Plotly.

    Args:
        training_data: Training data dictionary with metrics.
        metrics: List of metrics to plot (default: ["loss", "val_loss", "iou", "val_iou"]).
        real_time: Whether to enable real-time updates.
        save_path: Optional path to save the visualization.

    Returns:
        Interactive Plotly figure with training curves.

    Example:
        >>> training_data = {
        ...     "metrics": [
        ...         {"loss": 2.0, "val_loss": 1.8, "iou": 0.3, "val_iou": 0.25},
        ...         {"loss": 1.5, "val_loss": 1.4, "iou": 0.5, "val_iou": 0.45},
        ...     ]
        ... }
        >>> visualizer = InteractivePlotlyVisualizer()
        >>> fig = visualizer.create_interactive_training_curves(training_data)
    """
```

#### create_interactive_prediction_grid

```python
def create_interactive_prediction_grid(
    self,
    results: list[dict[str, Any]],
    max_images: int = 9,
    show_metrics: bool = True,
    show_confidence: bool = True,
    save_path: Path | None = None,
) -> PlotlyFigure:
    """Create interactive prediction comparison grid.

    Args:
        results: List of prediction result dictionaries.
        max_images: Maximum number of images to display.
        show_metrics: Whether to show metrics overlay.
        show_confidence: Whether to show confidence maps.
        save_path: Optional path to save the visualization.

    Returns:
        Interactive Plotly figure with prediction grid.

    Example:
        >>> results = [
        ...     {
        ...         "original_image": np.array(...),
        ...         "prediction_mask": np.array(...),
        ...         "ground_truth_mask": np.array(...),
        ...         "confidence_map": np.array(...),
        ...         "metrics": {"iou": 0.75, "f1": 0.8, "dice": 0.85},
        ...     }
        ... ]
        >>> visualizer = InteractivePlotlyVisualizer()
        >>> fig = visualizer.create_interactive_prediction_grid(results)
    """
```

#### create_3d_confidence_map

```python
def create_3d_confidence_map(
    self,
    result: dict[str, Any],
    save_path: Path | None = None,
) -> PlotlyFigure:
    """Create 3D interactive confidence map.

    Args:
        result: Prediction result with confidence data.
        save_path: Optional path to save the visualization.

    Returns:
        Interactive 3D Plotly figure with confidence map.

    Example:
        >>> result = {
        ...     "confidence_map": np.random.random((256, 256)),
        ...     "original_image": np.array(...),
        ... }
        >>> visualizer = InteractivePlotlyVisualizer()
        >>> fig = visualizer.create_3d_confidence_map(result)
    """
```

#### create_dynamic_error_analysis

```python
def create_dynamic_error_analysis(
    self,
    result: dict[str, Any],
    save_path: Path | None = None,
) -> PlotlyFigure:
    """Create dynamic error analysis visualization.

    Args:
        result: Prediction result with error analysis data.
        save_path: Optional path to save the visualization.

    Returns:
        Interactive Plotly figure with error analysis.

    Example:
        >>> result = {
        ...     "prediction_mask": np.array(...),
        ...     "ground_truth_mask": np.array(...),
        ...     "original_image": np.array(...),
        ... }
        >>> visualizer = InteractivePlotlyVisualizer()
        >>> fig = visualizer.create_dynamic_error_analysis(result)
    """
```

#### create_real_time_training_dashboard

```python
def create_real_time_training_dashboard(
    self,
    training_data: dict[str, Any],
    save_path: Path | None = None,
) -> PlotlyFigure:
    """Create real-time training dashboard.

    Args:
        training_data: Training data with real-time metrics.
        save_path: Optional path to save the visualization.

    Returns:
        Interactive Plotly figure with real-time dashboard.

    Example:
        >>> training_data = {
        ...     "metrics": [
        ...         {"loss": 2.0, "val_loss": 1.8, "learning_rate": 1e-3},
        ...         {"loss": 1.5, "val_loss": 1.4, "learning_rate": 9e-4},
        ...     ]
        ... }
        >>> visualizer = InteractivePlotlyVisualizer()
        >>> fig = visualizer.create_real_time_training_dashboard(training_data)
    """
```

## Advanced Prediction Visualizer

The `AdvancedPredictionVisualizer` provides comprehensive prediction analysis and comparison capabilities.

### Class: AdvancedPredictionVisualizer

```python
class AdvancedPredictionVisualizer:
    """Advanced prediction visualization with comprehensive analysis."""

    def __init__(
        self,
        style_config: dict[str, Any] | None = None,
        template: PredictionVisualizationTemplate | None = None,
    ) -> None:
        """Initialize the advanced prediction visualizer.

        Args:
            style_config: Optional style configuration dictionary.
            template: Optional visualization template.
        """
```

### Methods

#### create_comparison_grid

```python
def create_comparison_grid(
    self,
    results: list[dict[str, Any]],
    save_path: str | Path | None = None,
    max_images: int = 9,
    show_metrics: bool = True,
    show_confidence: bool = True,
    grid_layout: tuple[int, int] | None = None,
) -> Figure | PlotlyFigure:
    """Create comparison grid for prediction results.

    Args:
        results: List of prediction result dictionaries.
        save_path: Optional path to save the visualization.
        max_images: Maximum number of images to display.
        show_metrics: Whether to show metrics overlay.
        show_confidence: Whether to show confidence maps.
        grid_layout: Optional grid layout (rows, cols).

    Returns:
        Figure or Plotly figure with comparison grid.
    """
```

#### create_confidence_map

```python
def create_confidence_map(
    self,
    result: dict[str, Any],
    save_path: str | Path | None = None,
    show_original: bool = True,
    show_contours: bool = True,
) -> Figure | PlotlyFigure:
    """Create confidence map visualization.

    Args:
        result: Prediction result with confidence data.
        save_path: Optional path to save the visualization.
        show_original: Whether to show original image.
        show_contours: Whether to show confidence contours.

    Returns:
        Figure or Plotly figure with confidence map.
    """
```

#### create_error_analysis

```python
def create_error_analysis(
    self,
    result: dict[str, Any],
    save_path: str | Path | None = None,
) -> Figure | PlotlyFigure:
    """Create error analysis visualization.

    Args:
        result: Prediction result with error analysis data.
        save_path: Optional path to save the visualization.

    Returns:
        Figure or Plotly figure with error analysis.
    """
```

#### create_segmentation_overlay

```python
def create_segmentation_overlay(
    self,
    result: dict[str, Any],
    save_path: str | Path | None = None,
    show_confidence: bool = True,
) -> Figure | PlotlyFigure:
    """Create segmentation overlay visualization.

    Args:
        result: Prediction result with segmentation data.
        save_path: Optional path to save the visualization.
        show_confidence: Whether to show confidence overlay.

    Returns:
        Figure or Plotly figure with segmentation overlay.
    """
```

#### create_tabular_comparison

```python
def create_tabular_comparison(
    self,
    results: list[dict[str, Any]],
    save_path: str | Path | None = None,
) -> Figure | PlotlyFigure:
    """Create tabular comparison of prediction results.

    Args:
        results: List of prediction result dictionaries.
        save_path: Optional path to save the visualization.

    Returns:
        Figure or Plotly figure with tabular comparison.
    """
```

## Advanced Training Visualizer

The `AdvancedTrainingVisualizer` provides comprehensive training analysis and monitoring capabilities.

### Class: AdvancedTrainingVisualizer

```python
class AdvancedTrainingVisualizer:
    """Advanced training visualization with comprehensive analysis."""

    def __init__(
        self,
        style_config: dict[str, Any] | None = None,
        template: TrainingVisualizationTemplate | None = None,
    ) -> None:
        """Initialize the advanced training visualizer.

        Args:
            style_config: Optional style configuration dictionary.
            template: Optional visualization template.
        """
```

### Methods

#### create_training_curves

```python
def create_training_curves(
    self,
    training_data: dict[str, Any],
    save_path: str | Path | None = None,
) -> Figure | PlotlyFigure:
    """Create training curves visualization.

    Args:
        training_data: Training data with metrics.
        save_path: Optional path to save the visualization.

    Returns:
        Figure or Plotly figure with training curves.
    """
```

#### create_learning_rate_analysis

```python
def create_learning_rate_analysis(
    self,
    training_data: dict[str, Any],
    save_path: str | Path | None = None,
) -> Figure | PlotlyFigure:
    """Create learning rate analysis visualization.

    Args:
        training_data: Training data with learning rate information.
        save_path: Optional path to save the visualization.

    Returns:
        Figure or Plotly figure with learning rate analysis.
    """
```

#### create_gradient_flow_analysis

```python
def create_gradient_flow_analysis(
    self,
    training_data: dict[str, Any],
    save_path: str | Path | None = None,
) -> Figure | PlotlyFigure:
    """Create gradient flow analysis visualization.

    Args:
        training_data: Training data with gradient information.
        save_path: Optional path to save the visualization.

    Returns:
        Figure or Plotly figure with gradient flow analysis.
    """
```

#### create_parameter_distribution

```python
def create_parameter_distribution(
    self,
    training_data: dict[str, Any],
    save_path: str | Path | None = None,
) -> Figure | PlotlyFigure:
    """Create parameter distribution visualization.

    Args:
        training_data: Training data with parameter information.
        save_path: Optional path to save the visualization.

    Returns:
        Figure or Plotly figure with parameter distribution.
    """
```

## Experiment Visualizer

The `ExperimentVisualizer` provides experiment comparison and analysis capabilities.

### Class: ExperimentVisualizer

```python
class ExperimentVisualizer:
    """Experiment visualization with comparison and analysis."""

    def __init__(
        self,
        style_config: dict[str, Any] | None = None,
        template: ExperimentVisualizationTemplate | None = None,
    ) -> None:
        """Initialize the experiment visualizer.

        Args:
            style_config: Optional style configuration dictionary.
            template: Optional visualization template.
        """
```

### Methods

#### create_experiment_summary

```python
def create_experiment_summary(
    self,
    experiment_data: dict[str, Any],
    save_path: str | Path | None = None,
) -> Figure | PlotlyFigure:
    """Create experiment summary visualization.

    Args:
        experiment_data: Experiment data with results.
        save_path: Optional path to save the visualization.

    Returns:
        Figure or Plotly figure with experiment summary.
    """
```

#### create_metric_comparison

```python
def create_metric_comparison(
    self,
    experiments_data: list[dict[str, Any]],
    save_path: str | Path | None = None,
) -> Figure | PlotlyFigure:
    """Create metric comparison across experiments.

    Args:
        experiments_data: List of experiment data dictionaries.
        save_path: Optional path to save the visualization.

    Returns:
        Figure or Plotly figure with metric comparison.
    """
```

#### create_configuration_analysis

```python
def create_configuration_analysis(
    self,
    experiment_data: dict[str, Any],
    save_path: str | Path | None = None,
) -> Figure | PlotlyFigure:
    """Create configuration analysis visualization.

    Args:
        experiment_data: Experiment data with configuration.
        save_path: Optional path to save the visualization.

    Returns:
        Figure or Plotly figure with configuration analysis.
    """
```

## Export Handlers

The `ExportHandler` provides multi-format export capabilities with metadata preservation.

### Class: ExportHandler

```python
class ExportHandler:
    """Handle multi-format plot export with metadata preservation."""

    def __init__(self, export_formats: list[str]) -> None:
        """Initialize export handler.

        Args:
            export_formats: List of supported export formats.
        """
```

### Methods

#### save_plot

```python
def save_plot(
    self,
    fig: go.Figure | Figure,
    save_path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save plot in multiple formats with metadata.

    Args:
        fig: Plotly figure or matplotlib figure.
        save_path: Base path for saving.
        metadata: Optional metadata to include.
    """
```

## Metadata Handlers

The `MetadataHandler` provides metadata creation, embedding, and management capabilities.

### Class: MetadataHandler

```python
class MetadataHandler:
    """Handle metadata for visualization components."""

    def __init__(self) -> None:
        """Initialize metadata handler."""
```

### Methods

#### create_metadata

```python
def create_metadata(
    self,
    plot_type: str,
    data_info: dict[str, Any] | None = None,
    custom_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create metadata for a plot.

    Args:
        plot_type: Type of plot (e.g., 'training_curves', 'prediction_grid').
        data_info: Information about the data used in the plot.
        custom_metadata: Additional custom metadata.

    Returns:
        Metadata dictionary.
    """
```

#### embed_metadata_in_figure

```python
def embed_metadata_in_figure(
    self,
    fig: go.Figure,
    metadata: dict[str, Any],
) -> go.Figure:
    """Embed metadata in Plotly figure.

    Args:
        fig: Plotly figure to embed metadata in.
        metadata: Metadata to embed.

    Returns:
        Figure with embedded metadata.
    """
```

#### extract_metadata_from_figure

```python
def extract_metadata_from_figure(self, fig: go.Figure) -> dict[str, Any]:
    """Extract metadata from Plotly figure.

    Args:
        fig: Plotly figure to extract metadata from.

    Returns:
        Extracted metadata dictionary.
    """
```

#### save_metadata

```python
def save_metadata(
    self,
    metadata: dict[str, Any],
    save_path: Path,
) -> None:
    """Save metadata to file.

    Args:
        metadata: Metadata to save.
        save_path: Path to save metadata file.
    """
```

#### load_metadata

```python
def load_metadata(self, metadata_path: Path) -> dict[str, Any]:
    """Load metadata from file.

    Args:
        metadata_path: Path to metadata file.

    Returns:
        Loaded metadata dictionary.
    """
```

#### validate_metadata

```python
def validate_metadata(self, metadata: dict[str, Any]) -> bool:
    """Validate metadata structure.

    Args:
        metadata: Metadata to validate.

    Returns:
        True if metadata is valid, False otherwise.
    """
```

#### merge_metadata

```python
def merge_metadata(
    self,
    base_metadata: dict[str, Any],
    additional_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Merge metadata dictionaries.

    Args:
        base_metadata: Base metadata dictionary.
        additional_metadata: Additional metadata to merge.

    Returns:
        Merged metadata dictionary.
    """
```

## Usage Examples

### Basic Training Visualization

```python
from crackseg.evaluation.visualization import InteractivePlotlyVisualizer


# Initialize visualizer
visualizer = InteractivePlotlyVisualizer(
    export_formats=["html", "png", "pdf"]
)

# Create training data
training_data = {
    "metrics": [
        {"loss": 2.0, "val_loss": 1.8, "iou": 0.3, "val_iou": 0.25},
        {"loss": 1.5, "val_loss": 1.4, "iou": 0.5, "val_iou": 0.45},
        {"loss": 1.2, "val_loss": 1.1, "iou": 0.7, "val_iou": 0.65},
    ]
}

# Create interactive training curves
fig = visualizer.create_interactive_training_curves(
    training_data=training_data,
    save_path=Path("outputs/training_curves")
)
```

### Prediction Analysis

```python
from crackseg.evaluation.visualization import AdvancedPredictionVisualizer


# Initialize visualizer
visualizer = AdvancedPredictionVisualizer()

# Create prediction results
results = [
    {
        "original_image": np.array(...),
        "prediction_mask": np.array(...),
        "ground_truth_mask": np.array(...),
        "confidence_map": np.array(...),
        "metrics": {"iou": 0.75, "f1": 0.8, "dice": 0.85},
    }
]

# Create comparison grid
fig = visualizer.create_comparison_grid(
    results=results,
    save_path=Path("outputs/prediction_comparison")
)
```

### Real-time Dashboard

```python
from crackseg.evaluation.visualization import InteractivePlotlyVisualizer


# Initialize visualizer with real-time capabilities
visualizer = InteractivePlotlyVisualizer(
    export_formats=["html", "png", "json"]
)

# Create real-time training dashboard
fig = visualizer.create_real_time_training_dashboard(
    training_data=training_data,
    save_path=Path("outputs/real_time_dashboard")
)
```

### Custom Template Integration

```python
from crackseg.evaluation.visualization import InteractivePlotlyVisualizer

from crackseg.evaluation.visualization.templates import TrainingVisualizationTemplate


# Create custom template
template = TrainingVisualizationTemplate({
    "figure_size": (12, 8),
    "color_palette": "viridis",
    "line_width": 3,
    "font_size": 14,
    "dpi": 100,
    "grid_alpha": 0.3,
    "title_font_size": 16,
    "legend_font_size": 12,
})

# Initialize visualizer with template
visualizer = InteractivePlotlyVisualizer(
    template=template,
    export_formats=["html", "png", "pdf", "svg", "jpg", "json"]
)

# Create visualization with custom styling
fig = visualizer.create_interactive_training_curves(
    training_data=training_data,
    save_path=Path("outputs/custom_styled_training")
)
```

## Configuration Guide

### Export Formats

Supported export formats:

- **html**: Interactive HTML with embedded metadata
- **png**: High-resolution PNG images
- **pdf**: Vector PDF documents
- **svg**: Scalable vector graphics
- **jpg**: JPEG images
- **json**: JSON data with metadata

### Template Configuration

```python
# Training template configuration
training_config = {
    "figure_size": (12, 8),
    "color_palette": "viridis",
    "line_width": 3,
    "font_size": 14,
    "dpi": 100,
    "grid_alpha": 0.3,
    "title_font_size": 16,
    "legend_font_size": 12,
}

# Prediction template configuration
prediction_config = {
    "figure_size": (10, 8),
    "color_palette": "plasma",
    "line_width": 2,
    "font_size": 12,
    "dpi": 100,
    "grid_alpha": 0.2,
    "title_font_size": 14,
    "legend_font_size": 10,
}
```

### Data Format Requirements

#### Training Data Format

```python
training_data = {
    "metrics": [
        {
            "loss": float,           # Training loss
            "val_loss": float,       # Validation loss
            "iou": float,            # Training IoU
            "val_iou": float,        # Validation IoU
            "learning_rate": float,  # Learning rate (optional)
            "gradient_norm": float,  # Gradient norm (optional)
        },
        # ... more epochs
    ]
}
```

#### Prediction Data Format

```python
prediction_results = [
    {
        "original_image": np.ndarray,      # Original image (H, W, C)
        "prediction_mask": np.ndarray,     # Prediction mask (H, W)
        "ground_truth_mask": np.ndarray,   # Ground truth mask (H, W)
        "confidence_map": np.ndarray,      # Confidence map (H, W)
        "metrics": {
            "iou": float,                  # IoU score
            "precision": float,            # Precision score
            "recall": float,               # Recall score
            "f1": float,                   # F1 score
            "dice": float,                 # Dice coefficient
        },
    },
    # ... more predictions
]
```

### Error Handling

The visualization system includes comprehensive error handling:

- **Invalid data**: Graceful handling with empty figure creation
- **Missing metrics**: Automatic filtering of unavailable metrics
- **File system errors**: Proper error logging and recovery
- **Memory issues**: Efficient memory management for large datasets

### Performance Considerations

- **Large datasets**: Optimized for datasets with 1000+ epochs
- **High-resolution images**: Efficient handling of 1024x1024+ images
- **Concurrent operations**: Thread-safe visualization creation
- **Memory management**: Automatic cleanup of temporary resources

### Best Practices

1. **Use appropriate export formats** for your use case
2. **Include metadata** for reproducibility
3. **Validate data** before visualization
4. **Use templates** for consistent styling
5. **Handle errors gracefully** in production code
6. **Monitor memory usage** with large datasets
7. **Test visualizations** with sample data first
