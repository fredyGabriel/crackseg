# Visualization Customization Guide

Comprehensive guide for customizing the CrackSeg visualization system.

## Table of Contents

1. [Overview](#overview)
2. [Template System](#template-system)
3. [Color Schemes](#color-schemes)
4. [Layout Customization](#layout-customization)
5. [Interactive Features](#interactive-features)
6. [Export Customization](#export-customization)
7. [Performance Optimization](#performance-optimization)
8. [Advanced Customization](#advanced-customization)
9. [Best Practices](#best-practices)

## Overview

The CrackSeg visualization system is designed to be highly customizable while maintaining
consistency and professional appearance. This guide covers all aspects of customization from basic
styling to advanced features.

## Template System

### Creating Custom Templates

```python
from crackseg.evaluation.visualization.templates import BaseVisualizationTemplate


class CustomTrainingTemplate(BaseVisualizationTemplate):
    """Custom template for training visualizations."""

    def __init__(self) -> None:
        """Initialize custom template."""
        super().__init__({
            "figure_size": (14, 10),
            "color_palette": "viridis",
            "line_width": 3,
            "font_size": 14,
            "dpi": 150,
            "grid_alpha": 0.2,
            "title_font_size": 18,
            "legend_font_size": 12,
            "background_color": "#f8f9fa",
            "grid_color": "#e9ecef",
            "text_color": "#212529",
        })

    def apply_training_curves_style(self, fig) -> None:
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
        )

        # Update traces
        for trace in fig.data:
            trace.line.width = self.config["line_width"]
            trace.marker.size = 8
```

### Using Custom Templates

```python
from crackseg.evaluation.visualization import InteractivePlotlyVisualizer


# Create custom template
custom_template = CustomTrainingTemplate()

# Initialize visualizer with custom template
visualizer = InteractivePlotlyVisualizer(
    template=custom_template,
    export_formats=["html", "png", "pdf"]
)

# Create visualization with custom styling
fig = visualizer.create_interactive_training_curves(
    training_data=training_data,
    save_path=Path("outputs/custom_training")
)
```

### Template Inheritance

```python
class AdvancedTrainingTemplate(CustomTrainingTemplate):
    """Advanced training template with additional features."""

    def __init__(self) -> None:
        """Initialize advanced template."""
        super().__init__()
        self.config.update({
            "show_confidence_intervals": True,
            "show_learning_rate": True,
            "show_gradient_norm": True,
            "subplot_layout": (2, 2),
        })

    def apply_advanced_training_style(self, fig) -> None:
        """Apply advanced training style."""
        super().apply_training_curves_style(fig)

        # Add confidence intervals
        if self.config["show_confidence_intervals"]:
            self._add_confidence_intervals(fig)

        # Add learning rate subplot
        if self.config["show_learning_rate"]:
            self._add_learning_rate_subplot(fig)
```

## Color Schemes

### Built-in Color Palettes

```python
# Available color palettes
color_palettes = {
    "viridis": "Sequential color palette",
    "plasma": "Sequential color palette",
    "inferno": "Sequential color palette",
    "magma": "Sequential color palette",
    "cividis": "Sequential color palette",
    "coolwarm": "Diverging color palette",
    "RdBu": "Red-Blue diverging palette",
    "PiYG": "Pink-Yellow-Green diverging palette",
    "BrBG": "Brown-Blue-Green diverging palette",
    "Spectral": "Spectral diverging palette",
    "Set1": "Qualitative color palette",
    "Set2": "Qualitative color palette",
    "Set3": "Qualitative color palette",
    "Paired": "Qualitative color palette",
    "Accent": "Qualitative color palette",
    "Dark2": "Qualitative color palette",
    "Pastel1": "Qualitative color palette",
    "Pastel2": "Qualitative color palette",
}
```

### Custom Color Schemes

```python
# Define custom color scheme
custom_colors = {
    "primary": "#1f77b4",      # Blue
    "secondary": "#ff7f0e",     # Orange
    "success": "#2ca02c",       # Green
    "warning": "#d62728",       # Red
    "info": "#9467bd",          # Purple
    "light": "#8c564b",         # Brown
    "dark": "#e377c2",          # Pink
    "muted": "#7f7f7f",         # Gray
    "background": "#f8f9fa",    # Light gray
    "grid": "#e9ecef",          # Lighter gray
    "text": "#212529",          # Dark gray
}

# Create custom color template
class CustomColorTemplate(BaseVisualizationTemplate):
    """Template with custom color scheme."""

    def __init__(self) -> None:
        """Initialize with custom colors."""
        super().__init__({
            "color_palette": "custom",
            "custom_colors": custom_colors,
            "figure_size": (12, 8),
            "line_width": 2,
            "font_size": 12,
        })

    def get_color(self, index: int) -> str:
        """Get color from custom palette."""
        colors = list(self.config["custom_colors"].values())
        return colors[index % len(colors)]
```

### Color Mapping for Different Data Types

```python
# Color mapping for different metrics
metric_colors = {
    "loss": "#d62728",          # Red for loss
    "val_loss": "#ff7f0e",      # Orange for validation loss
    "iou": "#2ca02c",           # Green for IoU
    "val_iou": "#1f77b4",       # Blue for validation IoU
    "precision": "#9467bd",      # Purple for precision
    "recall": "#8c564b",        # Brown for recall
    "f1": "#e377c2",            # Pink for F1
    "dice": "#7f7f7f",          # Gray for Dice
}

# Color mapping for prediction results
prediction_colors = {
    "true_positive": "#2ca02c",  # Green
    "false_positive": "#d62728", # Red
    "false_negative": "#ff7f0e", # Orange
    "true_negative": "#1f77b4",  # Blue
}
```

## Layout Customization

### Figure Size and Aspect Ratio

```python
# Different figure sizes for different use cases
figure_sizes = {
    "presentation": (16, 12),    # Large for presentations
    "paper": (12, 8),            # Standard for papers
    "poster": (20, 16),          # Large for posters
    "web": (10, 6),              # Medium for web
    "mobile": (8, 5),            # Small for mobile
    "square": (10, 10),          # Square aspect ratio
    "wide": (16, 8),             # Wide aspect ratio
    "tall": (8, 16),             # Tall aspect ratio
}

# Custom template with responsive sizing
class ResponsiveTemplate(BaseVisualizationTemplate):
    """Template with responsive figure sizing."""

    def __init__(self, size_type: str = "paper") -> None:
        """Initialize with responsive sizing."""
        super().__init__({
            "figure_size": figure_sizes[size_type],
            "responsive": True,
            "autosize": True,
        })
```

### Subplot Layouts

```python
# Common subplot layouts
subplot_layouts = {
    "training_curves": (2, 2),      # 2x2 for training metrics
    "prediction_comparison": (3, 3), # 3x3 for prediction grid
    "error_analysis": (2, 3),        # 2x3 for error analysis
    "confidence_maps": (2, 2),       # 2x2 for confidence maps
    "experiment_comparison": (3, 2), # 3x2 for experiment comparison
}

# Custom subplot template
class SubplotTemplate(BaseVisualizationTemplate):
    """Template with custom subplot layouts."""

    def __init__(self, layout_type: str = "training_curves") -> None:
        """Initialize with custom subplot layout."""
        super().__init__({
            "subplot_layout": subplot_layouts[layout_type],
            "subplot_titles": self._get_subplot_titles(layout_type),
            "subplot_spacing": 0.1,
        })

    def _get_subplot_titles(self, layout_type: str) -> list[str]:
        """Get subplot titles for layout type."""
        titles = {
            "training_curves": ["Training Loss", "Validation Loss", "Training IoU", "Validation IoU"],
            "prediction_comparison": ["Original", "Prediction", "Ground Truth", "Confidence", "Error", "Overlay"],
            "error_analysis": ["False Positives", "False Negatives", "True Positives", "True Negatives", "Error Map", "Confidence"],
            "confidence_maps": ["Original", "Confidence", "Threshold", "Segmentation"],
        }
        return titles.get(layout_type, [])
```

### Font and Text Customization

```python
# Font configurations
font_configs = {
    "scientific": {
        "family": "serif",
        "size": 12,
        "color": "#000000",
    },
    "presentation": {
        "family": "sans-serif",
        "size": 14,
        "color": "#333333",
    },
    "web": {
        "family": "Arial, sans-serif",
        "size": 10,
        "color": "#212529",
    },
    "print": {
        "family": "Times New Roman, serif",
        "size": 11,
        "color": "#000000",
    },
}

# Custom font template
class FontTemplate(BaseVisualizationTemplate):
    """Template with custom font configuration."""

    def __init__(self, font_type: str = "scientific") -> None:
        """Initialize with custom font configuration."""
        super().__init__({
            "font_config": font_configs[font_type],
            "title_font_size": font_configs[font_type]["size"] + 4,
            "legend_font_size": font_configs[font_type]["size"] - 2,
            "axis_font_size": font_configs[font_type]["size"] - 1,
        })
```

## Interactive Features

### Custom Hover Information

```python
# Custom hover template
class CustomHoverTemplate(BaseVisualizationTemplate):
    """Template with custom hover information."""

    def __init__(self) -> None:
        """Initialize with custom hover configuration."""
        super().__init__({
            "hover_mode": "closest",
            "hover_info": "all",
            "custom_hover_template": "<b>%{fullData.name}</b><br>" +
                                   "Epoch: %{x}<br>" +
                                   "Value: %{y:.4f}<br>" +
                                   "<extra></extra>",
        })

    def apply_hover_style(self, fig) -> None:
        """Apply custom hover style."""
        for trace in fig.data:
            trace.hovertemplate = self.config["custom_hover_template"]
            trace.hoverinfo = self.config["hover_info"]
```

### Zoom and Pan Configuration

```python
# Zoom and pan configurations
zoom_configs = {
    "training_curves": {
        "dragmode": "zoom",
        "selectdirection": "any",
        "zoom_direction": "both",
    },
    "prediction_grid": {
        "dragmode": "pan",
        "selectdirection": "any",
        "zoom_direction": "both",
    },
    "confidence_map": {
        "dragmode": "zoom",
        "selectdirection": "any",
        "zoom_direction": "both",
    },
}

# Custom zoom template
class ZoomTemplate(BaseVisualizationTemplate):
    """Template with custom zoom and pan configuration."""

    def __init__(self, zoom_type: str = "training_curves") -> None:
        """Initialize with custom zoom configuration."""
        super().__init__({
            "zoom_config": zoom_configs[zoom_type],
            "show_zoom_buttons": True,
            "show_pan_buttons": True,
            "show_reset_buttons": True,
        })

    def apply_zoom_style(self, fig) -> None:
        """Apply custom zoom style."""
        fig.update_layout(
            dragmode=self.config["zoom_config"]["dragmode"],
            selectdirection=self.config["zoom_config"]["selectdirection"],
        )
```

### Animation Configuration

```python
# Animation configurations
animation_configs = {
    "training_progress": {
        "frame_duration": 1000,
        "transition_duration": 500,
        "redraw": True,
    },
    "prediction_sequence": {
        "frame_duration": 2000,
        "transition_duration": 1000,
        "redraw": True,
    },
    "confidence_evolution": {
        "frame_duration": 1500,
        "transition_duration": 750,
        "redraw": True,
    },
}

# Custom animation template
class AnimationTemplate(BaseVisualizationTemplate):
    """Template with custom animation configuration."""

    def __init__(self, animation_type: str = "training_progress") -> None:
        """Initialize with custom animation configuration."""
        super().__init__({
            "animation_config": animation_configs[animation_type],
            "show_animation_controls": True,
            "auto_play": False,
        })
```

## Export Customization

### Multi-format Export Configuration

```python
# Export format configurations
export_configs = {
    "presentation": {
        "formats": ["png", "pdf"],
        "dpi": 300,
        "transparent": False,
        "bbox_inches": "tight",
    },
    "web": {
        "formats": ["html", "png"],
        "dpi": 150,
        "transparent": True,
        "bbox_inches": "tight",
    },
    "publication": {
        "formats": ["pdf", "svg"],
        "dpi": 600,
        "transparent": False,
        "bbox_inches": "tight",
    },
    "mobile": {
        "formats": ["png", "jpg"],
        "dpi": 100,
        "transparent": False,
        "bbox_inches": "tight",
    },
}

# Custom export template
class ExportTemplate(BaseVisualizationTemplate):
    """Template with custom export configuration."""

    def __init__(self, export_type: str = "presentation") -> None:
        """Initialize with custom export configuration."""
        super().__init__({
            "export_config": export_configs[export_type],
            "include_metadata": True,
            "embed_metadata": True,
        })
```

### Metadata Customization

```python
# Custom metadata template
class MetadataTemplate(BaseVisualizationTemplate):
    """Template with custom metadata configuration."""

    def __init__(self) -> None:
        """Initialize with custom metadata configuration."""
        super().__init__({
            "metadata_fields": [
                "created_at",
                "plot_type",
                "data_info",
                "custom_metadata",
                "export_timestamp",
                "export_formats",
                "responsive",
                "figure_type",
            ],
            "custom_metadata": {
                "project": "CrackSeg",
                "version": "1.0.0",
                "author": "CrackSeg Team",
                "license": "MIT",
            },
        })

    def create_custom_metadata(self, plot_type: str, data_info: dict) -> dict:
        """Create custom metadata."""
        metadata = super().create_metadata(plot_type, data_info)
        metadata.update(self.config["custom_metadata"])
        return metadata
```

## Performance Optimization

### Large Dataset Handling

```python
# Performance optimization configurations
performance_configs = {
    "large_dataset": {
        "downsample_threshold": 1000,
        "downsample_factor": 10,
        "memory_limit": "2GB",
        "chunk_size": 100,
    },
    "real_time": {
        "update_interval": 1000,
        "max_points": 100,
        "memory_limit": "1GB",
        "chunk_size": 50,
    },
    "high_resolution": {
        "max_image_size": (2048, 2048),
        "compression": "lzw",
        "memory_limit": "4GB",
        "chunk_size": 25,
    },
}

# Performance optimization template
class PerformanceTemplate(BaseVisualizationTemplate):
    """Template with performance optimization."""

    def __init__(self, performance_type: str = "large_dataset") -> None:
        """Initialize with performance optimization."""
        super().__init__({
            "performance_config": performance_configs[performance_type],
            "lazy_loading": True,
            "progressive_rendering": True,
        })

    def optimize_for_large_dataset(self, data: dict) -> dict:
        """Optimize data for large datasets."""
        config = self.config["performance_config"]

        if len(data["metrics"]) > config["downsample_threshold"]:
            # Downsample data
            step = config["downsample_factor"]
            data["metrics"] = data["metrics"][::step]

        return data
```

### Memory Management

```python
# Memory management template
class MemoryTemplate(BaseVisualizationTemplate):
    """Template with memory management."""

    def __init__(self) -> None:
        """Initialize with memory management."""
        super().__init__({
            "memory_limit": "2GB",
            "cleanup_interval": 1000,
            "garbage_collection": True,
        })

    def cleanup_memory(self) -> None:
        """Clean up memory."""
        import gc
        gc.collect()

    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits."""
        import psutil
        memory_usage = psutil.virtual_memory().percent
        return memory_usage < 80  # 80% threshold
```

## Advanced Customization

### Custom Plot Types

```python
# Custom plot type template
class CustomPlotTemplate(BaseVisualizationTemplate):
    """Template for custom plot types."""

    def __init__(self) -> None:
        """Initialize custom plot template."""
        super().__init__({
            "custom_plot_types": [
                "gradient_flow",
                "parameter_distribution",
                "activation_maps",
                "attention_weights",
                "feature_importance",
            ],
        })

    def create_gradient_flow_plot(self, data: dict) -> "PlotlyFigure":
        """Create gradient flow plot."""
        # Custom implementation for gradient flow visualization
        pass

    def create_parameter_distribution_plot(self, data: dict) -> "PlotlyFigure":
        """Create parameter distribution plot."""
        # Custom implementation for parameter distribution
        pass
```

### Custom Metrics Visualization

```python
# Custom metrics template
class CustomMetricsTemplate(BaseVisualizationTemplate):
    """Template for custom metrics visualization."""

    def __init__(self) -> None:
        """Initialize custom metrics template."""
        super().__init__({
            "custom_metrics": [
                "gradient_norm",
                "learning_rate",
                "parameter_norm",
                "activation_norm",
                "loss_landscape",
            ],
        })

    def create_custom_metric_plot(self, metric_name: str, data: dict) -> "PlotlyFigure":
        """Create custom metric plot."""
        # Custom implementation for specific metrics
        pass
```

### Custom Export Formats

```python
# Custom export format template
class CustomExportTemplate(BaseVisualizationTemplate):
    """Template for custom export formats."""

    def __init__(self) -> None:
        """Initialize custom export template."""
        super().__init__({
            "custom_export_formats": [
                "eps",
                "tiff",
                "webp",
                "json",
                "csv",
            ],
        })

    def export_to_custom_format(self, fig, save_path: Path, format_type: str) -> None:
        """Export to custom format."""
        # Custom export implementation
        pass
```

## Best Practices

### Template Design Principles

1. **Consistency**: Use consistent styling across all visualizations
2. **Modularity**: Create reusable template components
3. **Inheritance**: Use template inheritance for specialized cases
4. **Configuration**: Make templates configurable through parameters
5. **Documentation**: Document all custom template features

### Performance Guidelines

1. **Memory Management**: Implement proper memory cleanup
2. **Lazy Loading**: Use lazy loading for large datasets
3. **Progressive Rendering**: Implement progressive rendering for real-time updates
4. **Caching**: Cache frequently used template configurations
5. **Optimization**: Optimize for specific use cases

### Customization Workflow

1. **Define Requirements**: Clearly define customization requirements
2. **Create Template**: Create custom template class
3. **Test Implementation**: Test with sample data
4. **Validate Performance**: Ensure performance meets requirements
5. **Document Usage**: Document template usage and configuration
6. **Maintain Consistency**: Ensure consistency with existing templates

### Error Handling

```python
# Error handling template
class ErrorHandlingTemplate(BaseVisualizationTemplate):
    """Template with comprehensive error handling."""

    def __init__(self) -> None:
        """Initialize error handling template."""
        super().__init__({
            "error_handling": {
                "invalid_data": "create_empty_figure",
                "missing_metrics": "filter_available_metrics",
                "file_system_error": "log_error_and_continue",
                "memory_error": "cleanup_and_retry",
            },
        })

    def handle_invalid_data(self, data: dict) -> dict:
        """Handle invalid data gracefully."""
        # Implementation for handling invalid data
        pass

    def handle_missing_metrics(self, data: dict) -> dict:
        """Handle missing metrics gracefully."""
        # Implementation for handling missing metrics
        pass
```

### Testing Custom Templates

```python
# Test custom template
def test_custom_template():
    """Test custom template functionality."""
    template = CustomTrainingTemplate()
    visualizer = InteractivePlotlyVisualizer(template=template)

    # Test with sample data
    sample_data = {
        "metrics": [
            {"loss": 2.0, "val_loss": 1.8, "iou": 0.3, "val_iou": 0.25},
            {"loss": 1.5, "val_loss": 1.4, "iou": 0.5, "val_iou": 0.45},
        ]
    }

    fig = visualizer.create_interactive_training_curves(sample_data)
    assert fig is not None
    assert hasattr(fig, "layout")

    # Test template-specific features
    assert fig.layout.plot_bgcolor == template.config["background_color"]
    assert fig.layout.font.size == template.config["font_size"]
```

This comprehensive customization guide provides all the tools and examples needed to create highly
customized visualizations while maintaining consistency and performance.
