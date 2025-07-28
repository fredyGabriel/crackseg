# Visualization System Architecture

## Overview

The CrackSeg visualization system provides comprehensive training and prediction visualization
capabilities with professional-quality outputs. The system is designed with modularity,
extensibility, and consistency in mind.

## Core Architecture

### 1. Component-Based Design

The visualization system follows a component-based architecture where each visualizer is a
self-contained module with clear interfaces:

```bash
visualization/
├── __init__.py                 # Main package interface
├── advanced_training_viz.py    # Main orchestrator
├── training_curves.py          # Training curves component
├── learning_rate_analysis.py   # Learning rate analysis component
├── parameter_analysis.py       # Parameter distribution component
├── prediction_viz.py          # Prediction visualization (planned)
├── templates/                  # Template system (planned)
│   ├── base_template.py
│   ├── training_template.py
│   └── prediction_template.py
└── utils/                      # Shared utilities
    ├── style_manager.py
    ├── export_manager.py
    └── data_processor.py
```

### 2. Interface Standards

All visualizer components implement a consistent interface:

```python
class BaseVisualizer:
    def __init__(self, style_config: dict[str, Any]):
        """Initialize with style configuration."""

    def create_visualization(self, data: dict[str, Any], **kwargs) -> Figure | PlotlyFigure:
        """Create visualization from data."""

    def _create_static_visualization(self, data: dict[str, Any], **kwargs) -> Figure:
        """Create static matplotlib visualization."""

    def _create_interactive_visualization(self, data: dict[str, Any], **kwargs) -> PlotlyFigure:
        """Create interactive plotly visualization."""

    def _create_empty_plot(self, title: str) -> Figure:
        """Create empty plot with informative message."""
```

### 3. Data Flow Architecture

```bash
Input Data Sources:
├── Training Metrics (JSONL)
├── Model Checkpoints (PyTorch)
├── Configuration Files (YAML)
└── Experiment Metadata (JSON)

Data Processing:
├── Data Validation
├── Type Conversion
├── Statistical Analysis
└── Format Standardization

Visualization Pipeline:
├── Style Configuration
├── Data Processing
├── Plot Generation
├── Quality Enhancement
└── Export/Storage
```

## Template System Design

### 1. Base Template Interface

```python
class VisualizationTemplate:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.style_manager = StyleManager(config)

    def apply_template(self, fig: Figure | PlotlyFigure) -> Figure | PlotlyFigure:
        """Apply template styling to figure."""

    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration for template."""
```

### 2. Template Categories

#### Training Templates

- **Training Curves Template**: Multi-metric plotting with consistent styling
- **Learning Rate Template**: LR schedule analysis with zoom capabilities
- **Gradient Flow Template**: Gradient norm visualization with layer breakdown
- **Parameter Distribution Template**: Statistical plots with outlier detection

#### Prediction Templates

- **Comparison Grid Template**: Image-prediction-ground truth grids
- **Confidence Map Template**: Heatmap visualization with threshold controls
- **Error Analysis Template**: Error distribution and pattern analysis

### 3. Style Configuration Schema

```yaml
visualization:
  templates:
    training:
      figure_size: [12, 8]
      dpi: 300
      color_palette: "viridis"
      grid_alpha: 0.3
      line_width: 2
      font_size: 12
      title_font_size: 14
      legend_font_size: 10

    prediction:
      figure_size: [16, 12]
      dpi: 300
      color_palette: "plasma"
      grid_alpha: 0.2
      line_width: 1.5
      font_size: 11
      title_font_size: 16
      legend_font_size: 12
```

## Component Interfaces

### 1. AdvancedTrainingVisualizer

**Purpose**: Main orchestrator for training visualizations
**Responsibilities**:

- Coordinate component visualizers
- Manage artifact integration
- Handle data loading and validation
- Provide unified interface

**Key Methods**:

```python
def create_training_curves(self, training_data: dict, **kwargs) -> Figure | PlotlyFigure
def analyze_learning_rate_schedule(self, training_data: dict, **kwargs) -> Figure | PlotlyFigure
def visualize_parameter_distributions(self, model_path: Path, **kwargs) -> Figure | PlotlyFigure
def create_comprehensive_report(self, experiment_dir: Path, output_dir: Path) -> dict[str, Path]
```

### 2. TrainingCurvesVisualizer

**Purpose**: Specialized component for training curve visualization
**Responsibilities**:

- Multi-metric plotting
- Auto-detection of available metrics
- Interactive and static plot generation
- Consistent styling application

### 3. LearningRateAnalyzer

**Purpose**: Learning rate schedule analysis and visualization
**Responsibilities**:

- LR schedule extraction from training data
- Schedule type detection (step, cosine, etc.)
- Visualization with zoom and pan capabilities
- Statistical analysis of LR patterns

### 4. ParameterAnalyzer

**Purpose**: Model parameter distribution analysis
**Responsibilities**:

- Parameter statistics extraction
- Distribution visualization
- Outlier detection
- Model complexity analysis

## Integration Points

### 1. ArtifactManager Integration

```python
def _save_visualization_with_artifacts(self, fig: Figure | PlotlyFigure, filename: str, description: str) -> tuple[str, Any] | None:
    """Save visualization using ArtifactManager if available."""
```

### 2. ExperimentManager Integration

```python
def load_training_data(self, experiment_dir: Path, include_gradients: bool = False) -> dict[str, Any]:
    """Load training data from experiment directory."""
```

### 3. Configuration System Integration

```python
def _get_default_style(self) -> dict[str, Any]:
    """Get default styling configuration."""
```

## Quality Assurance

### 1. Testing Strategy

- **Unit Tests**: Each component has comprehensive unit tests
- **Integration Tests**: End-to-end visualization pipeline testing
- **Performance Tests**: Large dataset handling and memory usage
- **Style Tests**: Visual consistency and quality verification

### 2. Code Quality

- **Type Safety**: Complete type annotations (Python 3.12+)
- **Documentation**: Comprehensive docstrings for all public APIs
- **Linting**: Ruff, Black, and basedpyright compliance
- **Coverage**: >80% test coverage for core functionality

### 3. Performance Considerations

- **Memory Efficiency**: Streaming for large datasets
- **Rendering Speed**: Optimized plotting for real-time updates
- **File Size**: Compressed exports for storage efficiency
- **Scalability**: Support for multiple concurrent visualizations

## Future Extensibility

### 1. Plugin Architecture

The system is designed to support plugin-based extensions:

- Custom visualizer components
- New template types
- Specialized export formats
- Integration with external tools

### 2. Configuration-Driven Development

All aspects are configurable:

- Visualizer behavior
- Template styling
- Export formats
- Integration settings

### 3. API Stability

The system maintains backward compatibility:

- Stable public interfaces
- Deprecation warnings for changes
- Migration guides for updates
- Version compatibility matrix
