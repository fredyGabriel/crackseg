# CrackSeg Evaluation Module

Modular evaluation system for crack segmentation models with professional visualizations and
comprehensive analysis capabilities.

## Architecture Overview

The evaluation module follows a modular design with clear separation of concerns:

```bash
src/crackseg/evaluation/
├── __init__.py              # Main API exports
├── core/                    # Core analysis components
│   ├── __init__.py
│   ├── analyzer.py          # Main prediction analyzer (250 lines)
│   ├── model_loader.py      # Model loading utilities (150 lines)
│   └── image_processor.py   # Image preprocessing (150 lines)
├── metrics/                 # Metrics computation
│   ├── __init__.py
│   ├── calculator.py        # Segmentation metrics (120 lines)
│   └── batch_processor.py   # Batch analysis (180 lines)
├── visualization/           # Visualization components
│   ├── __init__.py
│   ├── prediction_viz.py    # Prediction visualizations (200 lines)
│   └── experiment_viz.py    # Experiment visualizations (250 lines)
└── cli/                     # Command-line interfaces
    ├── __init__.py
    └── prediction_cli.py    # Unified CLI (180 lines)
```

## Key Features

### ✅ **Modular Design**

- **Single Responsibility**: Each module has a clear, focused purpose
- **Loose Coupling**: Components can be used independently
- **Easy Testing**: Small, focused modules are easier to test
- **Maintainable**: Changes in one module don't affect others

### ✅ **Professional Visualizations**

- **Multi-panel layouts**: Original image, prediction, ground truth, confidence
- **Metrics overlay**: IoU, F1, precision, recall displayed on visualizations
- **Comparison grids**: Side-by-side analysis of multiple predictions
- **High-quality output**: 300 DPI, professional styling

### ✅ **Automatic Mask Inference**

- **Smart path detection**: Automatically finds corresponding masks from unified structure
- **Multiple extensions**: Supports .png, .jpg, .jpeg, .tiff
- **Flexible configuration**: Can be enabled/disabled per analysis
- **Unified data structure**: Works with `data/unified/` directory structure

### ✅ **Comprehensive Metrics**

- **Standard metrics**: IoU, F1, precision, recall, accuracy
- **Batch processing**: Efficient processing of multiple images
- **Statistical analysis**: Average metrics across datasets

## Quick Start

### Basic Usage

```python
from crackseg.evaluation import PredictionAnalyzer, PredictionVisualizer

# Initialize analyzer
analyzer = PredictionAnalyzer(
    checkpoint_path="outputs/checkpoints/model_best.pth.tar",
    config_path="outputs/configurations/default_experiment/config_epoch_0500.yaml",
    mask_dir="data/unified/masks",  # Enable auto-inference with unified structure
)

# Analyze single image
result = analyzer.analyze_image(
    image_path="data/unified/images/98.jpg",
    threshold=0.5,
    auto_find_mask=True,
)

# Create visualization
visualizer = PredictionVisualizer(analyzer.config)
visualizer.create_visualization(result, "output.png")
```

### Command Line Interface

```bash
# Single image analysis with automatic mask inference
python -m crackseg.evaluation.cli \
    --checkpoint outputs/checkpoints/model_best.pth.tar \
    --config outputs/configurations/default_experiment/config_epoch_0500.yaml \
    --image data/unified/images/98.jpg \
    --mask-dir data/unified/masks \
    --output results/analysis.png

# Batch analysis
python -m crackseg.evaluation.cli \
    --checkpoint outputs/checkpoints/model_best.pth.tar \
    --config outputs/configurations/default_experiment/config_epoch_0500.yaml \
    --image-dir data/unified/images \
    --mask-dir data/unified/masks \
    --output-dir results/batch_analysis
```

## API Reference

### Core Components

#### `PredictionAnalyzer`

Main class for prediction analysis and evaluation.

```python
analyzer = PredictionAnalyzer(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    device: str | None = None,
    mask_dir: str | Path | None = None,
)
```

**Key Methods:**

- `analyze_image()`: Main analysis method with auto-inference
- `predict_single_image()`: Prediction-only analysis
- `analyze_with_ground_truth()`: Analysis with explicit mask
- `get_model_info()`: Get model information

#### `ModelLoader`

Handles model loading and configuration management.

```python
loader = ModelLoader(checkpoint_path)
model = loader.create_model(config)
info = loader.get_model_info(config)
```

#### `ImageProcessor`

Handles image and mask preprocessing.

```python
processor = ImageProcessor(config)
image_tensor = processor.load_and_preprocess_image(image_path)
mask = processor.load_mask(mask_path)
mask_path = processor.infer_mask_path(image_path, mask_dir)
```

### Metrics Components

#### `MetricsCalculator`

Computes segmentation metrics.

```python
calculator = MetricsCalculator()
calculator.update(predictions, targets)
metrics = calculator.compute()
iou = calculator.calculate_iou(predictions, targets)
```

#### `BatchProcessor`

Processes multiple images efficiently.

```python
processor = BatchProcessor(image_processor)
summary = processor.process_batch(
    image_dir, mask_dir, output_dir
)
```

### Visualization Components

#### `PredictionVisualizer`

Creates professional prediction visualizations.

```python
visualizer = PredictionVisualizer(config)
viz = visualizer.create_visualization(result, save_path)
grid = visualizer.create_comparison_grid(results, save_path)
```

#### `ExperimentVisualizer`

Creates experiment comparison visualizations.

```python
visualizer = ExperimentVisualizer()
visualizer.plot_training_curves(experiments_data, save_path)
visualizer.create_performance_radar(experiments_data, save_path)
```

## Migration Guide

### From Old API

**Old (deprecated):**

```python
from crackseg.evaluation.simple_prediction_analyzer import SimplePredictionAnalyzer

analyzer = SimplePredictionAnalyzer(checkpoint_path, config_path)
result = analyzer.analyze_image(image_path)
analyzer.create_visualization(result, output_path)
```

**New (recommended):**

```python
from crackseg.evaluation import PredictionAnalyzer, PredictionVisualizer

analyzer = PredictionAnalyzer(checkpoint_path, config_path)
result = analyzer.analyze_image(image_path)
visualizer = PredictionVisualizer(analyzer.config)
visualizer.create_visualization(result, output_path)
```

## Benefits of New Architecture

### 1. **Maintainability**

- **Small modules**: Each file <300 lines (vs 600+ lines before)
- **Clear interfaces**: Well-defined APIs between components
- **Easy debugging**: Isolated functionality makes issues easier to trace

### 2. **Extensibility**

- **Add new metrics**: Just extend `MetricsCalculator`
- **New visualizations**: Add to `visualization/` module
- **Custom processors**: Implement new `ImageProcessor` variants

### 3. **Testing**

- **Unit tests**: Each module can be tested independently
- **Mock components**: Easy to mock dependencies
- **Coverage**: Better test coverage with smaller modules

### 4. **Performance**

- **Lazy loading**: Components loaded only when needed
- **Memory efficient**: Smaller memory footprint
- **Parallel processing**: Batch processing optimized

## File Size Compliance

All modules comply with project standards:

| Module | Lines | Status |
|--------|-------|--------|
| `analyzer.py` | 250 | ✅ Under 300 |
| `model_loader.py` | 150 | ✅ Under 300 |
| `image_processor.py` | 150 | ✅ Under 300 |
| `calculator.py` | 120 | ✅ Under 300 |
| `batch_processor.py` | 180 | ✅ Under 300 |
| `prediction_viz.py` | 200 | ✅ Under 300 |
| `experiment_viz.py` | 250 | ✅ Under 300 |
| `prediction_cli.py` | 180 | ✅ Under 300 |

## Error Handling

The new architecture includes comprehensive error handling:

- **File validation**: Checks for missing files before processing
- **Type safety**: Full type annotations with Python 3.12+ features
- **Graceful degradation**: Continues processing even if some images fail
- **Detailed logging**: Comprehensive logging for debugging

## Future Enhancements

- **Real-time analysis**: Stream processing for live predictions
- **Advanced metrics**: Boundary-aware metrics for crack detection
- **Interactive visualizations**: Web-based visualization dashboard
- **Performance profiling**: Built-in performance analysis tools
