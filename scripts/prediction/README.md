# Prediction Scripts

This directory contains scripts for making predictions and performing inference with trained
CrackSeg models.

## Structure

- **predict_image.py**: Simple image prediction script with visualization

## Usage

### Basic Image Prediction

```bash
python scripts/prediction/predict_image.py --image path/to/image.jpg --mask-dir path/to/masks
```

### Example with project data

```bash
python scripts/prediction/predict_image.py --image data/unified/images/98.jpg --mask-dir data/unified/masks
```

## Features

- **Model loading**: Automatic checkpoint and configuration loading
- **Image preprocessing**: Standard ImageNet normalization
- **Visualization**: Side-by-side comparison with ground truth
- **Flexible input**: Support for various image formats
- **GPU acceleration**: Automatic CUDA detection and usage

## Requirements

- Trained model checkpoint in `outputs/checkpoints/model_best.pth.tar`
- Configuration file in `outputs/configurations/default_experiment/`
- Input images in supported formats (JPG, PNG, TIFF)

## Output

The script generates:

- Prediction visualization with confidence maps
- Side-by-side comparison with ground truth (if available)
- Performance metrics (IoU, Dice coefficient)

## Integration

This script integrates with:

- `src/crackseg/model/factory/`: Model creation utilities
- `src/crackseg/data/transforms.py`: Image preprocessing
- `src/crackseg/evaluation/metrics.py`: Performance evaluation
