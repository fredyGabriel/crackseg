# Data Directory

This directory contains the data used for training, validation, and testing of the pavement crack
segmentation model.

## Directory Structure

```text
data/
├── train/
│   ├── images/     # Training images
│   └── masks/      # Corresponding segmentation masks
├── val/
│   ├── images/     # Validation images
│   └── masks/      # Corresponding segmentation masks
└── test/
    ├── images/     # Test images
    └── masks/      # Corresponding segmentation masks
```

## Data Format

- **Images**: PNG format
- **Dimensions**: Variable (resized during preprocessing)
- **Masks**: Binary PNG format (0 = background, 255 = crack)

## Data Organization

The dataset is split into three subsets:

- **Training (train/)**: 70% of the dataset
- **Validation (val/)**: 15% of the dataset
- **Test (test/)**: 15% of the dataset

## Preprocessing

Images are preprocessed during training according to the configuration in `configs/data/transform/`.

Common transformations include:

- Resizing to standard dimensions
- Normalization
- Data augmentation (training only)

## Usage Guidelines

When adding new data:

1. Follow the existing directory structure
2. Ensure images and masks have matching filenames
3. Document any manual preprocessing performed

### Important Notes

1. Verify image-mask correspondence before training
2. Maintain consistency in format and data structure

## Example Data

The `examples/` directory contains dummy data files that can be used for:

- Testing the data pipeline
- Verifying the model's basic functionality
- Running quick experiments

## Data Configuration

The data loading configuration can be found in:

- `configs/data/default.yaml`: Main data configuration
- `configs/data/transform.yaml`: Data augmentation settings
- `configs/data/dataloader.yaml`: DataLoader parameters
