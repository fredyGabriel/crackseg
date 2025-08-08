# Configurable Crack Image Processing Script

## Description

This script processes pavement crack images to square format through intelligent cropping
that preserves the maximum number of crack pixels. The script automatically determines
the target size as the minimum dimension of the input image.

## Features

- **Configurable dimensions**: Accepts any input dimensions and automatically calculates square output
- **Intelligent analysis**: Analyzes binarized masks to determine which side contains more cracks
- **Optimized cropping**: Crops the image and mask on the side with higher crack density
- **Sequential renaming**: Renames files as 1.jpg, 2.jpg, etc.
- **Robust validation**: Verifies dimensions and handles errors
- **Complete logging**: Generates detailed processing logs
- **Statistical reports**: Provides processing statistics

## Supported Formats

### Common Transformations

| Input Dimensions | Output Dimensions | Example Use Case |
|------------------|-------------------|------------------|
| 480x320 | 320x320 | CFD dataset |
| 640x360 | 360x360 | Original dataset |
| 800x600 | 600x600 | High-resolution images |
| 1024x768 | 768x768 | Large format images |

### Algorithm Logic

The script automatically calculates the target size as:

```bash
target_size = min(input_width, input_height)
```

This ensures:

- **Square output**: All processed images are square
- **Maximum preservation**: Uses the largest possible square that fits
- **Consistent format**: All outputs have the same dimensions

## Data Structure

### Input

```txt
input_directory/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── masks/
    ├── image1.png
    ├── image2.png
    └── ...
```

### Output

```txt
output_directory/
├── images/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── masks/
    ├── 1.png
    ├── 2.png
    └── ...
```

## Processing Algorithm

### 1. Crack Density Analysis

- Divides the mask into two horizontal halves
- Calculates crack density in each half
- Determines which side has more cracks

### 2. Intelligent Cropping

- **Left side**: Keeps pixels from the beginning of the image
- **Right side**: Keeps pixels from the end of the image
- Applies the same crop to image and mask

### 3. Sequential Renaming

- Renames files sequentially (1, 2, 3, ...)
- Maintains correspondence between image and mask

## Usage

### Basic Command

```bash
python crop_crack_images_configurable.py \
    --input_dir "data/CFD" \
    --output_dir "data/CFD_processed" \
    --expected_width 480 \
    --expected_height 320
```

### Complete Command

```bash
python crop_crack_images_configurable.py \
    --input_dir "data/CFD" \
    --output_dir "data/CFD_processed" \
    --expected_width 480 \
    --expected_height 320 \
    --image_dir "images" \
    --mask_dir "masks" \
    --log_level "INFO"
```

### Parameters

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| `--input_dir` | Input directory | - | ✅ |
| `--output_dir` | Output directory | - | ✅ |
| `--expected_width` | Expected input width | - | ✅ |
| `--expected_height` | Expected input height | - | ✅ |
| `--image_dir` | Images subdirectory | "images" | ❌ |
| `--mask_dir` | Masks subdirectory | "masks" | ❌ |
| `--log_level` | Logging level | "INFO" | ❌ |

### Logging Levels

- `DEBUG`: Detailed processing information
- `INFO`: General information (recommended)
- `WARNING`: Only warnings and errors
- `ERROR`: Only errors

## Requirements

### Python Libraries

```bash
pip install opencv-python numpy
```

### File Structure

- Images must be in JPG format
- Masks must be in PNG format
- Masks must be binarized (0 and 255)
- All images must have the same dimensions

## Usage Examples

### Example 1: CFD Dataset (480x320 → 320x320)

```bash
# Process CFD dataset
python crop_crack_images_configurable.py \
    --input_dir "data/CFD" \
    --output_dir "data/CFD_320x320" \
    --expected_width 480 \
    --expected_height 320 \
    --log_level "DEBUG"
```

### Example 2: Original Dataset (640x360 → 360x360)

```bash
# Process original dataset
python crop_crack_images_configurable.py \
    --input_dir "data/original_dataset" \
    --output_dir "data/original_360x360" \
    --expected_width 640 \
    --expected_height 360 \
    --log_level "INFO"
```

### Example 3: High-Resolution Images (800x600 → 600x600)

```bash
# Process high-resolution images
python crop_crack_images_configurable.py \
    --input_dir "data/high_res" \
    --output_dir "data/high_res_600x600" \
    --expected_width 800 \
    --expected_height 600 \
    --log_level "INFO"
```

## Verification

### Verify Input Dimensions

```bash
# Check input image dimensions
python -c "
import cv2
import os
for img_file in os.listdir('data/CFD/images')[:5]:
    img = cv2.imread(f'data/CFD/images/{img_file}')
    print(f'{img_file}: {img.shape}')
"
```

### Verify Output Dimensions

```bash
# Check output image dimensions
python -c "
import cv2
import os
for i in range(1, 6):
    img = cv2.imread(f'data/CFD_320x320/images/{i}.jpg')
    mask = cv2.imread(f'data/CFD_320x320/masks/{i}.png', cv2.IMREAD_GRAYSCALE)
    print(f'File {i}: Image {img.shape}, Mask {mask.shape}')
"
```

## Script Output

### Processing Log

```txt
2024-01-15 10:30:00 - INFO - Starting configurable crack image processing
2024-01-15 10:30:00 - INFO - Input directory: data/CFD
2024-01-15 10:30:00 - INFO - Output directory: data/CFD_320x320
2024-01-15 10:30:00 - INFO - Expected input dimensions: 480x320
2024-01-15 10:30:00 - INFO - Found 100 file pairs to process
2024-01-15 10:30:01 - INFO - Processed: img1.jpg -> 1.jpg (crop: left, size: 320x320)
2024-01-15 10:30:01 - INFO - Processed: img2.jpg -> 2.jpg (crop: right, size: 320x320)
...
2024-01-15 10:30:05 - INFO - ==================================================
2024-01-15 10:30:05 - INFO - PROCESSING REPORT
2024-01-15 10:30:05 - INFO - ==================================================
2024-01-15 10:30:05 - INFO - Total files: 100
2024-01-15 10:30:05 - INFO - Successfully processed: 98
2024-01-15 10:30:05 - INFO - Errors: 2
2024-01-15 10:30:05 - INFO - Left cuts: 65
2024-01-15 10:30:05 - INFO - Right cuts: 33
2024-01-15 10:30:05 - INFO - Target sizes used: {(320, 320)}
2024-01-15 10:30:05 - INFO - Total time: 5.23 seconds
```

## Validations

### Input

- ✅ Verifies that images match expected dimensions
- ✅ Verifies that masks match expected dimensions
- ✅ Verifies that image and mask have the same dimensions
- ✅ Verifies that masks are binarized

### Output

- ✅ Verifies that output images are square
- ✅ Verifies that output masks are square
- ✅ Verifies that filenames are sequential
- ✅ Verifies that each image has its corresponding mask

## Error Handling

### Common Errors

1. **Incorrect dimensions**: Images don't match expected dimensions
2. **Corrupted file**: Cannot load image or mask
3. **Missing mask**: Corresponding mask not found
4. **Non-existent directory**: Input directory does not exist

### Behavior

- Script continues processing other files if errors are found
- Errors are logged
- Final report is generated with error statistics

## CrackSeg Integration

### Dataset Configuration

After processing images, update the configuration:

```yaml
# configs/data/default.yaml
data:
  image_size: [320, 320]  # For CFD dataset
  # OR
  image_size: [360, 360]  # For original dataset
```

### Transformations

```yaml
# configs/data/transform/augmentations.yaml
train:
  - name: Resize
    params:
      height: 320  # For CFD dataset
      width: 320   # For CFD dataset
```

### Model

```yaml
# configs/experiments/swinv2_hybrid/swinv2_hybrid_experiment.yaml
model:
  encoder:
    img_size: 320  # For CFD dataset
    target_img_size: 320  # New parameter
```

## Performance Considerations

### Memory Usage

- Processes files sequentially to avoid memory issues
- Automatic memory cleanup after each file
- Efficient logging with different levels

### Processing Speed

- **Time per file**: ~0.05 seconds
- **Memory**: ~50MB per processed file
- **Throughput**: ~20 files per second

## Troubleshooting

### Problem: "Incorrect image dimensions"

**Solution**: Verify that all images have the expected dimensions.

```bash
# Check actual dimensions
python -c "
import cv2
import os
for img_file in os.listdir('data/CFD/images')[:3]:
    img = cv2.imread(f'data/CFD/images/{img_file}')
    print(f'{img_file}: {img.shape}')
"
```

### Problem: "No image-mask file pairs found"

**Solution**: Verify that `images` and `masks` directories exist and contain files.

### Problem: "Mask is not completely binarized"

**Solution**: Verify that masks contain only values 0 and 255.

## Migration from Original Script

### Old Script Usage

```bash
# Old script (hardcoded 640x360)
python crop_crack_images.py \
    --input_dir "data/original_dataset" \
    --output_dir "data/processed_dataset"
```

### New Script Usage

```bash
# New script (configurable)
python crop_crack_images_configurable.py \
    --input_dir "data/original_dataset" \
    --output_dir "data/processed_dataset" \
    --expected_width 640 \
    --expected_height 360
```

### Benefits of New Script

1. **Flexibility**: Works with any input dimensions
2. **Automatic sizing**: Calculates optimal square size
3. **Better logging**: Shows target dimensions in logs
4. **Validation**: Ensures all images have expected dimensions
5. **Statistics**: Reports target sizes used

## Contributions

To improve the script:

1. Add support for other image formats (PNG, TIFF)
2. Implement parallel processing for large datasets
3. Add more cropping options (center crop, random crop)
4. Improve crack detection algorithms
5. Add image quality validation
6. Support for different aspect ratio handling

## References

- **Original Script**: [crop_crack_images.py](crop_crack_images.py)
- **Project Standards**: [coding-standards.mdc](/.cursor/rules/coding-standards.mdc)
- **Development Workflow**: [development-workflow.mdc](/.cursor/rules/development-workflow.mdc)
