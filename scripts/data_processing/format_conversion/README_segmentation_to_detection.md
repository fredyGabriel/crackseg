# Segmentation to Object Detection Conversion

This directory contains scripts to convert segmentation masks to object detection annotations in
various formats (YOLO, COCO, Pascal VOC).

## Overview

The conversion process analyzes binary segmentation masks to extract object contours and generate
bounding box annotations. This is particularly useful for the CrackSeg project where we have
detailed segmentation masks but need bounding boxes for object detection training.

## Scripts

### 1. `segmentation_to_detection.py`

Core conversion library with the `MaskToDetectionConverter` class.

**Features:**

- Supports YOLO, COCO, and Pascal VOC formats
- Configurable contour detection parameters
- Automatic image dimension detection
- Comprehensive error handling and statistics

### 2. `test_segmentation_conversion.py`

Test script for validating the conversion on sample images.

**Features:**

- Visual validation with matplotlib plots
- Sample-based testing
- Debugging and verification

### 3. `convert_crackseg_dataset.py`

Production batch conversion script for the entire dataset.

**Features:**

- Organized output structure
- Dataset validation
- Comprehensive statistics and info files

## Quick Start

### Basic Usage

```bash
# Test on a few samples first
python test_segmentation_conversion.py \
  --samples 368 357 355

# Convert entire dataset to YOLO format
python convert_crackseg_dataset.py \
  --output-dir data/BD_estudio/2-Object_detection \
  --format yolo \
  --organize
```

### Advanced Usage

```bash
# Convert to COCO format with custom parameters
python segmentation_to_detection.py \
  --image-dir "data/BD_estudio/1-Segmentation/Original image" \
  --mask-dir "data/BD_estudio/1-Segmentation/Ground truth" \
  --output-dir "data/BD_estudio/2-Object_detection" \
  --format coco \
  --class-name crack \
  --class-id 0
```

## Output Formats

### YOLO Format

```txt
# classes.txt
crack

# annotation.txt (one per image)
0 0.5 0.3 0.2 0.1
```

### COCO Format

```json
{
  "images": [...],
  "annotations": [...],
  "categories": [{"id": 0, "name": "crack"}]
}
```

### Pascal VOC Format

```xml
<annotation>
  <object>
    <name>crack</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>50</ymin>
      <xmax>200</xmax>
      <ymax>150</ymax>
    </bndbox>
  </object>
</annotation>
```

## Algorithm Details

### Contour Detection Process

1. **Mask Preprocessing**
   - Convert to grayscale if needed
   - Apply binary threshold (127)
   - Ensure proper binary values (0, 255)

2. **Contour Extraction**
   - Use `cv2.findContours()` with `RETR_EXTERNAL`
   - Filter by minimum area (default: 50 pixels)
   - Apply contour approximation with configurable epsilon

3. **Bounding Box Generation**
   - Calculate bounding rectangle for each contour
   - Convert to appropriate format (normalized for YOLO)
   - Include area and additional metadata

### Key Parameters

- **min_area**: Minimum contour area (default: 50 pixels)
- **approximation_epsilon**: Contour simplification factor (default: 0.002)
- **class_id**: Object class identifier (default: 0)
- **class_name**: Object class name (default: "crack")

## Best Practices

### For Crack Detection

1. **Parameter Tuning**

   ```python
   # For thin cracks
   min_area = 25  # Lower threshold
   approximation_epsilon = 0.001  # More precise contours

   # For general objects
   min_area = 100  # Higher threshold
   approximation_epsilon = 0.005  # Simplified contours
   ```

2. **Quality Control**
   - Always test on samples first
   - Use visualization to verify results
   - Check conversion statistics
   - Validate dataset completeness

3. **Format Selection**
   - **YOLO**: Best for real-time detection, simpler format
   - **COCO**: Best for research, comprehensive metadata
   - **Pascal VOC**: Good for traditional CV pipelines

## Troubleshooting

### Common Issues

1. **No detections found**
   - Check mask is binary (0 and 255 values)
   - Reduce `min_area` parameter
   - Verify mask file format

2. **Too many small detections**
   - Increase `min_area` parameter
   - Apply morphological operations to masks

3. **Inaccurate bounding boxes**
   - Adjust `approximation_epsilon`
   - Check image-mask correspondence

### Error Messages

- `"Could not load mask"`: File format or path issue
- `"Shape mismatch"`: Image and mask dimensions differ
- `"No mask files found"`: Check directory and file extensions

## Integration with CrackSeg

### Project Structure

```txt
data/BD_estudio/
├── 1-Segmentation/
│   ├── Original image/    # Source images
│   └── Ground truth/      # Segmentation masks
└── 2-Object detection/    # Generated annotations
    ├── images/           # Organized images
    ├── annotations/      # Annotation files
    ├── classes.txt       # Class definitions
    └── dataset_info.txt  # Dataset statistics
```

### Usage in Training Pipeline

```python
# In training configuration
data:
  train: data/BD_estudio/2-Object_detection/images
  val: data/BD_estudio/2-Object_detection/images
  names: data/BD_estudio/2-Object_detection/classes.txt
  nc: 1  # Number of classes
```

## Performance Considerations

- **Speed**: ~10-50 images/second depending on mask complexity
- **Memory**: Low memory footprint, processes one image at a time
- **Accuracy**: Depends on mask quality and parameter tuning

## Dependencies

```python
opencv-python>=4.5.0
numpy>=1.19.0
matplotlib>=3.3.0  # For visualization
tqdm>=4.60.0       # For progress bars
```

## Examples Output

### Sample Conversion Statistics

```txt
CONVERSION STATISTICS
==================================================
Total images processed: 95
Images with detections: 89
Total detections found: 142
Failed conversions: 6
Success rate: 93.7%
Average detections per image: 1.6
```

### Generated Files (YOLO format)

```txt
data/BD_estudio/2-Object_detection/
├── annotations/
│   ├── 368.txt
│   ├── 357.txt
│   └── classes.txt
├── images/
│   ├── 368.jpg
│   └── 357.jpg
└── dataset_info.txt
```

## Future Enhancements

1. **Multi-class Support**: Extension for multiple crack types
2. **Polygon Annotations**: Keep segmentation detail for instance segmentation
3. **Data Augmentation**: Integrated augmentation during conversion
4. **Quality Metrics**: Automatic quality assessment of conversions
5. **GUI Interface**: User-friendly interface for parameter tuning

## References

- [YOLO Annotation Format](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
- [COCO Dataset Format](https://cocodataset.org/#format-data)
- [Pascal VOC Format](http://host.robots.ox.ac.uk/pascal/VOC/)
- [OpenCV Contour Detection](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
