# File Upload Functionality Guide

## Overview

The CrackSeg GUI application now supports uploading YAML configuration files directly from your
local computer. This feature provides a convenient way to use custom configurations without manually
copying files to the project directory.

## Features

### 🚀 Key Capabilities

- **Drag & Drop Upload**: Simply drag YAML files or use the file picker
- **Real-time Validation**: Automatic YAML syntax and structure validation
- **Progress Indication**: Visual feedback during upload process
- **File Safety**: Automatic file size and extension validation
- **Smart Naming**: Timestamped filenames prevent conflicts
- **Live Preview**: Instant configuration preview after upload

### 📋 Supported File Types

- `.yaml` files
- `.yml` files
- Maximum file size: **10 MB**
- UTF-8 encoding required

## How to Use

### 1. Access Upload Feature

Navigate to the **Configuration** page in the CrackSeg GUI. Look for the "📤 Upload Configuration
from Computer" expandable section.

### 2. Upload Process

1. **Click the upload section** to expand it
2. **Choose file** using the file picker or drag & drop
3. **Review file information** displayed automatically:
   - File name and size
   - Extension validation status
   - Size validation status
4. **Click "🚀 Process File"** to upload and validate

### 3. Validation Results

After upload, you'll see:

- **✅ Success indicators** for valid configurations
- **⚠️ Warnings** for potential issues
- **❌ Errors** for critical problems that must be fixed
- **📋 Configuration preview** showing structure and content

### 4. Integration

Once uploaded successfully:

- File is saved to `generated_configs/` with timestamp
- Configuration is automatically loaded into the application
- You can proceed to architecture visualization or training

## File Naming Convention

Uploaded files are automatically renamed to prevent conflicts:

```txt
Format: YYYYMMDD_HHMMSS_originalname.yaml
Example: 20231201_143022_my_config.yaml
```

## Validation Process

### Automatic Checks

1. **File Extension**: Must be `.yaml` or `.yml`
2. **File Size**: Must be ≤ 10 MB
3. **Encoding**: Must be valid UTF-8
4. **YAML Syntax**: Must be parseable YAML
5. **Structure Validation**: Checks for common configuration patterns
6. **Hydra Compatibility**: Validates against expected schema

### Validation Levels

- **Errors** 🔴: Critical issues that prevent usage
- **Warnings** 🟡: Potential problems, but configuration is usable
- **Info** 🔵: Suggestions for improvements

## Error Handling

### Common Issues and Solutions

#### Invalid File Extension

```txt
❌ Error: Invalid file extension '.txt'
✅ Solution: Ensure file has .yaml or .yml extension
```

#### File Too Large

```txt
❌ Error: File size (15.2 MB) exceeds maximum allowed size of 10 MB
✅ Solution: Reduce file size or split into multiple files
```

#### Invalid YAML Syntax

```txt
❌ Error: Invalid YAML syntax: mapping values are not allowed here
✅ Solution: Check YAML indentation and structure
```

#### Encoding Issues

```txt
❌ Error: File encoding error. Please ensure the file is UTF-8 encoded
✅ Solution: Save file with UTF-8 encoding
```

## Best Practices

### 📝 File Preparation

1. **Validate locally** before uploading using YAML validators
2. **Use consistent indentation** (2 or 4 spaces, not tabs)
3. **Add comments** to document complex configurations
4. **Test configurations** in small batches first

### 🔒 Security Considerations

- Only upload trusted YAML files
- Review validation results before using
- Be cautious with configurations from external sources
- Keep backup copies of working configurations

### 📊 Performance Tips

- **Smaller files upload faster** - consider splitting large configs
- **Use compression** for very large files before upload
- **Validate locally first** to catch issues early
- **Use meaningful filenames** for easier organization

## Configuration Examples

### Minimal Valid Configuration

```yaml
model:
  name: simple_unet
  input_channels: 3
  output_channels: 1

training:
  learning_rate: 0.001
  batch_size: 16
  epochs: 100
```

### Complete Configuration

```yaml
# CrackSeg Training Configuration
defaults:
  - model: unet_base
  - training: default
  - data: crack_dataset

model:
  architecture: "hybrid_unet"
  encoder:
    type: "swin_transformer_v2"
    pretrained: true
    window_size: 7
  decoder:
    type: "cnn_decoder"
    use_attention: true
  bottleneck:
    type: "aspp"
    dilation_rates: [1, 6, 12, 18]

training:
  optimizer:
    name: "adamw"
    learning_rate: 0.0001
    weight_decay: 0.01
  scheduler:
    name: "cosine"
    max_epochs: 200
  loss:
    name: "combined_loss"
    weights:
      dice: 0.5
      focal: 0.3
      boundary: 0.2

data:
  dataset:
    train_path: "data/train"
    val_path: "data/val"
    test_path: "data/test"
  transforms:
    augmentation: true
    normalize: true
  dataloader:
    batch_size: 32
    num_workers: 4
```

## Troubleshooting

### Upload Not Working

1. **Check file format**: Ensure `.yaml` or `.yml` extension
2. **Verify file size**: Must be under 10 MB
3. **Test with simple file**: Try uploading a minimal configuration first
4. **Check browser**: Some ad blockers may interfere with uploads

### Validation Failures

1. **Check YAML syntax**: Use online YAML validators
2. **Verify indentation**: Ensure consistent spacing
3. **Review error messages**: Follow specific guidance provided
4. **Compare with examples**: Use working configurations as templates

### Performance Issues

1. **Reduce file size**: Remove unnecessary comments or sections
2. **Check network**: Ensure stable internet connection
3. **Try smaller batches**: Upload configurations incrementally
4. **Clear browser cache**: Remove old cached data

## Advanced Usage

### Batch Operations

For multiple files:

1. Upload one file at a time
2. Validate each individually
3. Use the file browser to manage uploaded configurations
4. Create a master configuration that imports others

### Integration with Version Control

1. Upload configurations to `generated_configs/`
2. Copy validated files to your project's `configs/` directory
3. Commit to version control
4. Share with team members

### Custom Validation

The upload system uses the same validation as the configuration editor, ensuring consistency across
all configuration methods.

## API Reference

For developers wanting to extend the upload functionality, see:

- `scripts/gui/utils/config/io.py` - Core upload functions
- `scripts/gui/components/file_upload_component.py` - UI components
- `tests/unit/gui/test_file_upload.py` - Test examples

## Support

If you encounter issues with file upload:

1. Check the validation messages for specific guidance
2. Review this guide for common solutions
3. Test with minimal example configurations
4. Report persistent issues to the development team

---

Last updated: June 2025
