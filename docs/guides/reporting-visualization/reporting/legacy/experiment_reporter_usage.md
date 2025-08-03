# ExperimentReporter Usage Guide

## Quick Start

### Basic Single Experiment Report

```python
from pathlib import Path
from crackseg.reporting import ExperimentReporter, ReportConfig, OutputFormat, TemplateType


# Initialize reporter with basic configuration
config = ReportConfig(
    output_formats=[OutputFormat.MARKDOWN, OutputFormat.HTML],
    output_dir=Path("reports"),
    template_type=TemplateType.EXECUTIVE_SUMMARY
)

reporter = ExperimentReporter(config=config)

# Generate report for a single experiment
experiment_dir = Path("outputs/experiments/20250127-123456-default")
metadata = reporter.generate_single_experiment_report(experiment_dir)

print(f"✅ Report generated successfully!")
print(f"📄 Report ID: {metadata.report_id}")
print(f"📁 Output files: {list(metadata.file_paths.keys())}")
print(f"⏱️  Generation time: {metadata.generation_time_seconds:.2f}s")
```

### Comparison Report

```python
# Generate comparison report for multiple experiments
experiment_dirs = [
    Path("outputs/experiments/swin_v2_baseline"),
    Path("outputs/experiments/swin_v2_improved"),
    Path("outputs/experiments/resnet50_comparison")
]

metadata = reporter.generate_comparison_report(
    experiment_dirs,
    report_type=TemplateType.COMPARISON_REPORT
)

print(f"✅ Comparison report generated!")
print(f"📊 Compared {len(metadata.experiment_ids)} experiments")
print(f"📁 Files: {list(metadata.file_paths.keys())}")
```

## Configuration Options

### Output Formats

```python
from crackseg.reporting import OutputFormat


# Markdown and HTML for web viewing
config = ReportConfig(
    output_formats=[OutputFormat.MARKDOWN, OutputFormat.HTML]
)

# LaTeX and PDF for academic publications
config = ReportConfig(
    output_formats=[OutputFormat.LATEX, OutputFormat.PDF]
)

# JSON for machine processing
config = ReportConfig(
    output_formats=[OutputFormat.JSON]
)
```

### Template Types

```python
from crackseg.reporting import TemplateType


# Executive summary for stakeholders
config = ReportConfig(
    template_type=TemplateType.EXECUTIVE_SUMMARY
)

# Detailed technical analysis
config = ReportConfig(
    template_type=TemplateType.TECHNICAL_DETAILED
)

# Publication-ready format
config = ReportConfig(
    template_type=TemplateType.PUBLICATION_READY
)

# Performance-focused analysis
config = ReportConfig(
    template_type=TemplateType.PERFORMANCE_ANALYSIS
)
```

### Custom Performance Thresholds

```python
# Set custom performance thresholds
config = ReportConfig(
    performance_thresholds={
        "iou_min": 0.8,        # Minimum IoU score
        "f1_min": 0.85,        # Minimum F1 score
        "precision_min": 0.9,   # Minimum precision
        "recall_min": 0.8,      # Minimum recall
        "loss_max": 0.1,        # Maximum loss
    }
)
```

### Visualization Settings

```python
# High-quality figures for publications
config = ReportConfig(
    figure_dpi=600,
    figure_format="pdf",
    chart_theme="plotly_white",
    color_palette="viridis"
)

# Standard figures for web viewing
config = ReportConfig(
    figure_dpi=300,
    figure_format="png",
    chart_theme="plotly_dark",
    color_palette="plasma"
)
```

## Content Control

### Include/Exclude Components

```python
# Comprehensive report with all components
config = ReportConfig(
    include_performance_analysis=True,
    include_comparison_charts=True,
    include_publication_figures=True,
    include_recommendations=True,
    include_trend_analysis=True
)

# Minimal executive summary
config = ReportConfig(
    include_performance_analysis=False,
    include_comparison_charts=False,
    include_publication_figures=False,
    include_recommendations=True,
    include_trend_analysis=False
)
```

### Analysis Settings

```python
# Enable anomaly detection and trend analysis
config = ReportConfig(
    trend_analysis_window=10,  # Analyze last 10 epochs
    anomaly_detection_enabled=True
)

# Disable advanced analysis for faster generation
config = ReportConfig(
    trend_analysis_window=0,
    anomaly_detection_enabled=False
)
```

## Advanced Usage

### Custom Template Directory

```python
# Use custom templates
config = ReportConfig(
    custom_templates_dir=Path("custom_templates"),
    default_template_vars={
        "company_name": "CrackSeg Labs",
        "project_name": "Pavement Crack Detection",
        "contact_email": "reports@crackseg.com"
    }
)
```

### Batch Processing

```python
import glob
from pathlib import Path

# Find all recent experiments
experiment_pattern = "outputs/experiments/*"
experiment_dirs = [Path(d) for d in glob.glob(experiment_pattern)]

# Generate reports for all experiments
for exp_dir in experiment_dirs:
    if reporter.validate_experiment_directory(exp_dir):
        metadata = reporter.generate_single_experiment_report(exp_dir)
        print(f"✅ Generated report for {exp_dir.name}")
    else:
        print(f" Skipped invalid experiment: {exp_dir.name}")
```

### Error Handling

```python
try:
    metadata = reporter.generate_single_experiment_report(experiment_dir)

    if metadata.success:
        print(f"✅ Report generated successfully")
        print(f"📁 Files: {metadata.file_paths}")
    else:
        print(f" Report generation failed: {metadata.error_message}")

except Exception as e:
    print(f"💥 Unexpected error: {e}")
```

## Integration Examples

### With Training Pipeline

```python
# Integrate with training pipeline
def on_training_complete(experiment_dir: Path):
    """Generate report when training completes."""

    config = ReportConfig(
        output_formats=[OutputFormat.HTML, OutputFormat.MARKDOWN],
        template_type=TemplateType.EXECUTIVE_SUMMARY
    )

    reporter = ExperimentReporter(config=config)
    metadata = reporter.generate_single_experiment_report(experiment_dir)

    # Send notification
    if metadata.success:
        print(f"📊 Training report ready: {metadata.file_paths}")
    else:
        print(f"⚠️  Report generation failed: {metadata.error_message}")
```

### With CI/CD Pipeline

```python
# GitHub Actions integration
def generate_ci_report():
    """Generate report for CI/CD pipeline."""

    config = ReportConfig(
        output_formats=[OutputFormat.MARKDOWN, OutputFormat.JSON],
        template_type=TemplateType.PERFORMANCE_ANALYSIS,
        include_performance_analysis=True,
        include_recommendations=True
    )

    reporter = ExperimentReporter(config=config)

    # Find latest experiment
    experiment_dirs = sorted(
        Path("outputs/experiments").glob("*"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if experiment_dirs:
        metadata = reporter.generate_single_experiment_report(experiment_dirs[0])

        # Upload artifacts
        for format_name, file_path in metadata.file_paths.items():
            print(f"📤 Uploading {format_name} report: {file_path}")
            # upload_to_artifacts(file_path)

    return metadata
```

### With Web Dashboard

```python
# Streamlit integration
import streamlit as st
from pathlib import Path

def render_report_generator():
    """Render report generator in Streamlit."""

    st.title("Experiment Report Generator")

    # Experiment selection
    experiment_dirs = list(Path("outputs/experiments").glob("*"))
    selected_experiments = st.multiselect(
        "Select experiments",
        [d.name for d in experiment_dirs],
        default=[experiment_dirs[0].name] if experiment_dirs else []
    )

    # Configuration options
    col1, col2 = st.columns(2)

    with col1:
        output_formats = st.multiselect(
            "Output formats",
            ["markdown", "html", "pdf", "latex", "json"],
            default=["markdown", "html"]
        )

        template_type = st.selectbox(
            "Template type",
            ["executive_summary", "technical_detailed", "publication_ready", "comparison_report"]
        )

    with col2:
        include_performance = st.checkbox("Include performance analysis", True)
        include_figures = st.checkbox("Include publication figures", True)
        include_recommendations = st.checkbox("Include recommendations", True)

    # Generate report
    if st.button("Generate Report"):
        config = ReportConfig(
            output_formats=[OutputFormat(f) for f in output_formats],
            template_type=TemplateType(template_type),
            include_performance_analysis=include_performance,
            include_publication_figures=include_figures,
            include_recommendations=include_recommendations
        )

        reporter = ExperimentReporter(config=config)

        selected_dirs = [Path("outputs/experiments") / exp for exp in selected_experiments]

        if len(selected_dirs) == 1:
            metadata = reporter.generate_single_experiment_report(selected_dirs[0])
        else:
            metadata = reporter.generate_comparison_report(selected_dirs)

        if metadata.success:
            st.success("✅ Report generated successfully!")
            st.json(metadata.file_paths)
        else:
            st.error(f" Report generation failed: {metadata.error_message}")
```

## Best Practices

### 1. Directory Structure

Ensure your experiment directories follow the expected structure:

```bash
outputs/experiments/
├── 20250127-123456-default/
│   ├── config.yaml              # Experiment configuration
│   ├── metrics/
│   │   ├── complete_summary.json  # Final metrics
│   │   └── metrics.jsonl         # Per-epoch metrics
│   ├── checkpoints/
│   │   ├── model_best.pth       # Best model
│   │   └── model_latest.pth     # Latest model
│   └── logs/
│       └── training.log         # Training logs
```

### 2. Configuration Management

```python
# Use environment-specific configurations
import os

if os.getenv("ENVIRONMENT") == "production":
    config = ReportConfig(
        output_formats=[OutputFormat.HTML, OutputFormat.PDF],
        template_type=TemplateType.EXECUTIVE_SUMMARY,
        include_performance_analysis=True,
        include_recommendations=True
    )
else:
    config = ReportConfig(
        output_formats=[OutputFormat.MARKDOWN],
        template_type=TemplateType.TECHNICAL_DETAILED,
        include_performance_analysis=False
    )
```

### 3. Error Handling

```python
# Validate experiments before processing
def safe_generate_report(experiment_dir: Path):
    """Safely generate report with validation."""

    if not reporter.validate_experiment_directory(experiment_dir):
        print(f" Invalid experiment directory: {experiment_dir}")
        return None

    try:
        metadata = reporter.generate_single_experiment_report(experiment_dir)
        return metadata
    except Exception as e:
        print(f"💥 Error generating report: {e}")
        return None
```

### 4. Performance Optimization

```python
# Use minimal configuration for quick reports
quick_config = ReportConfig(
    output_formats=[OutputFormat.MARKDOWN],
    include_performance_analysis=False,
    include_publication_figures=False,
    include_recommendations=False,
    include_trend_analysis=False
)

# Use comprehensive configuration for detailed reports
detailed_config = ReportConfig(
    output_formats=[OutputFormat.HTML, OutputFormat.PDF, OutputFormat.LATEX],
    include_performance_analysis=True,
    include_publication_figures=True,
    include_recommendations=True,
    include_trend_analysis=True,
    figure_dpi=600
)
```

## Troubleshooting

### Common Issues

1. **Missing experiment files**: Ensure all required files exist
2. **Invalid configuration**: Check configuration parameters
3. **Output directory permissions**: Ensure write permissions
4. **Memory issues**: Reduce figure DPI or disable features

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Generate report with detailed logging
metadata = reporter.generate_single_experiment_report(experiment_dir)
```

This guide provides comprehensive usage examples and best practices for the ExperimentReporter system.
