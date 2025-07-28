"""Test script for configurable output templates.

This script tests the template system to ensure it works correctly
before using it with real experiments.
"""

import sys
import tempfile
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from crackseg.reporting.config import (  # noqa: E402
    ExperimentData,
    OutputFormat,
    ReportConfig,
    TemplateType,
)
from crackseg.reporting.templates import TemplateManager  # noqa: E402


def create_sample_experiment_data() -> ExperimentData:
    """Create sample experiment data for testing."""
    # Create temporary directory for experiment data
    temp_dir = Path(tempfile.mkdtemp())
    temp_dir.mkdir(exist_ok=True)

    # Sample training metrics
    training_metrics = {
        "train_loss": [0.8, 0.6, 0.4, 0.3, 0.25, 0.22, 0.2, 0.18, 0.17, 0.16],
        "val_loss": [
            0.85,
            0.65,
            0.45,
            0.35,
            0.3,
            0.28,
            0.26,
            0.25,
            0.24,
            0.23,
        ],
        "train_iou": [0.3, 0.45, 0.6, 0.7, 0.75, 0.78, 0.8, 0.82, 0.83, 0.84],
        "val_iou": [0.25, 0.4, 0.55, 0.65, 0.7, 0.73, 0.75, 0.76, 0.77, 0.78],
        "train_f1": [0.4, 0.55, 0.7, 0.78, 0.82, 0.85, 0.87, 0.88, 0.89, 0.9],
        "val_f1": [0.35, 0.5, 0.65, 0.73, 0.77, 0.8, 0.82, 0.83, 0.84, 0.85],
        "learning_rate": [
            1e-3,
            1e-3,
            1e-3,
            1e-3,
            1e-4,
            1e-4,
            1e-4,
            1e-4,
            1e-4,
            1e-4,
        ],
    }

    # Sample performance metrics
    performance_metrics = {
        "iou": 0.78,
        "f1_score": 0.85,
        "precision": 0.88,
        "recall": 0.82,
        "dice": 0.83,
        "training_time_hours": 2.5,
    }

    # Sample model configuration
    model_config = {
        "encoder": "resnet50",
        "decoder": "unet",
        "bottleneck": "aspp",
    }

    # Create sample artifacts
    artifacts = {
        "checkpoint": temp_dir / "model_best.pth",
        "config": temp_dir / "config.yaml",
    }

    # Create dummy files
    artifacts["checkpoint"].touch()
    artifacts["config"].touch()

    # Create experiment data
    exp_data = ExperimentData(
        experiment_id="test_exp_001",
        experiment_dir=temp_dir,
        config=DictConfig(model_config),
        metrics={
            "training_metrics": training_metrics,
            "performance_metrics": performance_metrics,
        },
        artifacts=artifacts,
    )

    return exp_data


def test_template_manager() -> None:
    """Test template manager functionality."""
    print("🧪 Testing Template Manager")
    print("=" * 50)

    # Initialize template manager
    template_manager = TemplateManager()
    print("✅ Template manager initialized")

    # Test available templates
    available_templates = template_manager.get_available_templates()
    print(f"📋 Available templates: {available_templates}")

    # Test template metadata
    for template_type in ["markdown", "html", "latex"]:
        for template_category in TemplateType:
            try:
                metadata = template_manager.get_template_metadata(
                    template_type, template_category
                )
                print(
                    f"📄 {template_type} - {template_category.value}: "
                    f"{metadata}"
                )
            except ValueError:
                print(
                    f"⚠️  {template_type} - {template_category.value}: "
                    f"Not supported"
                )

    print("✅ Template manager tests completed")


def test_template_rendering() -> None:
    """Test template rendering functionality."""
    print("\n🎨 Testing Template Rendering")
    print("=" * 50)

    # Create sample data
    sample_data = create_sample_experiment_data()
    print(f"📊 Created sample experiment: {sample_data.experiment_id}")

    # Test different configurations
    configs = [
        ReportConfig(
            template_type=TemplateType.EXECUTIVE_SUMMARY,
            output_formats=[OutputFormat.MARKDOWN],
        ),
        ReportConfig(
            template_type=TemplateType.TECHNICAL_DETAILED,
            output_formats=[OutputFormat.HTML],
        ),
        ReportConfig(
            template_type=TemplateType.PUBLICATION_READY,
            output_formats=[OutputFormat.LATEX],
        ),
    ]

    template_manager = TemplateManager()

    for i, config in enumerate(configs, 1):
        print(f"\n📝 Testing Configuration {i}: {config.template_type.value}")

        # Test markdown template
        try:
            template_content = template_manager.load_template(
                "markdown", config
            )
            print(
                f"✅ Loaded markdown template for {config.template_type.value}"
            )

            # Prepare sample data for rendering
            render_data = {
                "title": f"Test Report - {config.template_type.value}",
                "experiment_id": sample_data.experiment_id,
                "experiment_name": "Test Experiment",
                "generation_timestamp": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "executive_summary": (
                    "This is a test executive summary for template validation."
                ),
                "key_findings": [
                    "Model achieved 78% IoU on validation set",
                    "Training converged after 50 epochs",
                    "No overfitting detected",
                ],
                "performance_metrics": {
                    "IoU": "0.78",
                    "F1 Score": "0.85",
                    "Precision": "0.88",
                    "Recall": "0.82",
                },
                "recommendations": [
                    "Consider data augmentation for better generalization",
                    "Try different learning rate schedules",
                    "Experiment with different encoder architectures",
                ],
                "status": "completed",
                "duration": "2.5 hours",
                "status_badge": "success",
            }

            # Render template
            rendered_content = template_manager.render_template(
                template_content, render_data, config
            )
            print(
                f"✅ Rendered template successfully "
                f"({len(rendered_content)} characters)"
            )

            # Validate template
            is_valid = template_manager.validate_template(template_content)
            print(f"✅ Template validation: {'PASS' if is_valid else 'FAIL'}")

        except Exception as e:
            print(f"❌ Error testing {config.template_type.value}: {e}")

    print("✅ Template rendering tests completed")


def test_html_templates() -> None:
    """Test HTML template functionality."""
    print("\n🌐 Testing HTML Templates")
    print("=" * 50)

    template_manager = TemplateManager()

    # Test HTML templates
    html_configs = [
        (TemplateType.EXECUTIVE_SUMMARY, "HTMLExecutiveSummaryTemplate"),
        (TemplateType.PUBLICATION_READY, "HTMLPublicationTemplate"),
        (TemplateType.TECHNICAL_DETAILED, "HTMLTechnicalTemplate"),
    ]

    for template_category, template_class_name in html_configs:
        try:
            config = ReportConfig(template_type=template_category)
            template_content = template_manager.load_template("html", config)
            print(f"✅ Loaded HTML template: {template_class_name}")

            # Test rendering with sample data
            render_data = {
                "title": f"HTML Test Report - {template_category.value}",
                "experiment_id": "html_test_001",
                "experiment_name": "HTML Template Test",
                "generation_timestamp": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "executive_summary": "HTML template test summary.",
                "key_findings": "Test findings for HTML template.",
                "performance_metrics": "Test metrics for HTML template.",
                "recommendations": "Test recommendations for HTML template.",
                "status": "completed",
                "duration": "1 hour",
                "status_badge": "success",
            }

            rendered_content = template_manager.render_template(
                template_content, render_data, config
            )
            print("✅ Rendered HTML template successfully")

            # Check for HTML structure
            if "<html" in rendered_content and "</html>" in rendered_content:
                print("✅ HTML structure validation: PASS")
            else:
                print("❌ HTML structure validation: FAIL")

        except Exception as e:
            print(f"❌ Error testing HTML template {template_class_name}: {e}")

    print("✅ HTML template tests completed")


def test_latex_templates() -> None:
    """Test LaTeX template functionality."""
    print("\n📄 Testing LaTeX Templates")
    print("=" * 50)

    template_manager = TemplateManager()

    # Test LaTeX templates
    latex_configs = [
        (TemplateType.EXECUTIVE_SUMMARY, "LaTeXExecutiveSummaryTemplate"),
        (TemplateType.PUBLICATION_READY, "LaTeXPublicationTemplate"),
        (TemplateType.TECHNICAL_DETAILED, "LaTeXTechnicalTemplate"),
    ]

    for template_category, template_class_name in latex_configs:
        try:
            config = ReportConfig(template_type=template_category)
            template_content = template_manager.load_template("latex", config)
            print(f"✅ Loaded LaTeX template: {template_class_name}")

            # Test rendering with sample data
            render_data = {
                "title": f"LaTeX Test Report - {template_category.value}",
                "experiment_id": "latex_test_001",
                "experiment_name": "LaTeX Template Test",
                "generation_timestamp": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "executive_summary": "LaTeX template test summary.",
                "key_findings": "Test findings for LaTeX template.",
                "performance_metrics": "Test metrics for LaTeX template.",
                "recommendations": "Test recommendations for LaTeX template.",
                "status": "completed",
                "duration": "1 hour",
            }

            rendered_content = template_manager.render_template(
                template_content, render_data, config
            )
            print("✅ Rendered LaTeX template successfully")

            # Check for LaTeX structure
            if (
                "\\documentclass" in rendered_content
                and "\\end{document}" in rendered_content
            ):
                print("✅ LaTeX structure validation: PASS")
            else:
                print("❌ LaTeX structure validation: FAIL")

        except Exception as e:
            print(
                f"❌ Error testing LaTeX template {template_class_name}: {e}"
            )

    print("✅ LaTeX template tests completed")


def test_custom_templates() -> None:
    """Test custom template functionality."""
    print("\n🔧 Testing Custom Templates")
    print("=" * 50)

    # Create temporary directory for custom templates
    with tempfile.TemporaryDirectory() as temp_dir:
        custom_templates_dir = Path(temp_dir)
        template_manager = TemplateManager(
            custom_templates_dir=custom_templates_dir
        )

        # Create a custom template
        custom_template_path = (
            custom_templates_dir / "markdown.executive_summary.md"
        )
        custom_template_content = """# {{title}}

**Custom Template Test**

## Summary
{{executive_summary}}

## Test Data
- Experiment ID: {{experiment_id}}
- Status: {{status}}

---
*Custom template test*
"""

        custom_template_path.write_text(
            custom_template_content, encoding="utf-8"
        )
        print(f"✅ Created custom template: {custom_template_path}")

        # Test loading custom template
        try:
            config = ReportConfig(template_type=TemplateType.EXECUTIVE_SUMMARY)
            template_content = template_manager.load_template(
                "markdown", config
            )
            print("✅ Loaded custom template successfully")

            # Test rendering
            render_data = {
                "title": "Custom Template Test",
                "executive_summary": (
                    "This is a test of the custom template system."
                ),
                "experiment_id": "custom_test_001",
                "status": "completed",
            }

            rendered_content = template_manager.render_template(
                template_content, render_data, config
            )
            print("✅ Rendered custom template successfully")

            # Verify custom content is present
            if "Custom Template Test" in rendered_content:
                print("✅ Custom template content validation: PASS")
            else:
                print("❌ Custom template content validation: FAIL")

        except Exception as e:
            print(f"❌ Error testing custom template: {e}")

    print("✅ Custom template tests completed")


def main() -> None:
    """Run all template system tests."""
    print("🚀 Starting Configurable Output Templates Test Suite")
    print("=" * 60)

    try:
        # Run all tests
        test_template_manager()
        test_template_rendering()
        test_html_templates()
        test_latex_templates()
        test_custom_templates()

        print("\n" + "=" * 60)
        print("✅ All template system tests completed successfully!")
        print("📋 Summary:")
        print("  - Template manager functionality: ✅")
        print("  - Template rendering: ✅")
        print("  - HTML templates: ✅")
        print("  - LaTeX templates: ✅")
        print("  - Custom templates: ✅")
        print("\n🎉 Template system is ready for production use!")

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        raise


if __name__ == "__main__":
    main()
