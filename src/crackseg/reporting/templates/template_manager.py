"""Template manager for configurable output templates.

This module provides the core template management system for generating
professional experiment reports in multiple formats with customizable styling.
"""

import logging
from pathlib import Path
from typing import Any

from crackseg.reporting.config import ReportConfig, TemplateType
from crackseg.reporting.interfaces import (
    TemplateManager as TemplateManagerInterface,
)

logger = logging.getLogger(__name__)


class TemplateManager(TemplateManagerInterface):
    """Main template manager for experiment reports."""

    def __init__(self, custom_templates_dir: Path | None = None) -> None:
        """Initialize template manager.

        Args:
            custom_templates_dir: Optional directory for custom templates.
        """
        self.custom_templates_dir = custom_templates_dir
        self._template_cache: dict[str, str] = {}
        self.logger = logging.getLogger(__name__)

    def load_template(
        self,
        template_type: str,
        config: ReportConfig,
    ) -> str:
        """Load template content for the specified type.

        Args:
            template_type: Type of template to load.
            config: Report configuration.

        Returns:
            Template content as string.

        Raises:
            ValueError: If template type is not supported.
        """
        # Check cache first
        cache_key = f"{template_type}_{config.template_type.value}"
        if cache_key in self._template_cache:
            return self._template_cache[cache_key]

        # Try custom templates first
        if self.custom_templates_dir:
            custom_path = (
                self.custom_templates_dir
                / f"{template_type}.{config.template_type.value}.md"
            )
            if custom_path.exists():
                template_content = custom_path.read_text(encoding="utf-8")
                self._template_cache[cache_key] = template_content
                return template_content

        # Load built-in template
        template_content = self._load_builtin_template(
            template_type, config.template_type
        )
        self._template_cache[cache_key] = template_content
        return template_content

    def render_template(
        self,
        template_content: str,
        data: dict[str, Any],
        config: ReportConfig,
    ) -> str:
        """Render template with provided data.

        Args:
            template_content: Template content to render.
            data: Data to inject into template.
            config: Report configuration.

        Returns:
            Rendered template content.
        """
        try:
            # Simple template rendering with variable substitution
            rendered_content = template_content

            # Replace template variables
            for key, value in data.items():
                placeholder = f"{{{{{key}}}}}"
                if isinstance(value, str | int | float | bool):
                    rendered_content = rendered_content.replace(
                        placeholder, str(value)
                    )
                elif isinstance(value, list):
                    # Handle list rendering
                    list_content = self._render_list_value(value)
                    rendered_content = rendered_content.replace(
                        placeholder, list_content
                    )
                elif isinstance(value, dict):
                    # Handle dict rendering
                    dict_content = self._render_dict_value(value)
                    rendered_content = rendered_content.replace(
                        placeholder, dict_content
                    )

            # Apply configuration-specific styling
            rendered_content = self._apply_styling(rendered_content, config)

            return rendered_content

        except Exception as e:
            self.logger.error(f"Error rendering template: {e}")
            raise ValueError(f"Template rendering failed: {e}") from e

    def get_available_templates(self) -> list[str]:
        """Get list of available templates.

        Returns:
            List of available template types.
        """
        return [
            "executive_summary",
            "technical_detailed",
            "publication_ready",
            "comparison_report",
            "performance_analysis",
        ]

    def _load_builtin_template(
        self, template_type: str, template_category: TemplateType
    ) -> str:
        """Load built-in template content.

        Args:
            template_type: Type of template (markdown, html, latex).
            template_category: Category of template.

        Returns:
            Template content as string.

        Raises:
            ValueError: If template is not found.
        """
        # Map template types to template classes
        template_mapping = {
            "markdown": {
                TemplateType.EXECUTIVE_SUMMARY: "ExecutiveSummaryTemplate",
                TemplateType.TECHNICAL_DETAILED: "TechnicalDetailedTemplate",
                TemplateType.PUBLICATION_READY: "PublicationReadyTemplate",
                TemplateType.COMPARISON_REPORT: "ComparisonReportTemplate",
                TemplateType.PERFORMANCE_ANALYSIS: (
                    "PerformanceAnalysisTemplate"
                ),
            },
            "html": {
                TemplateType.EXECUTIVE_SUMMARY: "HTMLExecutiveSummaryTemplate",
                TemplateType.PUBLICATION_READY: "HTMLPublicationTemplate",
                TemplateType.TECHNICAL_DETAILED: "HTMLTechnicalTemplate",
            },
            "latex": {
                TemplateType.EXECUTIVE_SUMMARY: (
                    "LaTeXExecutiveSummaryTemplate"
                ),
                TemplateType.PUBLICATION_READY: "LaTeXPublicationTemplate",
                TemplateType.TECHNICAL_DETAILED: "LaTeXTechnicalTemplate",
            },
        }

        if template_type not in template_mapping:
            raise ValueError(f"Unsupported template type: {template_type}")

        if template_category not in template_mapping[template_type]:
            raise ValueError(
                f"Unsupported template category: {template_category}"
            )

        # Get template class name
        template_class_name = template_mapping[template_type][
            template_category
        ]

        # Import and instantiate template
        if template_type == "markdown":
            from .markdown_templates import get_template_class
        elif template_type == "html":
            from .html_templates import get_template_class
        elif template_type == "latex":
            from .latex_templates import get_template_class
        else:
            raise ValueError(f"Unsupported template type: {template_type}")

        template_class = get_template_class(template_class_name)
        template_instance = template_class()
        return template_instance.get_template_content()

    def _render_list_value(self, value: list) -> str:
        """Render list value for template.

        Args:
            value: List to render.

        Returns:
            Rendered list as string.
        """
        if not value:
            return "None"

        # Handle list of strings
        if all(isinstance(item, str) for item in value):
            return "\n".join(f"- {item}" for item in value)

        # Handle list of numbers
        if all(isinstance(item, int | float) for item in value):
            return ", ".join(str(item) for item in value)

        # Handle mixed content
        return "\n".join(f"- {str(item)}" for item in value)

    def _render_dict_value(self, value: dict) -> str:
        """Render dict value for template.

        Args:
            value: Dictionary to render.

        Returns:
            Rendered dictionary as string.
        """
        if not value:
            return "None"

        lines = []
        for key, val in value.items():
            if isinstance(val, int | float):
                lines.append(f"- **{key}**: {val}")
            elif isinstance(val, str):
                lines.append(f"- **{key}**: {val}")
            else:
                lines.append(f"- **{key}**: {str(val)}")

        return "\n".join(lines)

    def _apply_styling(self, content: str, config: ReportConfig) -> str:
        """Apply configuration-specific styling to content.

        Args:
            content: Content to style.
            config: Report configuration.

        Returns:
            Styled content.
        """
        # Apply custom template variables
        for key, value in config.default_template_vars.items():
            placeholder = f"{{{{{key}}}}}"
            content = content.replace(placeholder, str(value))

        return content

    def validate_template(self, template_content: str) -> bool:
        """Validate template content.

        Args:
            template_content: Template content to validate.

        Returns:
            True if template is valid.
        """
        try:
            # Check for basic template structure
            if not template_content.strip():
                return False

            # Check for required sections (basic validation)
            required_sections = ["title", "content"]
            for section in required_sections:
                if f"{{{{{section}}}}}" not in template_content:
                    self.logger.warning(
                        f"Template missing required section: {section}"
                    )

            return True

        except Exception as e:
            self.logger.error(f"Template validation failed: {e}")
            return False

    def get_template_metadata(
        self, template_type: str, template_category: TemplateType
    ) -> dict[str, Any]:
        """Get metadata for a template.

        Args:
            template_type: Type of template.
            template_category: Category of template.

        Returns:
            Template metadata.
        """
        return {
            "type": template_type,
            "category": template_category.value,
            "supports_formats": self._get_supported_formats(template_type),
            "description": self._get_template_description(template_category),
        }

    def _get_supported_formats(self, template_type: str) -> list[str]:
        """Get supported formats for template type.

        Args:
            template_type: Type of template.

        Returns:
            List of supported formats.
        """
        format_mapping = {
            "markdown": ["md", "markdown"],
            "html": ["html", "htm"],
            "latex": ["tex", "latex", "pdf"],
        }
        return format_mapping.get(template_type, [])

    def _get_template_description(
        self, template_category: TemplateType
    ) -> str:
        """Get description for template category.

        Args:
            template_category: Category of template.

        Returns:
            Template description.
        """
        descriptions = {
            TemplateType.EXECUTIVE_SUMMARY: (
                "High-level summary for stakeholders"
            ),
            TemplateType.TECHNICAL_DETAILED: "Detailed technical analysis",
            TemplateType.PUBLICATION_READY: "Academic publication format",
            TemplateType.COMPARISON_REPORT: "Multi-experiment comparison",
            TemplateType.PERFORMANCE_ANALYSIS: "Focused performance analysis",
        }
        return descriptions.get(template_category, "Unknown template type")
