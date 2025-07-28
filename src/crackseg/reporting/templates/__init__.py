"""Configurable output templates for experiment reports.

This module provides a comprehensive template system for generating
professional reports in multiple formats (Markdown, HTML, LaTeX, PDF).
Supports customizable styling and content organization.
"""

from .html_templates import (
    HTMLExecutiveSummaryTemplate,
    HTMLPublicationTemplate,
    HTMLTechnicalTemplate,
)
from .latex_templates import (
    LaTeXExecutiveSummaryTemplate,
    LaTeXPublicationTemplate,
    LaTeXTechnicalTemplate,
)
from .markdown_templates import (
    ComparisonReportTemplate,
    ExecutiveSummaryTemplate,
    PerformanceAnalysisTemplate,
    PublicationReadyTemplate,
    TechnicalDetailedTemplate,
)
from .template_manager import TemplateManager

__all__ = [
    "TemplateManager",
    # Markdown templates
    "ExecutiveSummaryTemplate",
    "TechnicalDetailedTemplate",
    "PublicationReadyTemplate",
    "ComparisonReportTemplate",
    "PerformanceAnalysisTemplate",
    # LaTeX templates
    "LaTeXExecutiveSummaryTemplate",
    "LaTeXPublicationTemplate",
    "LaTeXTechnicalTemplate",
    # HTML templates
    "HTMLExecutiveSummaryTemplate",
    "HTMLPublicationTemplate",
    "HTMLTechnicalTemplate",
]
