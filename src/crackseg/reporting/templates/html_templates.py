"""HTML templates for experiment reports.

This module provides professional HTML templates for generating
interactive experiment reports with modern styling and features.
"""

from abc import ABC, abstractmethod


class BaseHTMLTemplate(ABC):
    """Base class for HTML templates."""

    @abstractmethod
    def get_template_content(self) -> str:
        """Get template content."""
        pass

    def get_css_styles(self) -> str:
        """Get CSS styles for the template."""
        return """
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f8f9fa;
            }

            .container {
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }

            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 30px;
            }

            h2 {
                color: #34495e;
                margin-top: 30px;
                margin-bottom: 15px;
                border-left: 4px solid #3498db;
                padding-left: 15px;
            }

            h3 {
                color: #2c3e50;
                margin-top: 25px;
                margin-bottom: 10px;
            }

            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                margin: 15px 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }

            .metric-value {
                font-size: 2em;
                font-weight: bold;
                margin-bottom: 5px;
            }

            .metric-label {
                font-size: 0.9em;
                opacity: 0.9;
            }

            .summary-box {
                background: #ecf0f1;
                border-left: 4px solid #3498db;
                padding: 20px;
                margin: 20px 0;
                border-radius: 4px;
            }

            .recommendation {
                background: #e8f5e8;
                border-left: 4px solid #27ae60;
                padding: 15px;
                margin: 10px 0;
                border-radius: 4px;
            }

            .warning {
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 15px;
                margin: 10px 0;
                border-radius: 4px;
            }

            .error {
                background: #f8d7da;
                border-left: 4px solid #dc3545;
                padding: 15px;
                margin: 10px 0;
                border-radius: 4px;
            }

            .experiment-info {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }

            .info-item {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 6px;
                border: 1px solid #dee2e6;
            }

            .info-label {
                font-weight: bold;
                color: #495057;
                margin-bottom: 5px;
            }

            .info-value {
                color: #6c757d;
            }

            .chart-container {
                margin: 20px 0;
                padding: 20px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }

            .footer {
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #dee2e6;
                text-align: center;
                color: #6c757d;
                font-size: 0.9em;
            }

            .badge {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: bold;
                text-transform: uppercase;
            }

            .badge-success {
                background: #d4edda;
                color: #155724;
            }

            .badge-warning {
                background: #fff3cd;
                color: #856404;
            }

            .badge-danger {
                background: #f8d7da;
                color: #721c24;
            }

            .badge-info {
                background: #d1ecf1;
                color: #0c5460;
            }

            @media (max-width: 768px) {
                body {
                    padding: 10px;
                }

                .container {
                    padding: 20px;
                }

                .experiment-info {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """


class HTMLExecutiveSummaryTemplate(BaseHTMLTemplate):
    """Executive summary HTML template."""

    def get_template_content(self) -> str:
        """Get executive summary HTML template content."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{title}}</title>
    {self.get_css_styles()}
</head>
<body>
    <div class="container">
        <h1>{{title}}</h1>
        <p><em>Experiment Report - Executive Summary</em><br>
        <small>Generated on {{generation_timestamp}}</small></p>

        <div class="summary-box">
            <h2>📊 Executive Summary</h2>
            <p>{{executive_summary}}</p>
        </div>

        <h2>🎯 Key Findings</h2>
        <div class="experiment-info">
            {{key_findings}}
        </div>

        <h2>📈 Performance Overview</h2>
        <div class="metric-card">
            {{performance_metrics}}
        </div>

        <h2>🚀 Recommendations</h2>
        <div class="recommendation">
            {{recommendations}}
        </div>

        <h2>📋 Experiment Details</h2>
        <div class="experiment-info">
            <div class="info-item">
                <div class="info-label">Experiment ID</div>
                <div class="info-value">{{experiment_id}}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Experiment Name</div>
                <div class="info-value">{{experiment_name}}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Status</div>
                <div class="info-value">
                    <span class="badge badge-{{status_badge}}">
                        {{status}}
                    </span>
                </div>
            </div>
            <div class="info-item">
                <div class="info-label">Duration</div>
                <div class="info-value">{{duration}}</div>
            </div>
        </div>

        <div class="footer">
            <p>Report generated by CrackSeg Experimental Reporting System</p>
        </div>
    </div>
</body>
</html>"""


class HTMLPublicationTemplate(BaseHTMLTemplate):
    """Publication-ready HTML template."""

    def get_template_content(self) -> str:
        """Get publication-ready HTML template content."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{title}}</title>
    {self.get_css_styles()}
</head>
<body>
    <div class="container">
        <h1>{{title}}</h1>
        <p><em>Crack Segmentation Experiment Report</em><br>
        <small>{{generation_timestamp}}</small></p>

        <h2>Abstract</h2>
        <div class="summary-box">
            <p>{{abstract}}</p>
        </div>

        <h2>1. Introduction</h2>
        <p>{{introduction}}</p>

        <h2>2. Methodology</h2>

        <h3>2.1 Model Architecture</h3>
        <div class="experiment-info">
            {{model_architecture}}
        </div>

        <h3>2.2 Training Configuration</h3>
        <div class="experiment-info">
            {{training_config}}
        </div>

        <h3>2.3 Dataset</h3>
        <p>{{dataset_description}}</p>

        <h2>3. Results</h2>

        <h3>3.1 Performance Metrics</h3>
        <div class="metric-card">
            {{performance_metrics}}
        </div>

        <h3>3.2 Training Analysis</h3>
        <div class="chart-container">
            {{training_curves}}
        </div>

        <h3>3.3 Comparative Analysis</h3>
        <div class="chart-container">
            {{comparison_results}}
        </div>

        <h2>4. Discussion</h2>
        <p>{{discussion}}</p>

        <h2>5. Conclusions</h2>
        <p>{{conclusions}}</p>

        <h2>6. Future Work</h2>
        <p>{{future_work}}</p>

        <h2>References</h2>
        <div class="summary-box">
            {{references}}
        </div>

        <h2>Appendix</h2>

        <h3>A. Experimental Setup</h3>
        <div class="experiment-info">
            {{experimental_setup}}
        </div>

        <h3>B. Detailed Results</h3>
        <div class="chart-container">
            {{detailed_results}}
        </div>

        <h3>C. Figures</h3>
        <div class="chart-container">
            {{publication_figures}}
        </div>

        <div class="footer">
            <p>Generated by CrackSeg Experimental Reporting System</p>
        </div>
    </div>
</body>
</html>"""


class HTMLTechnicalTemplate(BaseHTMLTemplate):
    """Technical detailed HTML template."""

    def get_template_content(self) -> str:
        """Get technical detailed HTML template content."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{title}}</title>
    {self.get_css_styles()}
</head>
<body>
    <div class="container">
        <h1>{{title}}</h1>
        <p><em>Technical Experiment Report</em><br>
        <small>Generated on {{generation_timestamp}}</small></p>

        <h2>📋 Experiment Information</h2>
        <div class="experiment-info">
            <div class="info-item">
                <div class="info-label">Experiment ID</div>
                <div class="info-value">{{experiment_id}}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Experiment Name</div>
                <div class="info-value">{{experiment_name}}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Configuration</div>
                <div class="info-value">{{config_summary}}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Duration</div>
                <div class="info-value">{{duration}}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Status</div>
                <div class="info-value">
                    <span class="badge badge-{{status_badge}}">
                        {{status}}
                    </span>
                </div>
            </div>
        </div>

        <h2>🎯 Executive Summary</h2>
        <div class="summary-box">
            <p>{{executive_summary}}</p>
        </div>

        <h2>📊 Performance Analysis</h2>

        <h3>Key Metrics</h3>
        <div class="metric-card">
            {{performance_metrics}}
        </div>

        <h3>Training Curves</h3>
        <div class="chart-container">
            {{training_curves}}
        </div>

        <h3>Detailed Analysis</h3>
        <div class="summary-box">
            {{technical_details}}
        </div>

        <h2>🔍 Technical Details</h2>

        <h3>Model Architecture</h3>
        <div class="experiment-info">
            {{model_architecture}}
        </div>

        <h3>Training Configuration</h3>
        <div class="experiment-info">
            {{training_config}}
        </div>

        <h3>Data Configuration</h3>
        <div class="experiment-info">
            {{data_config}}
        </div>

        <h2>📈 Results Analysis</h2>
        <div class="summary-box">
            {{results_analysis}}
        </div>

        <h2>🚀 Recommendations</h2>
        <div class="recommendation">
            {{recommendations}}
        </div>

        <h2>📁 Artifacts</h2>
        <div class="experiment-info">
            {{artifacts}}
        </div>

        <h2>📄 Metadata</h2>
        <div class="summary-box">
            {{metadata}}
        </div>

        <div class="footer">
            <p>Generated by CrackSeg Experimental Reporting System</p>
        </div>
    </div>
</body>
</html>"""


def get_template_class(template_name: str) -> type[BaseHTMLTemplate]:
    """Get template class by name.

    Args:
        template_name: Name of the template class.

    Returns:
        Template class.

    Raises:
        ValueError: If template class is not found.
    """
    template_classes = {
        "HTMLExecutiveSummaryTemplate": HTMLExecutiveSummaryTemplate,
        "HTMLPublicationTemplate": HTMLPublicationTemplate,
        "HTMLTechnicalTemplate": HTMLTechnicalTemplate,
    }

    if template_name not in template_classes:
        raise ValueError(f"Template class not found: {template_name}")

    return template_classes[template_name]
