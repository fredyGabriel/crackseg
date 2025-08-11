"""Common HTML helpers for report templates.

Provides default CSS and a wrapper to build full HTML documents.
"""

from __future__ import annotations


def default_css() -> str:
    """Return default CSS styles wrapped in a <style> tag."""
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
                body { padding: 10px; }
                .container { padding: 20px; }
                .experiment-info { grid-template-columns: 1fr; }
            }
        </style>
    """


def html_wrap(title: str, css: str, body: str, lang: str = "en") -> str:
    """Wrap provided body and css into a full HTML document string."""
    return f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {css}
</head>
<body>
{body}
</body>
</html>"""
