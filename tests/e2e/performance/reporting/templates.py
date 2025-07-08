"""HTML templates and CSS styles for performance reports.

This module contains the HTML templates and CSS styles used for generating
performance dashboards.
"""

from __future__ import annotations


def generate_css_styles() -> str:
    """Generate CSS styles for the performance dashboard."""
    return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .summary-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .risk-indicator {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .risk-low { background-color: #d4edda; color: #155724; }
        .risk-medium { background-color: #fff3cd; color: #856404; }
        .risk-high { background-color: #f8d7da; color: #721c24; }
        .risk-critical { background-color: #f5c6cb; color: #491217; }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .recommendations {
            background: #e7f3ff;
            border-left: 4px solid #0066cc;
            padding: 15px;
            margin: 15px 0;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .metadata {
            font-size: 0.9em;
            color: #666;
            border-top: 1px solid #ddd;
            padding-top: 15px;
            margin-top: 20px;
        }
        h1, h2, h3 { margin-top: 0; }
        ul { padding-left: 20px; }
        li { margin-bottom: 5px; }
    """


# HTML template for the performance dashboard
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Dashboard</title>
    <style>
        {css_styles}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Performance Dashboard</h1>
        <p>Generated: {gen_timestamp}</p>
        <p>Commit: {commit_sha}...</p>
    </div>

    <div class="summary-card">
        <h2>Executive Summary</h2>
        <div class="risk-indicator risk-{risk_level}">
            Risk Level: {risk_assessment}
        </div>
        <p>{summary_text}</p>
    </div>

    {key_findings_section}

    {recommendations_section}

    <div class="grid">
        {chart_sections}
    </div>

    {metadata_section}
</body>
</html>
"""
