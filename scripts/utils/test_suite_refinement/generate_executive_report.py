import csv
import os
import xml.etree.ElementTree as ET

from jinja2 import Template

REPORTS_DIR = "outputs/prd_project_refinement/test_suite_evaluation/reports/"
COVERAGE_XML = os.path.join(REPORTS_DIR, "coverage.xml")
TEST_INVENTORY = os.path.join(REPORTS_DIR, "test_inventory.csv")
SLOW_TESTS = os.path.join(REPORTS_DIR, "slow_tests.txt")
EXEC_REPORT = os.path.join(REPORTS_DIR, "executive_report.md")


# 1. Parse coverage.xml
def parse_coverage_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    line_rate = float(root.attrib.get("line-rate", "0"))
    branch_rate = float(root.attrib.get("branch-rate", "0"))
    return line_rate * 100, branch_rate * 100


# 2. Parse test_inventory.csv
def parse_test_inventory(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        header = reader[1]
        data = reader[2:]
    idx_status = header.index("Status")
    stats = {"passing": 0, "failing": 0, "flaky": 0, "other": 0}
    for row in data:
        status = row[idx_status].strip().lower()
        if status in stats:
            stats[status] += 1
        else:
            stats["other"] += 1
    return stats


# 3. Parse slow_tests.txt
def parse_slow_tests(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]
    return lines[:5]


# 4. Render Markdown report
def render_report(line_cov, branch_cov, stats, slow_tests):
    template = Template(
        """
# Executive Test & Coverage Report

**Coverage Summary:**
- Line coverage: {{ line_cov|round(1) }}%
- Branch coverage: {{ branch_cov|round(1) }}%

**Test Results:**
- Passing: {{ stats.passing }}
- Failing: {{ stats.failing }}
- Flaky: {{ stats.flaky }}
- Other/Unknown: {{ stats.other }}

**Top 5 Slowest Tests:**
{% for test in slow_tests %}- {{ test }}
{% endfor %}

{% if line_cov < 80 %}
> ⚠️ **Recommendation:** Coverage is below 80%. Increase test coverage,
especially for critical modules.
{% endif %}

---
*Report generated automatically.*
"""
    )
    return template.render(
        line_cov=line_cov,
        branch_cov=branch_cov,
        stats=stats,
        slow_tests=slow_tests,
    )


if __name__ == "__main__":
    line_cov, branch_cov = parse_coverage_xml(COVERAGE_XML)
    stats = parse_test_inventory(TEST_INVENTORY)
    slow_tests = parse_slow_tests(SLOW_TESTS)
    report = render_report(line_cov, branch_cov, stats, slow_tests)
    with open(EXEC_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Executive report generated at: {EXEC_REPORT}")
