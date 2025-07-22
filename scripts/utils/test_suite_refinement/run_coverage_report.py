"""
Coverage report generator for test suite refinement. This script runs
pytest with coverage reporting and moves the generated reports to the
appropriate directory for test suite evaluation.
"""

import os
import shutil
import subprocess

# Type definitions
REPORTS_DIR = "outputs/prd_project_refinement/test_suite_evaluation/reports/"
HTMLCOV_SRC = "htmlcov"
HTMLCOV_DST = os.path.join(REPORTS_DIR, "htmlcov")
COVERAGE_XML = "coverage.xml"
COVERAGE_XML_DST = os.path.join(REPORTS_DIR, "coverage.xml")

# 1. Ejecutar pytest con cobertura
subprocess.run(
    [
        "pytest",
        "--cov=src",
        "--cov-report=xml",
        "--cov-report=html",
        "--cov-report=term",
        "--maxfail=1000",
        "--disable-warnings",
        "--tb=short",
    ],
    check=True,
)

# 2. Mover coverage.xml
if os.path.exists(COVERAGE_XML):
    shutil.move(COVERAGE_XML, COVERAGE_XML_DST)

# 3. Copiar htmlcov/
if os.path.exists(HTMLCOV_SRC):
    if os.path.exists(HTMLCOV_DST):
        shutil.rmtree(HTMLCOV_DST)
    shutil.copytree(HTMLCOV_SRC, HTMLCOV_DST)

print(f"Coverage reports generated in: {REPORTS_DIR}")
