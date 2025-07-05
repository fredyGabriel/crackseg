"""CI/CD integration component for automated workflow execution.

This module provides integration capabilities with GitHub Actions and other
CI/CD systems to enable automated test execution in continuous integration
pipelines.
"""

import json
import os
from pathlib import Path
from typing import Any, TypedDict

from .automation_orchestrator import AutomationOrchestrator


class CIMetricsDict(TypedDict):
    """Type definition for CI metrics dictionary."""

    total_workflows: float
    workflow_success_rate: float
    total_execution_time: float
    average_execution_time: float


class CIThresholdsDict(TypedDict):
    """Type definition for CI thresholds dictionary."""

    min_success_rate: float
    max_execution_time: float
    performance_regression_threshold: float


class CIAlertDict(TypedDict):
    """Type definition for CI alert dictionary."""

    type: str
    message: str


class CIIntegrationAutomator:
    """Component for integrating automation with CI/CD systems."""

    def __init__(self) -> None:
        """Initialize CI integration component."""
        self.orchestrator = AutomationOrchestrator()
        self.ci_environment = self._detect_ci_environment()

    def _detect_ci_environment(self) -> dict[str, Any]:
        """Detect current CI environment and extract relevant information."""
        ci_info = {
            "is_ci": False,
            "system": "unknown",
            "branch": None,
            "commit_sha": None,
            "pr_number": None,
            "workflow_name": None,
            "environment_variables": {},
        }

        # Check for GitHub Actions
        if os.getenv("GITHUB_ACTIONS") == "true":
            ci_info.update(
                {
                    "is_ci": True,
                    "system": "github_actions",
                    "branch": os.getenv("GITHUB_REF_NAME"),
                    "commit_sha": os.getenv("GITHUB_SHA"),
                    "pr_number": os.getenv("GITHUB_PR_NUMBER"),
                    "workflow_name": os.getenv("GITHUB_WORKFLOW"),
                    "environment_variables": {
                        "GITHUB_REPOSITORY": os.getenv("GITHUB_REPOSITORY"),
                        "GITHUB_ACTOR": os.getenv("GITHUB_ACTOR"),
                        "GITHUB_RUN_ID": os.getenv("GITHUB_RUN_ID"),
                        "GITHUB_RUN_NUMBER": os.getenv("GITHUB_RUN_NUMBER"),
                    },
                }
            )

        # Check for other CI systems (Jenkins, GitLab CI, etc.)
        elif os.getenv("JENKINS_URL"):
            ci_info.update(
                {
                    "is_ci": True,
                    "system": "jenkins",
                    "branch": os.getenv("GIT_BRANCH"),
                    "commit_sha": os.getenv("GIT_COMMIT"),
                }
            )

        elif os.getenv("GITLAB_CI") == "true":
            ci_info.update(
                {
                    "is_ci": True,
                    "system": "gitlab_ci",
                    "branch": os.getenv("CI_COMMIT_REF_NAME"),
                    "commit_sha": os.getenv("CI_COMMIT_SHA"),
                }
            )

        return ci_info

    def execute_ci_automation_workflow(
        self,
        trigger_event: str = "push",
        config_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute automation workflow optimized for CI environment.

        Args:
            trigger_event: CI trigger event (push, pull_request, schedule)
            config_override: Optional configuration overrides for CI

        Returns:
            CI-optimized automation execution results
        """
        # Default CI configuration
        ci_config = {
            "execution_mode": (
                "parallel" if self.ci_environment["is_ci"] else "sequential"
            ),
            "timeout_seconds": 600,  # 10 minutes for CI
            "continue_on_failure": True,
            "generate_reports": True,
            "capture_artifacts": True,
            "performance_monitoring": True,
            "output_directory": Path("ci_automation_results"),
        }

        # Apply configuration overrides
        if config_override:
            ci_config.update(config_override)

        # Adjust configuration based on trigger event
        if trigger_event == "pull_request":
            # Faster execution for PR validation
            ci_config.update(
                {
                    "timeout_seconds": 300,
                    "execution_mode": "sequential",
                }
            )
        elif trigger_event == "schedule":
            # Comprehensive testing for scheduled runs
            ci_config.update(
                {
                    "timeout_seconds": 1200,  # 20 minutes
                    "execution_mode": "parallel",
                }
            )

        # Execute automation with CI-specific configuration
        result = self.orchestrator.execute_full_automation_suite(ci_config)

        # Add CI-specific metadata
        result["ci_metadata"] = {
            "trigger_event": trigger_event,
            "ci_environment": self.ci_environment,
            "ci_optimized": True,
        }

        # Generate CI-specific artifacts
        if result["execution_successful"] and ci_config["generate_reports"]:
            output_dir = ci_config["output_directory"]
            if not isinstance(output_dir, Path):
                output_dir = Path(str(output_dir))
            self._generate_ci_artifacts(result, output_dir)

        return result

    def _generate_ci_artifacts(
        self, automation_result: dict[str, Any], output_dir: Path
    ) -> None:
        """Generate CI-specific artifacts for integration with CI systems."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate JUnit XML for test result integration
        self._generate_junit_xml(
            automation_result, output_dir / "junit_results.xml"
        )

        # Generate GitHub Actions step summary (if in GitHub Actions)
        if self.ci_environment["system"] == "github_actions":
            self._generate_github_actions_summary(
                automation_result, output_dir / "github_summary.md"
            )

        # Generate CI metrics file
        self._generate_ci_metrics(
            automation_result, output_dir / "ci_metrics.json"
        )

    def _generate_junit_xml(
        self, automation_result: dict[str, Any], output_path: Path
    ) -> None:
        """Generate JUnit XML format for CI test result integration."""
        if not automation_result["execution_successful"]:
            return

        summary = automation_result["summary"]
        detailed_results = automation_result["detailed_results"]

        xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
        xml_content += '<testsuite name="CrackSeg Automation" '
        xml_content += f'tests="{summary["total_tests"]}" '
        xml_content += f'failures="{summary["total_failed"]}" '
        xml_content += f'time="{summary["total_execution_time"]:.2f}">\n'

        for result in detailed_results:
            xml_content += f'  <testcase name="{result.workflow_name}" '
            xml_content += 'classname="automation" '
            xml_content += f'time="{result.execution_time_seconds:.2f}">\n'

            if not result.success:
                xml_content += (
                    '    <failure message="Workflow execution failed">\n'
                )
                for error in result.error_details:
                    xml_content += f"      {error}\n"
                xml_content += "    </failure>\n"

            xml_content += "  </testcase>\n"

        xml_content += "</testsuite>\n"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_content)

    def _generate_github_actions_summary(
        self, automation_result: dict[str, Any], output_path: Path
    ) -> None:
        """Generate GitHub Actions step summary."""
        if not automation_result["execution_successful"]:
            return

        summary = automation_result["summary"]

        markdown_content = "# 🤖 CrackSeg Automation Execution Summary\n\n"

        # Success indicator
        success_icon = "✅" if summary["workflow_success_rate"] > 80 else "⚠️"
        markdown_content += f"{success_icon} **Overall Status**: "

        if summary["workflow_success_rate"] == 100:
            markdown_content += "All workflows passed\n\n"
        else:
            markdown_content += (
                f"{summary['workflow_success_rate']:.1f}% workflows passed\n\n"
            )

        # Summary statistics
        markdown_content += "## 📊 Summary Statistics\n\n"
        markdown_content += (
            f"- **Total Workflows**: {summary['total_workflows']}\n"
        )
        markdown_content += f"- **Total Tests**: {summary['total_tests']}\n"
        markdown_content += (
            f"- **Success Rate**: {summary['test_success_rate']:.1f}%\n"
        )
        markdown_content += (
            f"- **Execution Time**: {summary['total_execution_time']:.2f}s\n\n"
        )

        # Detailed results table
        markdown_content += "## 📋 Detailed Results\n\n"
        markdown_content += (
            "| Workflow | Status | Tests | Success Rate | Time |\n"
        )
        markdown_content += (
            "|----------|--------|-------|--------------|------|\n"
        )

        for result in automation_result["detailed_results"]:
            status_icon = "✅" if result.success else "❌"
            markdown_content += f"| {result.workflow_name} | {status_icon} | "
            markdown_content += f"{result.passed_count}/{result.test_count} | "
            markdown_content += f"{result.success_rate:.1f}% | "
            markdown_content += f"{result.execution_time_seconds:.2f}s |\n"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        # Set GitHub Actions step summary (if environment supports it)
        github_step_summary = os.getenv("GITHUB_STEP_SUMMARY")
        if github_step_summary:
            with open(github_step_summary, "a", encoding="utf-8") as f:
                f.write(markdown_content)

    def _generate_ci_metrics(
        self, automation_result: dict[str, Any], output_path: Path
    ) -> None:
        """Generate CI-specific metrics for monitoring and alerting."""
        summary = automation_result.get("summary", {})

        # Create typed metrics
        metrics: CIMetricsDict = {
            "total_workflows": float(summary.get("total_workflows", 0)),
            "workflow_success_rate": float(
                summary.get("workflow_success_rate", 0.0)
            ),
            "total_execution_time": float(
                summary.get("total_execution_time", 0.0)
            ),
            "average_execution_time": float(
                summary.get("average_execution_time", 0.0)
            ),
        }

        # Create typed thresholds
        thresholds: CIThresholdsDict = {
            "min_success_rate": 80.0,
            "max_execution_time": 600.0,
            "performance_regression_threshold": 20.0,
        }

        # Create alerts list
        alerts: list[CIAlertDict] = []

        # Check for threshold violations
        if metrics["workflow_success_rate"] < thresholds["min_success_rate"]:
            alerts.append(
                {
                    "type": "success_rate_violation",
                    "message": (
                        f"Success rate "
                        f"{metrics['workflow_success_rate']:.1f}% "
                        f"below threshold {thresholds['min_success_rate']}%"
                    ),
                }
            )

        if metrics["total_execution_time"] > thresholds["max_execution_time"]:
            alerts.append(
                {
                    "type": "execution_time_violation",
                    "message": (
                        f"Execution time "
                        f"{metrics['total_execution_time']:.1f}s "
                        f"exceeds threshold "
                        f"{thresholds['max_execution_time']}s"
                    ),
                }
            )

        # Compile final metrics
        ci_metrics = {
            "timestamp": summary.get("timestamp"),
            "ci_environment": self.ci_environment,
            "execution_successful": automation_result.get(
                "execution_successful", False
            ),
            "metrics": metrics,
            "thresholds": thresholds,
            "alerts": alerts,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ci_metrics, f, indent=2)

    def generate_ci_configuration_template(
        self, ci_system: str = "github_actions"
    ) -> str:
        """Generate CI configuration template for automation integration.

        Args:
            ci_system: Target CI system (github_actions, jenkins, gitlab_ci)

        Returns:
            CI configuration template as string
        """
        if ci_system == "github_actions":
            return self._generate_github_actions_template()
        elif ci_system == "jenkins":
            return self._generate_jenkins_template()
        elif ci_system == "gitlab_ci":
            return self._generate_gitlab_ci_template()
        else:
            raise ValueError(f"Unsupported CI system: {ci_system}")

    def _generate_github_actions_template(self) -> str:
        """Generate GitHub Actions workflow template."""
        return """
name: CrackSeg Automation Workflow

on:
  push:
    branches: [ main, develop ]
    paths:
      - 'src/**'
      - 'scripts/gui/**'
      - 'tests/integration/gui/**'
  pull_request:
    branches: [ main, develop ]
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM UTC
  workflow_dispatch:
    inputs:
      execution_mode:
        description: 'Execution mode'
        required: false
        default: 'parallel'
        type: choice
        options:
          - sequential
          - parallel

jobs:
  automation:
    runs-on: ubuntu-latest
    timeout-minutes: 20

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install Dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .

      - name: Execute Automation Suite
        run: |
          python -m pytest \\
            tests/integration/gui/automation/test_automation_execution.py \\
            -v --tb=short

      - name: Upload Automation Reports
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: automation-reports
          path: |
            ci_automation_results/
            automation_results/
          retention-days: 30

      - name: Upload Test Results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-results
          path: ci_automation_results/junit_results.xml
          retention-days: 30
        """

    def _generate_jenkins_template(self) -> str:
        """Generate Jenkins pipeline template."""
        return """
pipeline {
    agent any

    options {
        timeout(time: 20, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '50'))
    }

    triggers {
        cron('0 6 * * *')  // Daily at 6 AM
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup Environment') {
            steps {
                sh '''
                    python -m venv automation_env
                    source automation_env/bin/activate
                    pip install -r requirements.txt
                    pip install -e .
                '''
            }
        }

        stage('Execute Automation') {
            steps {
                sh '''
                    source automation_env/bin/activate
                    python -m pytest \\
                      tests/integration/gui/automation/\\
                      test_automation_execution.py \\
                      -v --tb=short
                '''
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'ci_automation_results/**/*', \\
              allowEmptyArchive: true
            publishTestResults \\
              testResultsPattern: 'ci_automation_results/junit_results.xml'
        }
    }
}
        """

    def _generate_gitlab_ci_template(self) -> str:
        """Generate GitLab CI template."""
        return """
stages:
  - automation

automation:
  stage: automation
  image: python:3.12
  timeout: 20 minutes

  before_script:
    - pip install -r requirements.txt
    - pip install -e .

  script:
    - python -m pytest
      tests/integration/gui/automation/test_automation_execution.py
      -v --tb=short

  artifacts:
    when: always
    expire_in: 30 days
    paths:
      - ci_automation_results/
      - automation_results/
    reports:
      junit: ci_automation_results/junit_results.xml

  rules:
    - if: $CI_PIPELINE_SOURCE == "push"
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_PIPELINE_SOURCE == "schedule"
    - if: $CI_PIPELINE_SOURCE == "web"
        """
