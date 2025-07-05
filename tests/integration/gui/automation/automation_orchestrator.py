"""Automation orchestrator for coordinating automated workflow execution.

This module provides the main orchestration logic for executing automated
test scenarios across all workflow components (9.1-9.4) with comprehensive
coordination, reporting, and monitoring capabilities.
"""

import concurrent.futures
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from ..test_base import WorkflowTestBase
from .automation_protocols import (
    AutomatableWorkflow,
    AutomationConfiguration,
    AutomationResult,
    AutomationStrategy,
)
from .workflow_automation import WorkflowAutomationComponent


class SequentialAutomationStrategy(AutomationStrategy):
    """Sequential execution strategy for automation workflows."""

    def execute_strategy(
        self,
        workflows: Sequence[AutomatableWorkflow],
        automation_config: dict[str, Any],
    ) -> list[AutomationResult]:
        """Execute workflows sequentially with comprehensive error handling."""
        results: list[AutomationResult] = []
        config = AutomationConfiguration(**automation_config)

        for workflow in workflows:
            try:
                # Execute automated workflow using the protocol method
                workflow_result = workflow.execute_automated_workflow(
                    automation_config
                )
                results.append(workflow_result)

            except Exception as e:
                # Create error result for failed workflow
                error_result = AutomationResult(
                    workflow_name=(
                        workflow.get_workflow_name()
                        if hasattr(workflow, "get_workflow_name")
                        else "Unknown Workflow"
                    ),
                    success=False,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    execution_time_seconds=0.0,
                    test_count=0,
                    passed_count=0,
                    failed_count=1,
                    error_details=[f"Workflow execution failed: {e}"],
                    performance_metrics={},
                    artifacts_generated=[],
                    metadata={"error": str(e)},
                )
                results.append(error_result)

                if not config.continue_on_failure:
                    break

        return results

    def get_strategy_name(self) -> str:
        """Get strategy name for reporting."""
        return "Sequential Execution"

    def validate_strategy_requirements(
        self, workflows: Sequence[AutomatableWorkflow]
    ) -> bool:
        """Validate workflows are compatible with sequential execution."""
        return len(workflows) > 0 and all(
            workflow.validate_automation_preconditions()
            for workflow in workflows
        )


class ParallelAutomationStrategy(AutomationStrategy):
    """Parallel execution strategy for automation workflows."""

    def execute_strategy(
        self,
        workflows: Sequence[AutomatableWorkflow],
        automation_config: dict[str, Any],
    ) -> list[AutomationResult]:
        """Execute workflows in parallel with thread pool."""
        results: list[AutomationResult] = []
        config = AutomationConfiguration(**automation_config)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all workflow executions
            futures = []
            for workflow in workflows:
                future = executor.submit(
                    self._execute_workflow_complete,
                    workflow,
                    automation_config,
                )
                futures.append(future)

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    workflow_result = future.result(
                        timeout=config.timeout_seconds
                    )
                    results.append(workflow_result)
                except Exception as e:
                    error_result = AutomationResult(
                        workflow_name="Parallel Execution Error",
                        success=False,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        execution_time_seconds=0.0,
                        test_count=0,
                        passed_count=0,
                        failed_count=1,
                        error_details=[f"Parallel execution failed: {e}"],
                        performance_metrics={},
                        artifacts_generated=[],
                        metadata={"error": str(e)},
                    )
                    results.append(error_result)

        return results

    def _execute_workflow_complete(
        self,
        workflow: AutomatableWorkflow,
        automation_config: dict[str, Any],
    ) -> AutomationResult:
        """Execute complete workflow automation for parallel execution."""
        return workflow.execute_automated_workflow(automation_config)

    def get_strategy_name(self) -> str:
        """Get strategy name for reporting."""
        return "Parallel Execution"

    def validate_strategy_requirements(
        self, workflows: Sequence[AutomatableWorkflow]
    ) -> bool:
        """Validate workflows are compatible with parallel execution."""
        return len(workflows) > 0 and all(
            workflow.validate_automation_preconditions()
            for workflow in workflows
        )


class AutomationReporterImpl:
    """Implementation of automation reporting capabilities."""

    def generate_automation_report(
        self, results: Sequence[AutomationResult]
    ) -> Path:
        """Generate comprehensive HTML automation report."""
        output_path = Path("automation_results") / "automation_report.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html_content = self._generate_html_report(results)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_path

    def generate_summary_report(
        self, results: Sequence[AutomationResult]
    ) -> dict[str, Any]:
        """Generate summary statistics for automation results."""
        if not results:
            return {"total_workflows": 0, "success_rate": 0.0}

        total_tests = sum(result.test_count for result in results)
        total_passed = sum(result.passed_count for result in results)
        total_failed = sum(result.failed_count for result in results)
        total_time = sum(result.execution_time_seconds for result in results)

        successful_workflows = sum(1 for result in results if result.success)

        return {
            "total_workflows": len(results),
            "successful_workflows": successful_workflows,
            "workflow_success_rate": (successful_workflows / len(results))
            * 100.0,
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "test_success_rate": (
                (total_passed / total_tests) * 100.0
                if total_tests > 0
                else 0.0
            ),
            "total_execution_time": total_time,
            "average_execution_time": total_time / len(results),
            "performance_metrics": self._aggregate_performance_metrics(
                results
            ),
        }

    def export_metrics_data(
        self, results: Sequence[AutomationResult], output_path: Path
    ) -> None:
        """Export automation metrics in JSON format for external analysis."""
        import json

        metrics_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.generate_summary_report(results),
            "detailed_results": [
                {
                    "workflow_name": result.workflow_name,
                    "success": result.success,
                    "execution_time": result.execution_time_seconds,
                    "test_count": result.test_count,
                    "success_rate": result.success_rate,
                    "performance_metrics": result.performance_metrics,
                    "error_count": len(result.error_details),
                }
                for result in results
            ],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)

    def _generate_html_report(
        self, results: Sequence[AutomationResult]
    ) -> str:
        """Generate HTML report content."""
        summary = self.generate_summary_report(results)

        workflow_success_rate = summary["workflow_success_rate"]
        test_success_rate = summary["test_success_rate"]
        total_execution_time = summary["total_execution_time"]

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CrackSeg Automation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 20px;
                           border-radius: 5px; }}
                .success {{ color: green; }}
                .failure {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%;
                        margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px;
                         text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>CrackSeg Automation Execution Report</h1>

            <div class="summary">
                <h2>Summary</h2>
                <p>Total Workflows: {summary["total_workflows"]}</p>
                <p>Workflow Success Rate: {workflow_success_rate:.1f}%</p>
                <p>Total Tests: {summary["total_tests"]}</p>
                <p>Test Success Rate: {test_success_rate:.1f}%</p>
                <p>Total Execution Time: {total_execution_time:.2f}s</p>
            </div>

            <h2>Detailed Results</h2>
            <table>
                <tr>
                    <th>Workflow</th>
                    <th>Status</th>
                    <th>Tests</th>
                    <th>Success Rate</th>
                    <th>Execution Time</th>
                </tr>
        """

        for result in results:
            status_class = "success" if result.success else "failure"
            status_text = "✓ PASSED" if result.success else "✗ FAILED"

            html += f"""
                <tr>
                    <td>{result.workflow_name}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{result.passed_count}/{result.test_count}</td>
                    <td>{result.success_rate:.1f}%</td>
                    <td>{result.execution_time_seconds:.2f}s</td>
                </tr>
            """

        html += """
            </table>
        </body>
        </html>
        """

        return html

    def _aggregate_performance_metrics(
        self, results: Sequence[AutomationResult]
    ) -> dict[str, float]:
        """Aggregate performance metrics across all results."""
        all_metrics: dict[str, list[float]] = {}

        for result in results:
            for metric_name, value in result.performance_metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)

        aggregated = {}
        for metric_name, values in all_metrics.items():
            aggregated[f"{metric_name}_avg"] = sum(values) / len(values)
            aggregated[f"{metric_name}_max"] = max(values)
            aggregated[f"{metric_name}_min"] = min(values)

        return aggregated


class AutomationOrchestrator:
    """Main orchestrator for automated workflow execution."""

    def __init__(self) -> None:
        """Initialize automation orchestrator with default configurations."""
        self.strategies = {
            "sequential": SequentialAutomationStrategy(),
            "parallel": ParallelAutomationStrategy(),
        }
        self.reporter = AutomationReporterImpl()

    def execute_full_automation_suite(
        self, automation_config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute complete automation suite with all workflows.

        Args:
            automation_config: Optional automation configuration

        Returns:
            Comprehensive automation execution report
        """
        if automation_config is None:
            automation_config = {"execution_mode": "sequential"}

        config = AutomationConfiguration(**automation_config)

        # Create test utilities and workflow automation component
        test_base = WorkflowTestBase()
        test_base.setup_method()

        try:
            workflow_automation = WorkflowAutomationComponent(test_base)

            # Execute automation strategy
            strategy = self.strategies[config.execution_mode]
            results = strategy.execute_strategy(
                [workflow_automation], automation_config
            )

            # Generate reports
            summary = self.reporter.generate_summary_report(results)

            if config.generate_reports:
                report_path = self.reporter.generate_automation_report(results)
                summary["report_path"] = str(report_path)

                metrics_path = (
                    config.output_directory / "automation_metrics.json"
                )
                self.reporter.export_metrics_data(results, metrics_path)
                summary["metrics_path"] = str(metrics_path)

            return {
                "execution_successful": True,
                "strategy_used": strategy.get_strategy_name(),
                "summary": summary,
                "detailed_results": results,
                "configuration": config.__dict__,
            }

        except Exception as e:
            return {
                "execution_successful": False,
                "error": str(e),
                "strategy_used": config.execution_mode,
                "configuration": config.__dict__,
            }
        finally:
            test_base.teardown_method()

    def get_available_strategies(self) -> list[str]:
        """Get list of available automation strategies."""
        return list(self.strategies.keys())

    def validate_automation_environment(self) -> dict[str, bool]:
        """Validate that automation environment is properly configured."""
        validation_results = {
            "workflow_components_available": True,
            "test_utilities_functional": True,
            "reporting_system_ready": True,
            "output_directory_writable": True,
        }

        try:
            # Test workflow component creation
            test_base = WorkflowTestBase()
            test_base.setup_method()
            WorkflowAutomationComponent(test_base)
            test_base.teardown_method()
        except Exception:
            validation_results["workflow_components_available"] = False
            validation_results["test_utilities_functional"] = False

        try:
            # Test output directory creation
            Path("automation_results").mkdir(exist_ok=True)
        except Exception:
            validation_results["output_directory_writable"] = False

        return validation_results
