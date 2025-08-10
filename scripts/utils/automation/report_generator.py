"""Report generation for post-phase regeneration workflow.

This module handles the generation of comprehensive reports from
regeneration workflow results.
"""

from typing import Any


def generate_regeneration_report(results: dict[str, Any]) -> str:
    """Generate a comprehensive regeneration report.

    Args:
        results: Results from the regeneration workflow

    Returns:
        Formatted report string
    """
    report_lines = [
        "🔄 Post-Phase Regeneration Report",
        "=" * 60,
        "",
    ]

    # Overall summary
    success_rate = results["successful_phases"] / results["total_phases"] * 100
    status = (
        "✅ SUCCESS" if results["overall_success"] else "❌ PARTIAL FAILURE"
    )

    report_lines.append("📊 OVERALL SUMMARY:")
    report_lines.append(f"  - Status: {status}")
    report_lines.append(
        f"  - Duration: {results['duration_seconds']:.2f} seconds"
    )
    report_lines.append(
        f"  - Success Rate: {success_rate:.1f}% ({results['successful_phases']}/{results['total_phases']})"
    )
    report_lines.append("")

    # Phase details
    report_lines.append("📋 PHASE DETAILS:")
    for phase_name, phase_result in results["phase_results"].items():
        phase_status = "✅" if phase_result.get("success", False) else "❌"
        report_lines.append(
            f"  {phase_status} {phase_name.replace('_', ' ').title()}"
        )

        if not phase_result.get("success", False):
            error = phase_result.get("error", "Unknown error")
            report_lines.append(f"      Error: {error}")

    report_lines.append("")

    # Consistency check details
    if "consistency_checks" in results["phase_results"]:
        checks_result = results["phase_results"]["consistency_checks"]
        if checks_result.get("success", False):
            errors = checks_result.get("errors", 0)
            warnings = checks_result.get("warnings", 0)
            report_lines.append("🔍 CONSISTENCY CHECK RESULTS:")
            report_lines.append(f"  - Errors: {errors}")
            report_lines.append(f"  - Warnings: {warnings}")
            report_lines.append("")

    return "\n".join(report_lines)


def generate_phase_summary(phase_results: dict[str, Any]) -> str:
    """Generate a summary for a specific phase.

    Args:
        phase_results: Results from a specific phase

    Returns:
        Formatted summary string
    """
    if phase_results.get("success", False):
        return f"✅ {phase_results.get('output', 'Completed successfully')}"
    else:
        return f"❌ {phase_results.get('error', 'Unknown error')}"


def generate_timing_report(results: dict[str, Any]) -> str:
    """Generate a timing-focused report.

    Args:
        results: Results from the regeneration workflow

    Returns:
        Formatted timing report string
    """
    report_lines = [
        "⏱️  Regeneration Timing Report",
        "=" * 40,
        "",
        f"Total Duration: {results['duration_seconds']:.2f} seconds",
        f"Start Time: {results['start_time']}",
        f"End Time: {results['end_time']}",
        "",
    ]

    # Add per-phase timing if available
    for phase_name, phase_result in results["phase_results"].items():
        if "timestamp" in phase_result:
            report_lines.append(f"{phase_name}: {phase_result['timestamp']}")

    return "\n".join(report_lines)
