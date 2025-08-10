"""Post-Phase Regeneration Workflow for Cross-Plan Consistency.

This script automates the regeneration of documentation, reports, and other
artifacts after structural changes to the project. It ensures that all
documentation stays in sync with the current project state.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "scripts" / "utils" / "automation"))

from regeneration_phases import (  # noqa: E402
    regenerate_api_documentation,
    regenerate_project_tree,
    regenerate_test_reports,
    run_consistency_checks,
    update_mapping_registry,
)
from report_generator import generate_regeneration_report  # noqa: E402
from simple_mapping_registry import (  # noqa: E402
    SimpleMappingRegistry,
    create_default_registry,
)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


class PostPhaseRegenerator:
    """Workflow for regenerating project artifacts after structural changes."""

    def __init__(self, registry: SimpleMappingRegistry):
        """Initialize the regenerator.

        Args:
            registry: Mapping registry for path validation
        """
        self.registry = registry
        self.logger = logging.getLogger(__name__)

    def run_full_regeneration(
        self, phases: list[str] | None = None
    ) -> dict[str, Any]:
        """Run the full post-phase regeneration workflow.

        Args:
            phases: List of phases to run (None for all)

        Returns:
            Dictionary with full regeneration results
        """
        if phases is None:
            phases = ["tree", "api", "tests", "registry", "checks"]

        self.logger.info("🚀 Starting post-phase regeneration workflow...")
        self.logger.info(f"Phases to run: {', '.join(phases)}")

        start_time = datetime.now()
        results = {}

        # Run each phase using the separated modules
        if "tree" in phases:
            results["project_tree"] = regenerate_project_tree(self.registry)

        if "api" in phases:
            results["api_docs"] = regenerate_api_documentation(self.registry)

        if "tests" in phases:
            results["test_reports"] = regenerate_test_reports(self.registry)

        if "registry" in phases:
            results["mapping_registry"] = update_mapping_registry(
                self.registry
            )

        if "checks" in phases:
            results["consistency_checks"] = run_consistency_checks(
                self.registry
            )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Calculate overall success
        successful_phases = sum(
            1 for r in results.values() if r.get("success", False)
        )
        total_phases = len(results)

        overall_success = successful_phases == total_phases

        self.logger.info(
            f"✅ Regeneration workflow completed in {duration:.2f} seconds"
        )
        self.logger.info(
            f"Success rate: {successful_phases}/{total_phases} phases"
        )

        return {
            "overall_success": overall_success,
            "duration_seconds": duration,
            "successful_phases": successful_phases,
            "total_phases": total_phases,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "phase_results": results,
        }


def main() -> int:
    """Main function to run the post-phase regeneration workflow.

    Returns:
        Exit code (0 for success, 1 for errors)
    """
    parser = argparse.ArgumentParser(
        description="Run post-phase regeneration workflow"
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["tree", "api", "tests", "registry", "checks"],
        help="Specific phases to run (default: all)",
    )
    parser.add_argument(
        "--skip-phases",
        nargs="+",
        choices=["tree", "api", "tests", "registry", "checks"],
        help="Phases to skip",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Save report to file",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Determine phases to run
    all_phases = ["tree", "api", "tests", "registry", "checks"]

    if args.phases:
        phases_to_run = args.phases
    elif args.skip_phases:
        phases_to_run = [p for p in all_phases if p not in args.skip_phases]
    else:
        phases_to_run = all_phases

    # Get registry
    registry = create_default_registry()

    # Create regenerator
    regenerator = PostPhaseRegenerator(registry)

    # Run regeneration
    results = regenerator.run_full_regeneration(phases_to_run)

    # Generate and print report
    report = generate_regeneration_report(results)
    print(report)

    # Save report to file if requested
    if args.output_file:
        try:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"Report saved to {args.output_file}")
        except Exception as e:
            logger.error(f"Failed to save report to {args.output_file}: {e}")

    # Return appropriate exit code
    if results["overall_success"]:
        logger.info("✅ Post-phase regeneration completed successfully")
        return 0
    else:
        logger.error("❌ Post-phase regeneration completed with errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
