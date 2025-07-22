"""Main CLI interface for debug artifacts utilities."""

import argparse
import json
from pathlib import Path

from .artifact_diagnostics import ArtifactDiagnostics
from .artifact_fixer import ArtifactFixer


def main() -> None:
    """Main entry point for debug artifacts CLI."""

    parser = argparse.ArgumentParser(
        description="Debug and diagnostic utilities for training artifacts",
        prog="debug_artifacts",
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    # Diagnose command
    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="Run comprehensive diagnostics on experiment artifacts",
    )
    diagnose_parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to experiment directory to diagnose",
    )
    diagnose_parser.add_argument(
        "--output",
        type=Path,
        help="Save diagnostic results to JSON file",
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate a specific checkpoint file"
    )
    validate_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint file to validate",
    )

    # Fix config command
    fix_parser = subparsers.add_parser(
        "fix-config", help="Attempt to fix configuration directory issues"
    )
    fix_parser.add_argument(
        "--config-dir",
        type=Path,
        required=True,
        help="Path to configuration directory to fix",
    )

    # Cleanup command
    cleanup_parser = subparsers.add_parser(
        "cleanup", help="Clean up temporary files in experiment directory"
    )
    cleanup_parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to experiment directory to clean",
    )

    args = parser.parse_args()

    if args.command == "diagnose":
        diagnostics = ArtifactDiagnostics(args.experiment_dir)
        results = diagnostics.diagnose_all()

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"\n📄 Results saved to: {args.output}")

    elif args.command == "validate":
        ArtifactFixer.validate_checkpoint_standalone(args.checkpoint)

    elif args.command == "fix-config":
        ArtifactFixer.fix_configuration_directory(args.config_dir)

    elif args.command == "cleanup":
        ArtifactFixer.cleanup_temporary_files(args.experiment_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
