"""Automated Link Fixer for Cross-Plan Consistency.

This script automatically fixes broken and outdated links in documentation
and reports based on the mapping registry. It can be run in dry-run mode
to preview changes or in fix mode to apply changes.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "scripts" / "utils" / "automation"))

from simple_mapping_registry import (  # noqa: E402
    SimpleMappingRegistry,
    create_default_registry,
)

from scripts.utils.quality.guardrails.link_checker_utils import (  # noqa: E402
    extract_links_from_html_with_lines,
    extract_links_from_markdown_with_lines,
)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def extract_links_from_markdown(content: str) -> list[tuple[str, str, int]]:
    """Backwards-compatible wrapper (use utils implementation)."""
    return extract_links_from_markdown_with_lines(content)


def extract_links_from_html(content: str) -> list[tuple[str, str, int]]:
    """Backwards-compatible wrapper (use utils implementation)."""
    return extract_links_from_html_with_lines(content)


def is_internal_link(url: str) -> bool:
    """Check if a URL is an internal link.

    Args:
        url: URL to check

    Returns:
        True if internal, False if external
    """
    # Check if it's a relative path
    if url.startswith("./") or url.startswith("../") or url.startswith("/"):
        return True

    # Check if it's a file path without protocol
    if not url.startswith(("http://", "https://", "ftp://", "mailto:")):
        return True

    return False


def resolve_internal_link(base_path: Path, link: str) -> Path:
    """Resolve an internal link to a file path.

    Args:
        base_path: Base path for resolution
        link: Link to resolve

    Returns:
        Resolved file path
    """
    if link.startswith("/"):
        # Absolute path from project root
        return project_root / link[1:]
    elif link.startswith("./"):
        # Relative to current file
        return base_path.parent / link[2:]
    elif link.startswith("../"):
        # Relative to parent directory
        return base_path.parent / link
    else:
        # Relative to current file
        return base_path.parent / link


def find_fixable_links(
    file_path: Path, registry: SimpleMappingRegistry
) -> list[dict[str, Any]]:
    """Find links that can be automatically fixed.

    Args:
        file_path: Path to the file to check
        registry: Mapping registry for validation

    Returns:
        List of fixable link issues
    """
    fixable_links = []

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Extract links based on file type
        if file_path.suffix.lower() == ".md":
            links = extract_links_from_markdown(content)
        elif file_path.suffix.lower() in [".html", ".htm"]:
            links = extract_links_from_html(content)
        else:
            return fixable_links

        for link_text, link_url, line_num in links:
            if is_internal_link(link_url):
                # Check if internal link exists
                resolved_path = resolve_internal_link(file_path, link_url)

                if not resolved_path.exists():
                    # Check if we can fix it with a mapping
                    for mapping in registry.mappings:
                        if (
                            mapping.old_path in link_url
                            and not mapping.deprecated
                        ):
                            new_url = link_url.replace(
                                mapping.old_path, mapping.new_path
                            )
                            new_resolved_path = resolve_internal_link(
                                file_path, new_url
                            )

                            if new_resolved_path.exists():
                                fixable_links.append(
                                    {
                                        "file": str(file_path),
                                        "line": line_num,
                                        "link_text": link_text,
                                        "old_url": link_url,
                                        "new_url": new_url,
                                        "mapping": f"{mapping.old_path} -> {mapping.new_path}",
                                        "mapping_type": mapping.mapping_type,
                                    }
                                )
                                break

                else:
                    # Link exists but check if it uses outdated paths
                    for mapping in registry.mappings:
                        if (
                            mapping.old_path in link_url
                            and not mapping.deprecated
                        ):
                            new_url = link_url.replace(
                                mapping.old_path, mapping.new_path
                            )
                            fixable_links.append(
                                {
                                    "file": str(file_path),
                                    "line": line_num,
                                    "link_text": link_text,
                                    "old_url": link_url,
                                    "new_url": new_url,
                                    "mapping": f"{mapping.old_path} -> {mapping.new_path}",
                                    "mapping_type": mapping.mapping_type,
                                    "outdated": True,
                                }
                            )
                            break

        return fixable_links

    except Exception as e:
        logging.warning(f"Error processing {file_path}: {e}")
        return []


def fix_links_in_file(
    file_path: Path, fixable_links: list[dict[str, Any]], dry_run: bool = True
) -> dict[str, Any]:
    """Fix links in a single file.

    Args:
        file_path: Path to the file to fix
        fixable_links: List of links to fix
        dry_run: If True, only preview changes

    Returns:
        Dictionary with fix results
    """
    if not fixable_links:
        return {"fixed": 0, "errors": 0}

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        fixed_count = 0
        error_count = 0

        # Sort links by line number in descending order to avoid line number shifts
        sorted_links = sorted(
            fixable_links, key=lambda x: x["line"], reverse=True
        )

        for link_info in sorted_links:
            line_num = link_info["line"] - 1  # Convert to 0-based index
            old_url = link_info["old_url"]
            new_url = link_info["new_url"]

            if line_num < len(lines):
                old_line = lines[line_num]

                # Replace the URL in the line
                if file_path.suffix.lower() == ".md":
                    # Markdown link replacement
                    new_line = old_line.replace(
                        f"]({old_url})", f"]({new_url})"
                    )
                elif file_path.suffix.lower() in [".html", ".htm"]:
                    # HTML link replacement
                    new_line = old_line.replace(
                        f'href="{old_url}"', f'href="{new_url}"'
                    )
                    new_line = new_line.replace(
                        f"href='{old_url}'", f"href='{new_url}'"
                    )
                else:
                    # Generic replacement
                    new_line = old_line.replace(old_url, new_url)

                if new_line != old_line:
                    lines[line_num] = new_line
                    fixed_count += 1

                    if not dry_run:
                        logging.info(
                            f"Fixed link in {file_path}:{line_num + 1} "
                            f"'{old_url}' -> '{new_url}'"
                        )
                    else:
                        logging.info(
                            f"Would fix link in {file_path}:{line_num + 1} "
                            f"'{old_url}' -> '{new_url}'"
                        )
                else:
                    error_count += 1
                    logging.warning(
                        f"Could not fix link in {file_path}:{line_num + 1} "
                        f"'{old_url}' -> '{new_url}'"
                    )
            else:
                error_count += 1
                logging.error(f"Line {line_num + 1} not found in {file_path}")

        # Write the fixed content
        if not dry_run and fixed_count > 0:
            new_content = "\n".join(lines)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

        return {"fixed": fixed_count, "errors": error_count}

    except Exception as e:
        logging.error(f"Error fixing links in {file_path}: {e}")
        return {"fixed": 0, "errors": 1}


def scan_directory_for_fixable_links(
    directory: Path, registry: SimpleMappingRegistry
) -> list[dict[str, Any]]:
    """Scan a directory for fixable links.

    Args:
        directory: Directory to scan
        registry: Mapping registry for validation

    Returns:
        List of all fixable link issues
    """
    all_fixable_links = []

    # File extensions to check
    check_extensions = {".md", ".html", ".htm"}

    for file_path in directory.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() in check_extensions:
            file_fixable_links = find_fixable_links(file_path, registry)
            all_fixable_links.extend(file_fixable_links)

    return all_fixable_links


def generate_fix_report(
    fixable_links: list[dict[str, Any]], fix_results: dict[str, int]
) -> str:
    """Generate a report of fixable links and fix results.

    Args:
        fixable_links: List of fixable link issues
        fix_results: Results from fixing links

    Returns:
        Formatted report string
    """
    if not fixable_links:
        return "✅ No fixable links found!"

    report_lines = [
        "🔧 Auto Link Fixer Report",
        "=" * 50,
        "",
    ]

    # Group by file
    files = {}
    for link in fixable_links:
        file_path = link["file"]
        if file_path not in files:
            files[file_path] = []
        files[file_path].append(link)

    # Summary
    total_links = len(fixable_links)
    broken_links = len(
        [link for link in fixable_links if "outdated" not in link]
    )
    outdated_links = len(
        [link for link in fixable_links if "outdated" in link]
    )

    report_lines.append("📊 SUMMARY:")
    report_lines.append(f"  - Total fixable links: {total_links}")
    report_lines.append(f"  - Broken links: {broken_links}")
    report_lines.append(f"  - Outdated links: {outdated_links}")
    report_lines.append(f"  - Files affected: {len(files)}")
    report_lines.append("")

    # Fix results
    if fix_results:
        report_lines.append("🔧 FIX RESULTS:")
        report_lines.append(f"  - Links fixed: {fix_results.get('fixed', 0)}")
        report_lines.append(f"  - Errors: {fix_results.get('errors', 0)}")
        report_lines.append("")

    # Detailed breakdown by file
    report_lines.append("📁 DETAILED BREAKDOWN:")
    for file_path, file_links in files.items():
        report_lines.append(f"  {file_path}:")
        for link in file_links:
            status = "🔗" if "outdated" in link else "❌"
            report_lines.append(
                f"    {status} Line {link['line']}: '{link['old_url']}' -> '{link['new_url']}'"
            )
        report_lines.append("")

    return "\n".join(report_lines)


def main() -> int:
    """Main function to run the auto link fixer.

    Returns:
        Exit code (0 for success, 1 for errors)
    """
    parser = argparse.ArgumentParser(
        description="Automatically fix broken and outdated links"
    )
    parser.add_argument(
        "--directories",
        nargs="+",
        default=["docs", "scripts"],
        help="Directories to scan for fixable links",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Preview changes without applying them (default)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply fixes to files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Determine if this is a dry run
    dry_run = not args.fix

    if dry_run:
        logger.info("🔍 Running in DRY-RUN mode (no changes will be applied)")
    else:
        logger.info("🔧 Running in FIX mode (changes will be applied)")

    # Get registry
    registry = create_default_registry()

    # Scan directories for fixable links
    all_fixable_links = []
    for directory_str in args.directories:
        directory = Path(directory_str)
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            continue

        logger.info(f"Scanning {directory} for fixable links...")
        directory_fixable_links = scan_directory_for_fixable_links(
            directory, registry
        )
        all_fixable_links.extend(directory_fixable_links)
        logger.info(f"  Found {len(directory_fixable_links)} fixable links")

    # Group fixable links by file
    files_to_fix = {}
    for link in all_fixable_links:
        file_path = Path(link["file"])
        if file_path not in files_to_fix:
            files_to_fix[file_path] = []
        files_to_fix[file_path].append(link)

    # Fix links in each file
    total_fixed = 0
    total_errors = 0

    for file_path, file_links in files_to_fix.items():
        logger.info(f"Processing {file_path}...")
        fix_results = fix_links_in_file(file_path, file_links, dry_run)
        total_fixed += fix_results["fixed"]
        total_errors += fix_results["errors"]

    # Generate and print report
    fix_results = {"fixed": total_fixed, "errors": total_errors}
    report = generate_fix_report(all_fixable_links, fix_results)
    print(report)

    # Return appropriate exit code
    if total_errors > 0:
        logger.error(f"Encountered {total_errors} errors during fixing")
        return 1
    elif total_fixed > 0:
        if dry_run:
            logger.info(
                f"Would fix {total_fixed} links (run with --fix to apply)"
            )
        else:
            logger.info(f"Successfully fixed {total_fixed} links")
        return 0
    else:
        logger.info("No links to fix")
        return 0


if __name__ == "__main__":
    sys.exit(main())
