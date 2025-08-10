"""Link checker for documentation and reports.

This script checks for broken or outdated links in documentation and reports,
cross-referencing with the mapping registry to validate internal links.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root / "scripts" / "utils" / "automation"))

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


def _strip_code_blocks_markdown(content: str) -> str:
    """Remove fenced and inline code blocks from markdown to avoid false positives.

    This prevents patterns inside code like dict[str, Any] or function calls from
    being parsed as links by simple regexes.
    """
    # Remove fenced code blocks ``` ... ``` (including language tags)
    import re as _re

    content_wo_fenced = _re.sub(r"```[\s\S]*?```", "", content)
    # Remove inline code `...`
    content_wo_inline = _re.sub(r"`[^`]*`", "", content_wo_fenced)
    return content_wo_inline


def extract_links_from_markdown(content: str) -> list[tuple[str, str]]:
    """Extract links from markdown content.

    Args:
        content: Markdown content to parse

    Returns:
        List of (link_text, link_url) tuples
    """
    # Strip code blocks first to reduce false positives
    sanitized = _strip_code_blocks_markdown(content)

    # Pattern for markdown links: [text](url)
    link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    links = re.findall(link_pattern, sanitized)

    # Pattern for reference links: [text][ref] and [ref]: url
    ref_pattern = r"\[([^\]]+)\]:\s*([^\s]+)"
    ref_links = re.findall(ref_pattern, sanitized)

    return links + ref_links


def extract_links_from_html(content: str) -> list[tuple[str, str]]:
    """Extract links from HTML content.

    Args:
        content: HTML content to parse

    Returns:
        List of (link_text, link_url) tuples
    """
    # Pattern for HTML links: <a href="url">text</a>
    link_pattern = r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>'
    links = re.findall(link_pattern, content)

    # Also find href attributes without text
    href_pattern = r'href=["\']([^"\']+)["\']'
    href_links = [(url, url) for url in re.findall(href_pattern, content)]

    return links + href_links


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


def check_file_links(
    file_path: Path, registry: SimpleMappingRegistry
) -> list[dict]:
    """Check links in a single file.

    Args:
        file_path: Path to the file to check
        registry: Mapping registry for validation

    Returns:
        List of link issues found
    """
    issues = []

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Extract links based on file type
        if file_path.suffix.lower() == ".md":
            links = extract_links_from_markdown(content)
        elif file_path.suffix.lower() in [".html", ".htm"]:
            links = extract_links_from_html(content)
        else:
            return issues

        for link_text, link_url in links:
            if is_internal_link(link_url):
                # Anchor handling: allow pure anchors (#section) and file#anchor
                anchor_split = link_url.split("#", 1)
                link_path_part = anchor_split[0]
                if link_path_part == "":
                    # Pure anchor within the same file – treat as valid
                    continue

                # Check if internal link path exists (ignore anchor part)
                resolved_path = resolve_internal_link(
                    file_path, link_path_part
                )

                if not resolved_path.exists():
                    issues.append(
                        {
                            "file": str(file_path),
                            "link_text": link_text,
                            "link_url": link_url,
                            "resolved_path": str(resolved_path),
                            "issue": "broken_internal_link",
                            "severity": "error",
                        }
                    )
                else:
                    # Check if link uses old paths that should be updated
                    for mapping in registry.mappings:
                        if (
                            mapping.old_path in link_url
                            and not mapping.deprecated
                        ):
                            issues.append(
                                {
                                    "file": str(file_path),
                                    "link_text": link_text,
                                    "link_url": link_url,
                                    "suggested_url": link_url.replace(
                                        mapping.old_path, mapping.new_path
                                    ),
                                    "issue": "outdated_path",
                                    "severity": "warning",
                                    "mapping": f"{mapping.old_path} -> {mapping.new_path}",
                                }
                            )
            else:
                # External link - could add validation here if needed
                pass

        return issues

    except Exception as e:
        return [
            {
                "file": str(file_path),
                "issue": "file_read_error",
                "error": str(e),
                "severity": "error",
            }
        ]


def check_directory_links(
    directory: Path, registry: SimpleMappingRegistry
) -> list[dict]:
    """Check links in all files in a directory.

    Args:
        directory: Directory to check
        registry: Mapping registry for validation

    Returns:
        List of all link issues found
    """
    all_issues = []

    # File extensions to check
    check_extensions = {".md", ".html", ".htm", ".rst", ".txt"}

    for file_path in directory.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() in check_extensions:
            file_issues = check_file_links(file_path, registry)
            all_issues.extend(file_issues)

    return all_issues


def generate_link_report(issues: list[dict]) -> str:
    """Generate a human-readable report from link issues.

    Args:
        issues: List of link issues found

    Returns:
        Formatted report string
    """
    if not issues:
        return "✅ No link issues found!"

    report_lines = ["🔗 Link Checker Report", "=" * 50, ""]

    # Group by severity
    errors = [i for i in issues if i.get("severity") == "error"]
    warnings = [i for i in issues if i.get("severity") == "warning"]

    if errors:
        report_lines.append("❌ ERRORS:")
        for issue in errors:
            if issue.get("issue") == "broken_internal_link":
                report_lines.append(
                    f"  - {issue['file']}: Broken link '{issue['link_text']}' -> {issue['resolved_path']}"
                )
            elif issue.get("issue") == "file_read_error":
                report_lines.append(f"  - {issue['file']}: {issue['error']}")
        report_lines.append("")

    if warnings:
        report_lines.append("⚠️  WARNINGS:")
        for issue in warnings:
            if issue.get("issue") == "outdated_path":
                report_lines.append(
                    f"  - {issue['file']}: Outdated path '{issue['link_url']}' "
                    f"-> '{issue['suggested_url']}' ({issue['mapping']})"
                )
        report_lines.append("")

    # Summary
    report_lines.append(
        f"Summary: {len(errors)} errors, {len(warnings)} warnings"
    )

    return "\n".join(report_lines)


def main() -> int:
    """Main function to run the link checker.

    Returns:
        Exit code (0 for success, 1 for issues found)
    """
    parser = argparse.ArgumentParser(
        description="Check links in documentation and reports"
    )
    parser.add_argument(
        "--directories",
        nargs="+",
        default=["docs", "scripts"],
        help="Directories to check for links",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Get registry
    registry = create_default_registry()

    # Check directories
    all_issues = []

    for directory_str in args.directories:
        directory = Path(directory_str)
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            continue

        logger.info(f"Checking links in: {directory}")
        directory_issues = check_directory_links(directory, registry)
        all_issues.extend(directory_issues)
        logger.info(f"  Found {len(directory_issues)} issues")

    # Generate and print report
    report = generate_link_report(all_issues)
    try:
        print(report)
    except UnicodeEncodeError:
        # Fallback for terminals without UTF-8 support (e.g., Windows cp1252)
        print(report.encode("ascii", "ignore").decode())

    # Return appropriate exit code
    errors = [i for i in all_issues if i.get("severity") == "error"]
    warnings = [i for i in all_issues if i.get("severity") == "warning"]

    if errors:
        logger.error(f"Found {len(errors)} errors")
        return 1
    elif warnings:
        logger.warning(f"Found {len(warnings)} warnings")
        return 0  # Warnings don't fail CI
    else:
        logger.info("No issues found")
        return 0


if __name__ == "__main__":
    sys.exit(main())
