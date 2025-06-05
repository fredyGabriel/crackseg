#!/usr/bin/env python3
"""
Utility script to validate rule cross-references.

This script checks that all mdc: links in consolidated-workspace-rules.mdc
point to existing files, helping maintain the integrity of the rule system.
"""

import re
from pathlib import Path


def extract_mdc_links(file_path: Path) -> list[tuple[str, str]]:
    """Extract all mdc: links from a markdown file.

    Args:
        file_path: Path to the markdown file to scan

    Returns:
        List of tuples (link_text, file_path) for each mdc: link found
    """
    if not file_path.exists():
        return []

    content = file_path.read_text(encoding="utf-8")
    # Pattern to match [text](mdc:path/to/file.mdc)
    pattern = r"\[([^\]]+)\]\(mdc:([^)]+)\)"
    matches = re.findall(pattern, content)
    return matches


def validate_rule_references(workspace_root: Path) -> dict[str, list[str]]:
    """Validate all rule references in the consolidated workspace rules.

    Args:
        workspace_root: Root directory of the workspace

    Returns:
        Dictionary with 'valid' and 'broken' keys containing lists of refs
    """
    consolidated_rules = (
        workspace_root
        / ".cursor"
        / "rules"
        / "consolidated-workspace-rules.mdc"
    )

    if not consolidated_rules.exists():
        print(f"❌ Consolidated rules file not found: {consolidated_rules}")
        return {
            "valid": [],
            "broken": ["consolidated-workspace-rules.mdc not found"],
        }

    links = extract_mdc_links(consolidated_rules)
    valid_links = []
    broken_links = []

    for link_text, file_path in links:
        # Convert mdc: path to actual file path
        actual_path = workspace_root / file_path.lstrip("./")

        if actual_path.exists():
            valid_links.append(f"✅ {link_text} → {file_path}")
        else:
            broken_links.append(f"❌ {link_text} → {file_path} (NOT FOUND)")

    return {"valid": valid_links, "broken": broken_links}


def main() -> int:
    """Main function to run the validation."""
    workspace_root = Path(__file__).parent.parent.parent
    print(f"🔍 Validating rule references from: {workspace_root}")
    print("📋 Checking consolidated workspace rules...")

    results = validate_rule_references(workspace_root)

    print("\n📊 Results:")
    print(f"✅ Valid references: {len(results['valid'])}")
    print(f"❌ Broken references: {len(results['broken'])}")

    if results["valid"]:
        print("\n✅ Valid References:")
        for ref in results["valid"]:
            print(f"  {ref}")

    if results["broken"]:
        print("\n❌ Broken References:")
        for ref in results["broken"]:
            print(f"  {ref}")
        return 1  # Exit with error code

    print("\n🎉 All rule references are valid!")
    return 0


if __name__ == "__main__":
    exit(main())
