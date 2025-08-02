#!/usr/bin/env python3
"""Catalog documentation files by type for import statement processing.

This script identifies and categorizes all documentation files in the project
for targeted processing of 'from src.' import statements.
"""

import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentationCatalog:
    """Catalog documentation files by type for targeted processing."""

    def __init__(self, project_root: Path) -> None:
        """Initialize the documentation catalog.

        Args:
            project_root: Root directory of the project.
        """
        self.project_root = project_root
        self.docs_root = project_root / "docs"
        self.catalog: dict[str, list[Path]] = {
            "README": [],
            "guide": [],
            "tutorial": [],
            "report": [],
            "api": [],
            "other": [],
        }

    def _is_readme_file(self, file_path: Path) -> bool:
        """Check if file is a README file.

        Args:
            file_path: Path to the file.

        Returns:
            True if file is a README, False otherwise.
        """
        return file_path.name.upper() in ["README.MD", "README.TXT", "README"]

    def _is_guide_file(self, file_path: Path) -> bool:
        """Check if file is a guide file.

        Args:
            file_path: Path to the file.

        Returns:
            True if file is a guide, False otherwise.
        """
        # Files in guides directory or with guide in name
        return (
            "guides" in file_path.parts
            or "guide" in file_path.name.lower()
            or file_path.suffix == ".md"
            and "guide" in file_path.stem.lower()
        )

    def _is_tutorial_file(self, file_path: Path) -> bool:
        """Check if file is a tutorial file.

        Args:
            file_path: Path to the file.

        Returns:
            True if file is a tutorial, False otherwise.
        """
        # Files in tutorials directory or with tutorial in name
        return (
            "tutorials" in file_path.parts
            or "tutorial" in file_path.name.lower()
            or file_path.suffix == ".md"
            and "tutorial" in file_path.stem.lower()
        )

    def _is_report_file(self, file_path: Path) -> bool:
        """Check if file is a report file.

        Args:
            file_path: Path to the file.

        Returns:
            True if file is a report, False otherwise.
        """
        # Files in reports directory or with report in name
        return (
            "reports" in file_path.parts
            or "report" in file_path.name.lower()
            or file_path.suffix == ".md"
            and "report" in file_path.stem.lower()
        )

    def _is_api_file(self, file_path: Path) -> bool:
        """Check if file is an API documentation file.

        Args:
            file_path: Path to the file.

        Returns:
            True if file is API documentation, False otherwise.
        """
        # Files in api directory or with api in name
        return (
            "api" in file_path.parts
            or "api" in file_path.name.lower()
            or file_path.suffix == ".md"
            and "api" in file_path.stem.lower()
        )

    def _categorize_file(self, file_path: Path) -> str:
        """Categorize a file by its type.

        Args:
            file_path: Path to the file.

        Returns:
            Category of the file.
        """
        if self._is_readme_file(file_path):
            return "README"
        elif self._is_guide_file(file_path):
            return "guide"
        elif self._is_tutorial_file(file_path):
            return "tutorial"
        elif self._is_report_file(file_path):
            return "report"
        elif self._is_api_file(file_path):
            return "api"
        else:
            return "other"

    def scan_documentation_files(self) -> None:
        """Scan and catalog all documentation files."""
        if not self.docs_root.exists():
            logger.error(f"Documentation root not found: {self.docs_root}")
            return

        # Scan all markdown files in docs directory
        for file_path in self.docs_root.rglob("*.md"):
            if file_path.is_file():
                category = self._categorize_file(file_path)
                self.catalog[category].append(file_path)
                logger.debug(f"Categorized {file_path} as {category}")

        # Also scan for other documentation file types
        for ext in [".txt", ".rst", ".adoc"]:
            for file_path in self.docs_root.rglob(f"*{ext}"):
                if file_path.is_file():
                    category = self._categorize_file(file_path)
                    self.catalog[category].append(file_path)
                    logger.debug(f"Categorized {file_path} as {category}")

    def get_statistics(self) -> dict[str, int]:
        """Get statistics about the catalog.

        Returns:
            Dictionary with category counts.
        """
        return {
            category: len(files) for category, files in self.catalog.items()
        }

    def get_files_by_category(self, category: str) -> list[Path]:
        """Get all files in a specific category.

        Args:
            category: Category to filter by.

        Returns:
            List of file paths in the category.
        """
        return self.catalog.get(category, [])

    def export_catalog(self, output_path: Path) -> None:
        """Export the catalog to a JSON file.

        Args:
            output_path: Path to save the catalog.
        """
        catalog_data = {
            "statistics": self.get_statistics(),
            "files_by_category": {
                category: [str(file_path) for file_path in files]
                for category, files in self.catalog.items()
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(catalog_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Catalog exported to {output_path}")

    def print_summary(self) -> None:
        """Print a summary of the catalog."""
        stats = self.get_statistics()
        total_files = sum(stats.values())

        print("\n📚 Documentation Catalog Summary")
        print("{'=' * 50}")
        print(f"Total files found: {total_files}")
        print("\nFiles by category:")

        for category, count in stats.items():
            if count > 0:
                print(f"  {category.capitalize()}: {count} files")

        print("\nDetailed breakdown:")
        for category, files in self.catalog.items():
            if files:
                print(f"\n{category.upper()} ({len(files)} files):")
                for file_path in sorted(files):
                    relative_path = file_path.relative_to(self.project_root)
                    print(f"  - {relative_path}")


def main() -> None:
    """Main function to catalog documentation files."""
    project_root = Path(__file__).parent.parent.parent
    catalog = DocumentationCatalog(project_root)

    logger.info("Scanning documentation files...")
    catalog.scan_documentation_files()

    # Print summary
    catalog.print_summary()

    # Export catalog
    output_path = (
        project_root / "docs" / "reports" / "documentation_catalog.json"
    )
    catalog.export_catalog(output_path)

    logger.info("Documentation catalog completed successfully!")


if __name__ == "__main__":
    main()
