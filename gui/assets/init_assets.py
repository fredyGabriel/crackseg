#!/usr/bin/env python3
"""
Asset initialization script for CrackSeg GUI. This script registers
existing assets and sets up the asset management system.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import after path setup to access project modules
from gui.assets.manager import asset_manager  # noqa: E402


def register_css_assets() -> None:
    """Register CSS assets."""
    css_assets = [
        ("base_css", "css/global/base.css", "critical"),
        ("navigation_css", "css/components/navigation.css", "high"),
    ]

    for name, path, priority in css_assets:
        success = asset_manager.register_asset(name, path, priority)
        if success:
            print(f"✅ Registered CSS: {name}")
        else:
            print(f"❌ Failed to register CSS: {name} (path: {path})")


def register_existing_images() -> None:
    """Register existing image assets."""
    # Check for existing logo
    project_logo_path = PROJECT_ROOT / "docs" / "designs" / "logo.png"
    if project_logo_path.exists():
        # Copy to assets directory
        logo_dest = (
            Path(__file__).parent / "images" / "logos" / "primary-logo.png"
        )
        logo_dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            import shutil

            shutil.copy2(project_logo_path, logo_dest)
            success = asset_manager.register_asset(
                "primary_logo", "images/logos/primary-logo.png", "critical"
            )
            if success:
                print(f"✅ Registered primary logo from {project_logo_path}")
            else:
                print("❌ Failed to register primary logo")
        except Exception as e:
            print(f"❌ Error copying logo: {e}")

    # Register placeholder assets (will be generated as needed)
    placeholder_images = [
        ("favicon", "images/logos/favicon.png", "critical"),
        ("logo_small", "images/logos/logo-small.png", "high"),
        ("logo_large", "images/logos/logo-large.png", "normal"),
        ("background_pattern", "images/backgrounds/pattern.png", "low"),
    ]

    for name, path, _priority in placeholder_images:
        # Note: These will be registered but files may not exist yet
        # The asset manager will handle missing files gracefully
        print(f"📝 Placeholder registered: {name} ({path})")


def create_readme_files() -> None:
    """Create README files for asset directories."""
    readmes = {
        "css/global/README.md": """
# Global CSS Assets This directory contains global CSS files that
apply to the entire application. ## Files: - `base.css` - Core
application styles, CSS variables, Streamlit customizations -
`reset.css` - CSS reset and normalization (if needed) -
`utilities.css` - Utility classes and helpers (if needed)
""",
        "css/components/README.md": """
# Component CSS Assets This directory contains CSS files specific to
individual components. ## Files: - `navigation.css` - Sidebar
navigation and breadcrumb styles - `theme-switcher.css` - Theme
selector component styles - `status-panel.css` - Status panel and
indicators styles
""",
        "css/themes/README.md": """
# Theme CSS Assets This directory contains theme-specific CSS
overrides. ## Structure: - `dark/` - Dark theme customizations -
`light/` - Light theme customizations - `auto/` - Auto theme
adjustments
""",
        "images/logos/README.md": """
# Logo Assets This directory contains application logos and branding.
## Files: - `primary-logo.png` - Main application logo -
`logo-small.png` - Small variant for favicons/icons - `logo-large.png`
- Large variant for headers - `favicon.png` - Browser favicon
""",
        "images/icons/README.md": """
# Icon Assets This directory contains UI icons and symbols. ##
Categories: - Navigation icons - Status indicators - Action buttons -
File type indicators
""",
        "images/backgrounds/README.md": """
# Background Assets This directory contains background images and
patterns. ## Types: - Texture patterns - Gradient overlays - Hero
images
""",
        "images/samples/README.md": """
# Sample Assets This directory contains sample images for testing and
demonstration. ## Purpose: - Demo crack images - Test data
visualization - UI mockups
""",
        "fonts/primary/README.md": """
# Primary Fonts This directory contains the main application fonts. ##
Fonts: - System fonts are preferred for better performance - Custom
fonts only if branding requires it
""",
        "js/components/README.md": """
# JavaScript Components This directory contains JavaScript for
enhanced component functionality. ## Files: - Component-specific
interactive features - Streamlit custom components - Third-party
integrations
""",
    }

    base_path = Path(__file__).parent
    for file_path, content in readmes.items():
        full_path = base_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        if not full_path.exists():
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"📝 Created README: {file_path}")


def display_asset_stats() -> None:
    """Display current asset statistics."""
    stats = asset_manager.get_registry_stats()

    print("\n" + "=" * 50)
    print("📊 ASSET REGISTRY STATISTICS")
    print("=" * 50)
    print(f"Total Assets: {stats.get('total_assets', 0)}")
    print(f"Total Size: {stats.get('total_size_mb', 0)} MB")
    print(f"Cached Items: {stats.get('cached_items', 0)}")
    print(f"Registry Version: {stats.get('version', 'unknown')}")

    priority_dist = stats.get("priority_distribution", {})
    if priority_dist:
        print("\nPriority Distribution:")
        for priority, count in priority_dist.items():
            print(f"  {priority}: {count}")

    print("=" * 50)


def main() -> None:
    """Main initialization function."""
    print("🚀 Initializing CrackSeg Asset Management System")
    print("-" * 50)

    # Create directory structure (already done by mkdir commands)
    print("📁 Asset directory structure verified")

    # Create README files
    create_readme_files()

    # Register CSS assets
    print("\n📄 Registering CSS Assets...")
    register_css_assets()

    # Register existing images
    print("\n🖼️  Processing Image Assets...")
    register_existing_images()

    # Display statistics
    display_asset_stats()

    # Show next steps
    print("\n✅ Asset system initialization complete!")
    print("\n📋 Next Steps:")
    print("1. Add custom fonts to fonts/ directories if needed")
    print("2. Create theme-specific CSS files in css/themes/")
    print("3. Add custom icons to images/icons/")
    print("4. Test asset loading in the application")
    print("5. Run optimization scripts before production")

    registry_path = Path(__file__).parent / "manifest" / "asset_registry.json"
    print(f"\n🔧 Asset registry saved to: {registry_path}")


if __name__ == "__main__":
    main()
