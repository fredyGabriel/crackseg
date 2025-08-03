"""
Logo component for the CrackSeg application.

This module provides professional logo generation, loading, and rendering
capabilities with comprehensive fallback systems and caching.
"""

import base64
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from gui.assets.manager import asset_manager


class LogoComponent:
    """Professional logo component with fallback and caching capabilities."""

    _cache: dict[str, str] = {}
    _fallback_styles = {
        "default": {"bg_color": "#2E2E2E", "road_color": "#808080"},
        "light": {"bg_color": "#F0F0F0", "road_color": "#606060"},
        "minimal": {"bg_color": "#FFFFFF", "road_color": "#404040"},
    }

    @staticmethod
    def generate_logo(
        style: str = "default", width: int = 300, height: int = 300
    ) -> Image.Image:
        """Generate a professional logo representing crack segmentation.

        Args:
            style: Logo style variant ('default', 'light', 'minimal')
            width: Logo width in pixels
            height: Logo height in pixels

        Returns:
            PIL Image object containing the generated logo
        """
        colors = LogoComponent._fallback_styles.get(
            style, LogoComponent._fallback_styles["default"]
        )

        # Create base image
        img = Image.new("RGB", (width, height), color=colors["bg_color"])
        draw = ImageDraw.Draw(img)

        # Calculate proportional dimensions
        margin = width // 10
        road_area = (
            margin,
            margin,
            width - margin,
            height - margin,
        )

        # Draw road surface
        draw.rectangle(road_area, fill=colors["road_color"])

        # Generate realistic crack patterns
        LogoComponent._draw_crack_network(draw, road_area, "#FF4444")

        # Add segmentation overlay
        LogoComponent._add_segmentation_overlay(img, road_area)

        # Add professional text
        LogoComponent._add_logo_text(draw, width, height)

        return img

    @staticmethod
    def _draw_crack_network(
        draw: ImageDraw.ImageDraw, area: tuple[int, int, int, int], color: str
    ) -> None:
        """Draw realistic crack network patterns."""
        x1, y1, x2, y2 = area
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Main crack - organic curved line
        main_points = [
            (x1 + (x2 - x1) * 0.2, y1 + (y2 - y1) * 0.3),
            (center_x - 20, center_y - 30),
            (center_x + 10, center_y + 10),
            (x1 + (x2 - x1) * 0.7, y1 + (y2 - y1) * 0.8),
        ]

        # Draw main crack with varying width
        for i in range(len(main_points) - 1):
            width = 4 if i == 1 else 3  # Wider in center
            draw.line(
                [main_points[i], main_points[i + 1]], fill=color, width=width
            )

        # Secondary cracks branching from main crack
        branch_points = [
            (main_points[1], (center_x + 30, center_y - 50)),
            (main_points[2], (center_x - 25, center_y + 40)),
            (main_points[2], (center_x + 35, center_y + 20)),
        ]

        for start, end in branch_points:
            draw.line([start, end], fill=color, width=2)

    @staticmethod
    def _add_segmentation_overlay(
        img: Image.Image, area: tuple[int, int, int, int]
    ) -> None:
        """Add segmentation mask overlay with transparency."""
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        x1, y1, x2, y2 = area
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Segmentation regions around cracks
        regions = [
            (center_x - 30, center_y - 40, 25),
            (center_x + 10, center_y, 20),
            (center_x - 10, center_y + 30, 18),
        ]

        for x, y, radius in regions:
            # Segmentation mask with subtle green overlay
            overlay_draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=(0, 255, 100, 60),  # Semi-transparent green
            )

        # Composite overlay onto main image
        img_rgba = img.convert("RGBA")
        img_final = Image.alpha_composite(img_rgba, overlay)
        img.paste(img_final.convert("RGB"))

    @staticmethod
    def _add_logo_text(
        draw: ImageDraw.ImageDraw, width: int, height: int
    ) -> None:
        """Add professional text to the logo."""
        # Define font options at the beginning for both main title and subtitle
        font_options = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]

        try:
            # Try multiple font options for better compatibility
            font = None

            for font_name in font_options:
                try:
                    font = ImageFont.truetype(font_name, 28)
                    break
                except OSError:
                    continue

            if font is None:
                font = ImageFont.load_default()

        except Exception:
            font = ImageFont.load_default()

        # Main title
        title = "CrackSeg"
        title_bbox = draw.textbbox((0, 0), title, font=font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2

        # Add text with shadow for better visibility
        shadow_offset = 2
        draw.text(
            (title_x + shadow_offset, 25 + shadow_offset),
            title,
            fill="#000000",
            font=font,
        )
        draw.text((title_x, 25), title, fill="#FFFFFF", font=font)

        # Subtitle
        try:
            subtitle_font = ImageFont.truetype(font_options[0], 14)
        except Exception:
            subtitle_font = ImageFont.load_default()

        subtitle = "AI-Powered Analysis"
        subtitle_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
        subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
        subtitle_x = (width - subtitle_width) // 2

        draw.text(
            (subtitle_x, height - 40),
            subtitle,
            fill="#CCCCCC",
            font=subtitle_font,
        )

    @staticmethod
    def load_with_fallback(
        primary_path: str | Path | None = None,
        style: str = "default",
        width: int = 300,
        height: int = 300,
        use_cache: bool = True,
        project_root: Path | None = None,
    ) -> str | None:
        """Load logo with comprehensive fallback system and caching.

        Args:
            primary_path: Primary logo file path
            style: Fallback logo style
            width: Logo width for generated fallback
            height: Logo height for generated fallback
            use_cache: Whether to use cached results
            project_root: Project root path for default locations

        Returns:
            Base64 encoded logo data URL or None if all methods fail
        """
        cache_key = f"{primary_path}_{style}_{width}_{height}"

        # Return cached result if available
        if use_cache and cache_key in LogoComponent._cache:
            return LogoComponent._cache[cache_key]

        logo_data = None

        # Attempt 1: Load from Asset Manager
        if not logo_data:
            # Try primary logo from asset manager
            asset_url = asset_manager.get_asset_url("primary_logo")
            if asset_url and asset_url.startswith("data:"):
                # Extract base64 data from crackseg.data URL
                try:
                    logo_base64 = asset_url.split("base64,")[1]
                    logo_data = base64.b64decode(logo_base64)
                except Exception:
                    pass

        # Attempt 2: Load from primary path
        if not logo_data and primary_path:
            logo_data = LogoComponent._load_from_file(Path(primary_path))

        # Attempt 3: Load from default locations
        if not logo_data and project_root:
            default_paths = [
                project_root / "docs" / "designs" / "logo.png",
                project_root / "assets" / "logo.png",
                project_root / "scripts" / "gui" / "assets" / "logo.png",
            ]

            for path in default_paths:
                logo_data = LogoComponent._load_from_file(path)
                if logo_data:
                    break

        # Attempt 4: Generate fallback logo
        if not logo_data:
            try:
                logo_img = LogoComponent.generate_logo(style, width, height)
                buffer = BytesIO()
                logo_img.save(buffer, format="PNG", optimize=True)
                logo_data = buffer.getvalue()

                # Save generated logo for future use
                if project_root:
                    LogoComponent._save_generated_logo(logo_data, project_root)

            except Exception as e:
                st.error(f"Failed to generate fallback logo: {e}")
                return None

        # Convert to base64 data URL
        if logo_data:
            try:
                logo_base64 = base64.b64encode(logo_data).decode()
                data_url = f"data:image/png;base64,{logo_base64}"

                # Cache the result
                if use_cache:
                    LogoComponent._cache[cache_key] = data_url

                return data_url

            except Exception as e:
                st.error(f"Failed to encode logo data: {e}")

        return None

    @staticmethod
    def _load_from_file(path: Path) -> bytes | None:
        """Load logo data from file with error handling."""
        try:
            if path.exists() and path.is_file():
                with open(path, "rb") as f:
                    return f.read()
        except Exception as e:
            st.warning(f"Could not load logo from {path}: {e}")
        return None

    @staticmethod
    def _save_generated_logo(logo_data: bytes, project_root: Path) -> None:
        """Save generated logo to default location."""
        try:
            default_path = project_root / "docs" / "designs" / "logo.png"
            default_path.parent.mkdir(parents=True, exist_ok=True)

            with open(default_path, "wb") as f:
                f.write(logo_data)

        except Exception:
            # Silent fail - this is just for caching
            pass

    @staticmethod
    def render(
        primary_path: str | Path | None = None,
        style: str = "default",
        width: int = 150,
        alt_text: str = "CrackSeg Logo",
        css_class: str = "",
        center: bool = True,
        project_root: Path | None = None,
    ) -> None:
        """Render logo component in Streamlit interface.

        Args:
            primary_path: Primary logo file path
            style: Fallback logo style
            width: Display width in pixels
            alt_text: Alt text for accessibility
            css_class: Additional CSS classes
            center: Whether to center the logo
            project_root: Project root path for fallback locations
        """
        logo_data = LogoComponent.load_with_fallback(
            primary_path=primary_path,
            style=style,
            width=width * 2,
            height=width * 2,
            project_root=project_root,
        )

        if logo_data:
            # Build CSS styles
            styles = [f"width: {width}px", "height: auto"]
            if center:
                styles.append("display: block")
                styles.append("margin: 0 auto")

            style_attr = "; ".join(styles)

            # Build container styles
            container_styles = []
            if center:
                container_styles.append("text-align: center")

            container_style = (
                "; ".join(container_styles) if container_styles else ""
            )

            # Render with proper HTML structure
            html = f"""
            <div style="{container_style}">
                <img src="{logo_data}"
                     alt="{alt_text}"
                     title="{alt_text}"
                     style="{style_attr}"
                     class="{css_class}">
            </div>
            """

            st.markdown(html, unsafe_allow_html=True)
        else:
            # Fallback text display
            if center:
                st.markdown(
                    f"<div style='text-align: center; font-size: 24px; "
                    f"font-weight: bold; color: #666;'>{alt_text}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"**{alt_text}**")

    @staticmethod
    def clear_cache() -> None:
        """Clear the logo cache."""
        LogoComponent._cache.clear()
