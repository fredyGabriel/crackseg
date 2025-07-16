"""
CrackSeg - Pavement Crack Segmentation GUI
Main entry point for the Streamlit application
"""

# ruff: noqa: E501

import base64
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class SessionState:
    """Structured session state management for the CrackSeg application."""

    # Core application state
    config_path: str | None = None
    run_directory: str | None = None
    current_page: str = "Config"
    theme: str = "dark"

    # Configuration state
    config_loaded: bool = False
    config_data: dict[str, Any] | None = None

    # Training state
    training_active: bool = False
    training_progress: float = 0.0
    training_metrics: dict[str, float] = field(default_factory=dict)

    # Model state
    model_loaded: bool = False
    model_architecture: str | None = None
    model_parameters: dict[str, Any] | None = None

    # Results state
    last_evaluation: dict[str, Any] | None = None
    results_available: bool = False

    # UI state
    sidebar_expanded: bool = True
    notifications: list[str] = field(default_factory=list)

    # Session metadata
    session_id: str = field(
        default_factory=lambda: str(datetime.now().timestamp())
    )
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert session state to dictionary for serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionState":
        """Create SessionState from dictionary."""
        # Handle datetime conversion
        if "last_updated" in data and isinstance(data["last_updated"], str):
            data["last_updated"] = datetime.fromisoformat(data["last_updated"])

        return cls(
            **{k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        )

    def update_config(
        self, config_path: str, config_data: dict[str, Any] | None = None
    ) -> None:
        """Update configuration state."""
        self.config_path = config_path
        self.config_data = config_data
        self.config_loaded = config_data is not None
        self.last_updated = datetime.now()

    def update_training_progress(
        self, progress: float, metrics: dict[str, float] | None = None
    ) -> None:
        """Update training progress and metrics."""
        self.training_progress = max(0.0, min(1.0, progress))
        if metrics:
            self.training_metrics.update(metrics)
        self.last_updated = datetime.now()

    def set_training_active(self, active: bool) -> None:
        """Set training active state."""
        self.training_active = active
        if not active:
            self.training_progress = 0.0
        self.last_updated = datetime.now()

    def add_notification(self, message: str) -> None:
        """Add a notification message."""
        self.notifications.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        )
        # Keep only last 10 notifications
        self.notifications = self.notifications[-10:]
        self.last_updated = datetime.now()

    def clear_notifications(self) -> None:
        """Clear all notifications."""
        self.notifications.clear()
        self.last_updated = datetime.now()

    def is_ready_for_training(self) -> bool:
        """Check if application is ready for training."""
        return self.config_loaded and self.run_directory is not None

    def is_ready_for_results(self) -> bool:
        """Check if results can be displayed."""
        return self.results_available or self.run_directory is not None

    def validate(self) -> list[str]:
        """Validate session state and return list of issues."""
        issues = []

        if self.config_path and not Path(self.config_path).exists():
            issues.append("Configuration file does not exist")

        if self.run_directory and not Path(self.run_directory).exists():
            issues.append("Run directory does not exist")

        if self.training_progress < 0 or self.training_progress > 1:
            issues.append("Training progress must be between 0 and 1")

        return issues


class SessionStateManager:
    """Manager class for handling session state operations."""

    @staticmethod
    def initialize() -> None:
        """Initialize session state if not already present."""
        if "app_state" not in st.session_state:
            st.session_state.app_state = SessionState()

        # Ensure backward compatibility with existing code
        SessionStateManager._sync_legacy_state()

    @staticmethod
    def get() -> SessionState:
        """Get current session state."""
        if "app_state" not in st.session_state:
            SessionStateManager.initialize()
        return st.session_state.app_state

    @staticmethod
    def update(updates: dict[str, Any]) -> None:
        """Update session state with new values."""
        state = SessionStateManager.get()
        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)
        state.last_updated = datetime.now()

    @staticmethod
    def _sync_legacy_state() -> None:
        """Sync with legacy session state variables for compatibility."""
        state = SessionStateManager.get()

        # Sync with legacy variables
        if "config_path" in st.session_state:
            state.config_path = st.session_state.config_path
        if "run_directory" in st.session_state:
            state.run_directory = st.session_state.run_directory
        if "current_page" in st.session_state:
            state.current_page = st.session_state.current_page
        if "theme" in st.session_state:
            state.theme = st.session_state.theme

        # Update legacy variables to match state
        st.session_state.config_path = state.config_path
        st.session_state.run_directory = state.run_directory
        st.session_state.current_page = state.current_page
        st.session_state.theme = state.theme

    @staticmethod
    def save_to_file(filepath: Path) -> bool:
        """Save session state to file."""
        try:
            state = SessionStateManager.get()
            with open(filepath, "w") as f:
                json.dump(state.to_dict(), f, indent=2)
            return True
        except Exception as e:
            st.error(f"Failed to save session state: {e}")
            return False

    @staticmethod
    def load_from_file(filepath: Path) -> bool:
        """Load session state from file."""
        try:
            if not filepath.exists():
                return False

            with open(filepath) as f:
                data = json.load(f)

            state = SessionState.from_dict(data)
            st.session_state.app_state = state
            SessionStateManager._sync_legacy_state()
            return True
        except Exception as e:
            st.error(f"Failed to load session state: {e}")
            return False


# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="CrackSeg - Crack Segmentation",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourusername/crackseg",
        "Report a bug": "https://github.com/yourusername/crackseg/issues",
        "About": (
            "# CrackSeg\nA deep learning application for pavement "
            "crack segmentation"
        ),
    },
)


# Shared page configuration
PAGE_CONFIG = {
    "Config": {
        "icon": "🔧",
        "description": "Configure model and training parameters",
        "requires": [],
    },
    "Architecture": {
        "icon": "🏗️",
        "description": "Visualize model architecture",
        "requires": ["config_loaded"],
    },
    "Train": {
        "icon": "🚀",
        "description": "Launch and monitor training",
        "requires": ["config_loaded", "run_directory"],
    },
    "Results": {
        "icon": "📊",
        "description": "View results and export reports",
        "requires": ["run_directory"],
    },
}


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
    ) -> str | None:
        """Load logo with comprehensive fallback system and caching.

        Args:
            primary_path: Primary logo file path
            style: Fallback logo style
            width: Logo width for generated fallback
            height: Logo height for generated fallback
            use_cache: Whether to use cached results

        Returns:
            Base64 encoded logo data URL or None if all methods fail
        """
        cache_key = f"{primary_path}_{style}_{width}_{height}"

        # Return cached result if available
        if use_cache and cache_key in LogoComponent._cache:
            return LogoComponent._cache[cache_key]

        logo_data = None

        # Attempt 1: Load from primary path
        if primary_path:
            logo_data = LogoComponent._load_from_file(Path(primary_path))

        # Attempt 2: Load from default locations
        if not logo_data:
            default_paths = [
                PROJECT_ROOT / "docs" / "designs" / "logo.png",
                PROJECT_ROOT / "assets" / "logo.png",
                PROJECT_ROOT / "scripts" / "gui" / "assets" / "logo.png",
            ]

            for path in default_paths:
                logo_data = LogoComponent._load_from_file(path)
                if logo_data:
                    break

        # Attempt 3: Generate fallback logo
        if not logo_data:
            try:
                logo_img = LogoComponent.generate_logo(style, width, height)
                buffer = BytesIO()
                logo_img.save(buffer, format="PNG", optimize=True)
                logo_data = buffer.getvalue()

                # Save generated logo for future use
                LogoComponent._save_generated_logo(logo_data)

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
    def _save_generated_logo(logo_data: bytes) -> None:
        """Save generated logo to default location."""
        try:
            default_path = PROJECT_ROOT / "docs" / "designs" / "logo.png"
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
    ) -> None:
        """Render logo component in Streamlit interface.

        Args:
            primary_path: Primary logo file path
            style: Fallback logo style
            width: Display width in pixels
            alt_text: Alt text for accessibility
            css_class: Additional CSS classes
            center: Whether to center the logo
        """
        logo_data = LogoComponent.load_with_fallback(
            primary_path=primary_path,
            style=style,
            width=width * 2,
            height=width * 2,
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


class SidebarComponent:
    """Professional sidebar navigation component with enhanced functionality."""

    @staticmethod
    def _get_page_status(page: str, state: SessionState) -> tuple[bool, str]:
        """Get page availability status and reason.

        Args:
            page: Page name to check
            state: Current session state

        Returns:
            Tuple of (is_available, status_message)
        """
        config = PAGE_CONFIG.get(page, {})
        requirements = config.get("requires", [])

        if not requirements:
            return True, "Ready"

        missing = []
        for req in requirements:
            if req == "config_loaded" and not state.config_loaded:
                missing.append("configuration")
            elif req == "run_directory" and not state.run_directory:
                missing.append("run directory")

        if missing:
            return False, f"Needs: {', '.join(missing)}"

        return True, "Ready"

    @staticmethod
    def _render_navigation_item(
        page: str,
        current_page: str,
        state: SessionState,
        is_available: bool,
        status: str,
    ) -> None:
        """Render a single navigation item with status indicators.

        Args:
            page: Page name
            current_page: Currently selected page
            state: Session state
            is_available: Whether page is available
            status: Status message
        """
        config = PAGE_CONFIG[page]
        icon = config["icon"]
        description = config["description"]

        # Determine styling based on status
        if page == current_page:
            # Current page - highlight
            container_style = """
                <div style='
                    padding: 8px 12px;
                    border-radius: 8px;
                    background-color: rgba(0, 122, 255, 0.1);
                    border-left: 4px solid #007AFF;
                    margin: 4px 0;
                '>
            """
        elif is_available:
            # Available page - normal
            container_style = """
                <div style='
                    padding: 8px 12px;
                    border-radius: 8px;
                    background-color: rgba(255, 255, 255, 0.05);
                    border-left: 4px solid transparent;
                    margin: 4px 0;
                    cursor: pointer;
                '>
            """
        else:
            # Unavailable page - muted
            container_style = """
                <div style='
                    padding: 8px 12px;
                    border-radius: 8px;
                    background-color: rgba(128, 128, 128, 0.1);
                    border-left: 4px solid #666;
                    margin: 4px 0;
                    opacity: 0.6;
                '>
            """

        # Status indicator
        if page == current_page:
            status_indicator = "●"
            status_color = "#007AFF"
        elif is_available:
            status_indicator = "●"
            status_color = "#34C759"
        else:
            status_indicator = "○"
            status_color = "#FF9500"

        # Build the item HTML
        item_html = f"""
        {container_style}
            <div style='display: flex; align-items: center; justify-content: space-between;'>
                <div style='display: flex; align-items: center;'>
                    <span style='font-size: 20px; margin-right: 12px;'>{icon}</span>
                    <div>
                        <div style='font-weight: {"bold" if page == current_page else "normal"};
                                    font-size: 16px; color: {"#007AFF" if page == current_page else "inherit"};'>
                            {page}
                        </div>
                        <div style='font-size: 12px; color: #666; margin-top: 2px;'>
                            {description}
                        </div>
                    </div>
                </div>
                <div style='text-align: right;'>
                    <span style='color: {status_color}; font-size: 12px;'>{status_indicator}</span>
                    <div style='font-size: 10px; color: #888; margin-top: 2px;'>
                        {status}
                    </div>
                </div>
            </div>
        </div>
        """

        st.markdown(item_html, unsafe_allow_html=True)

    @staticmethod
    def _render_page_selector(
        pages: list[str], current_page: str, state: SessionState
    ) -> str:
        """Render enhanced page selector with status indicators.

        Args:
            pages: List of available pages
            current_page: Currently selected page
            state: Session state

        Returns:
            Selected page name
        """
        st.markdown("### 🧭 Navigation")

        # Render navigation items
        for page in pages:
            is_available, status = SidebarComponent._get_page_status(
                page, state
            )
            SidebarComponent._render_navigation_item(
                page, current_page, state, is_available, status
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Page selector (functional)
        available_pages = []
        for page in pages:
            is_available, _ = SidebarComponent._get_page_status(page, state)
            if is_available:
                available_pages.append(page)

        if current_page not in available_pages and available_pages:
            # Auto-select first available page if current is unavailable
            selected_page = available_pages[0]
        else:
            selected_page = st.selectbox(
                "Page Navigation",
                options=available_pages,
                index=(
                    available_pages.index(current_page)
                    if current_page in available_pages
                    else 0
                ),
                key="page_selector",
                label_visibility="collapsed",
            )

        return selected_page

    @staticmethod
    def _render_status_panel(state: SessionState) -> None:
        """Render enhanced status information panel.

        Args:
            state: Current session state
        """
        st.markdown("### 📋 Status Panel")

        # Configuration status
        config_status = "✅ Loaded" if state.config_loaded else "⏳ Not loaded"
        config_color = "green" if state.config_loaded else "orange"

        st.markdown(
            f"""
            <div style='padding: 8px; border-radius: 6px; background-color: rgba(255,255,255,0.05); margin: 4px 0;'>
                <div style='font-weight: bold; color: {config_color};'>📄 Configuration</div>
                <div style='font-size: 12px; color: #888;'>{config_status}</div>
                <div style='font-size: 11px; color: #666; margin-top: 2px;'>
                    {state.config_path or "No path set"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Run directory status
        run_status = "✅ Set" if state.run_directory else "⏳ Not set"
        run_color = "green" if state.run_directory else "orange"

        st.markdown(
            f"""
            <div style='padding: 8px; border-radius: 6px; background-color: rgba(255,255,255,0.05); margin: 4px 0;'>
                <div style='font-weight: bold; color: {run_color};'>📁 Run Directory</div>
                <div style='font-size: 12px; color: #888;'>{run_status}</div>
                <div style='font-size: 11px; color: #666; margin-top: 2px;'>
                    {state.run_directory or "No directory set"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Training status
        if state.training_active:
            training_color = "blue"
            training_status = f"🚀 Active ({state.training_progress:.1%})"
            # Progress bar
            progress_html = f"""
                <div style='background-color: #333; border-radius: 4px; height: 6px; margin: 4px 0;'>
                    <div style='background-color: #007AFF; height: 100%; border-radius: 4px;
                                width: {state.training_progress * 100}%;'></div>
                </div>
            """
        else:
            training_color = "gray"
            training_status = "⏸️ Inactive"
            progress_html = ""

        st.markdown(
            f"""
            <div style='padding: 8px; border-radius: 6px; background-color: rgba(255,255,255,0.05); margin: 4px 0;'>
                <div style='font-weight: bold; color: {training_color};'>🎯 Training</div>
                <div style='font-size: 12px; color: #888;'>{training_status}</div>
                {progress_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def _render_notifications(state: SessionState) -> None:
        """Render enhanced notifications panel.

        Args:
            state: Current session state
        """
        if not state.notifications:
            return

        st.markdown("### 🔔 Recent Activity")

        # Show last 3 notifications with better styling
        for notification in state.notifications[-3:]:
            # Extract timestamp and message
            if "]" in notification:
                timestamp, message = notification.split("]", 1)
                timestamp = timestamp.strip("[")
                message = message.strip()
            else:
                timestamp = "Unknown"
                message = notification

            st.markdown(
                f"""
                <div style='padding: 6px 8px; border-radius: 4px;
                           background-color: rgba(0, 122, 255, 0.1);
                           border-left: 3px solid #007AFF; margin: 3px 0;'>
                    <div style='font-size: 11px; color: #007AFF; font-weight: bold;'>{timestamp}</div>
                    <div style='font-size: 12px; color: #333; margin-top: 2px;'>{message}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Clear button with better styling
        if st.button(
            "🗑️ Clear Notifications",
            key="clear_notifications_enhanced",
            use_container_width=True,
        ):
            state.clear_notifications()
            st.rerun()

    @staticmethod
    def _render_quick_actions(state: SessionState) -> None:
        """Render quick action buttons.

        Args:
            state: Current session state
        """
        st.markdown("### ⚡ Quick Actions")

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "🔄 Refresh", key="refresh_app", use_container_width=True
            ):
                st.rerun()

        with col2:
            readiness = state.is_ready_for_training()
            train_button_text = "✅ Ready" if readiness else "❌ Not Ready"
            st.button(
                train_button_text,
                key="training_readiness",
                disabled=True,
                use_container_width=True,
                help="Training readiness indicator",
            )

    @staticmethod
    def render() -> str:
        """Render complete enhanced sidebar navigation.

        Returns:
            Selected page name
        """
        state = SessionStateManager.get()

        with st.sidebar:
            # Logo section
            LogoComponent.render(
                primary_path=state.config_path,
                style=state.theme,
                width=150,
                alt_text="CrackSeg Logo",
                css_class="logo",
                center=True,
            )

            st.markdown("---")

            # Enhanced navigation
            pages = ["Config", "Architecture", "Train", "Results"]
            selected_page = SidebarComponent._render_page_selector(
                pages, state.current_page, state
            )

            # Update session state if page changed
            if selected_page != state.current_page:
                SessionStateManager.update({"current_page": selected_page})

            st.markdown("---")

            # Status panel
            SidebarComponent._render_status_panel(state)

            st.markdown("---")

            # Notifications
            SidebarComponent._render_notifications(state)

            st.markdown("---")

            # Quick actions
            SidebarComponent._render_quick_actions(state)

            # App info
            st.markdown("---")
            st.markdown(
                """
                <div style='text-align: center; font-size: 11px; color: #666;'>
                    <div>CrackSeg v1.0</div>
                    <div>AI-Powered Crack Detection</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        return selected_page


class PageRouter:
    """Centralized page routing system for the application."""

    # Page component mapping
    _page_components = {
        "Config": "page_config",
        "Architecture": "page_architecture",
        "Train": "page_train",
        "Results": "page_results",
    }

    # Page metadata for enhanced routing
    _page_metadata = {
        "Config": {
            "title": "🔧 Configuration Manager",
            "subtitle": "Configure your crack segmentation model and training parameters",
            "help_text": "Load YAML configurations and set up your training environment",
        },
        "Architecture": {
            "title": "🏗️ Model Architecture Viewer",
            "subtitle": "Visualize and understand your model structure",
            "help_text": "Explore model components, layers, and connections",
        },
        "Train": {
            "title": "🚀 Training Dashboard",
            "subtitle": "Launch and monitor model training",
            "help_text": "Start training, view real-time metrics, and manage checkpoints",
        },
        "Results": {
            "title": "📊 Results Gallery",
            "subtitle": "Analyze predictions and export reports",
            "help_text": "View segmentation results, metrics, and generate comprehensive reports",
        },
    }

    @staticmethod
    def get_available_pages(state: SessionState) -> list[str]:
        """Get list of currently available pages based on state.

        Args:
            state: Current session state

        Returns:
            List of available page names
        """
        available = []

        for page, config in PAGE_CONFIG.items():
            requirements = config.get("requires", [])
            is_available = True

            for req in requirements:
                if req == "config_loaded" and not state.config_loaded:
                    is_available = False
                    break
                elif req == "run_directory" and not state.run_directory:
                    is_available = False
                    break

            if is_available:
                available.append(page)

        return available

    @staticmethod
    def validate_page_transition(
        from_page: str, to_page: str, state: SessionState
    ) -> tuple[bool, str]:
        """Validate if page transition is allowed.

        Args:
            from_page: Current page name
            to_page: Target page name
            state: Current session state

        Returns:
            Tuple of (is_valid, error_message)
        """
        available_pages = PageRouter.get_available_pages(state)

        if to_page not in available_pages:
            requirements = PAGE_CONFIG.get(to_page, {}).get("requires", [])
            missing = []

            for req in requirements:
                if req == "config_loaded" and not state.config_loaded:
                    missing.append("Configuration must be loaded")
                elif req == "run_directory" and not state.run_directory:
                    missing.append("Run directory must be set")

            error_msg = (
                f"Cannot navigate to {to_page}. Missing: {', '.join(missing)}"
            )
            return False, error_msg

        return True, ""

    @staticmethod
    def route_to_page(page_name: str, state: SessionState) -> None:
        """Route to the specified page with validation.

        Args:
            page_name: Name of the page to route to
            state: Current session state
        """
        # Validate transition
        is_valid, error_msg = PageRouter.validate_page_transition(
            state.current_page, page_name, state
        )

        if not is_valid:
            st.error(error_msg)
            return

        # Update session state
        SessionStateManager.update({"current_page": page_name})
        state.add_notification(f"Navigated to {page_name}")

        # Get page metadata
        metadata = PageRouter._page_metadata.get(page_name, {})

        # Render page header
        if metadata:
            st.title(metadata.get("title", page_name))
            st.markdown(metadata.get("subtitle", ""))

            # Add help expander
            with st.expander("ℹ️ Page Help", expanded=False):
                st.info(metadata.get("help_text", "No help available"))

            st.markdown("---")

        # Route to appropriate page function
        page_function = PageRouter._page_components.get(page_name)
        if page_function:
            # Call the page function dynamically
            globals()[page_function]()
        else:
            st.error(f"Page component not found: {page_name}")

    @staticmethod
    def handle_navigation_change(new_page: str, state: SessionState) -> bool:
        """Handle navigation change with validation and state update.

        Args:
            new_page: New page to navigate to
            state: Current session state

        Returns:
            True if navigation was successful
        """
        if new_page == state.current_page:
            return True

        # Validate transition
        is_valid, error_msg = PageRouter.validate_page_transition(
            state.current_page, new_page, state
        )

        if is_valid:
            SessionStateManager.update({"current_page": new_page})
            state.add_notification(f"Navigated to {new_page}")
            return True
        else:
            st.sidebar.error(error_msg)
            return False

    @staticmethod
    def get_page_breadcrumbs(current_page: str) -> str:
        """Generate breadcrumb navigation for current page.

        Args:
            current_page: Current page name

        Returns:
            Breadcrumb HTML string
        """
        pages = ["Config", "Architecture", "Train", "Results"]
        breadcrumbs = []

        for page in pages:
            if page == current_page:
                breadcrumbs.append(f"**{page}**")
            else:
                breadcrumbs.append(page)

        return " → ".join(breadcrumbs)


def render_sidebar() -> str:
    """Render sidebar with navigation and logo - Enhanced with PageRouter integration."""
    state = SessionStateManager.get()

    with st.sidebar:
        # Logo
        LogoComponent.render(
            primary_path=state.config_path,
            style=state.theme,
            width=150,
            alt_text="CrackSeg Logo",
            css_class="logo",
            center=True,
        )

        st.markdown("---")

        # Navigation
        st.markdown("### 🧭 Navigation")

        # Get available pages from PageRouter
        available_pages = PageRouter.get_available_pages(state)
        all_pages = list(PAGE_CONFIG.keys())

        # Display page status
        for page in all_pages:
            config = PAGE_CONFIG[page]
            icon = config["icon"]
            description = config["description"]
            is_available = page in available_pages
            is_current = page == state.current_page

            if is_current:
                st.success(f"**➤ {icon} {page}** (Current)")
            elif is_available:
                if st.button(
                    f"✅ {icon} {page}",
                    key=f"nav_{page}",
                    use_container_width=True,
                    help=str(description) if description else None,
                ):
                    # Use PageRouter to handle navigation
                    if PageRouter.handle_navigation_change(page, state):
                        st.rerun()
            else:
                st.warning(f"⚠️ {icon} {page} (Needs setup)")
                # Show requirements
                requirements = config.get("requires", [])
                req_messages = []
                for req in requirements:
                    if req == "config_loaded" and not state.config_loaded:
                        req_messages.append("Load configuration")
                    elif req == "run_directory" and not state.run_directory:
                        req_messages.append("Set run directory")
                if req_messages:
                    st.caption(f"Required: {', '.join(req_messages)}")

        st.markdown("---")

        # Status Panel
        st.markdown("### 📋 Status")

        # Configuration status
        if state.config_loaded:
            st.success("📄 Configuration: Loaded")
            if state.config_path:
                st.caption(f"Path: {Path(state.config_path).name}")
        else:
            st.error("📄 Configuration: Not loaded")

        # Run directory status
        if state.run_directory:
            st.success("📁 Run Directory: Set")
            st.caption(f"Path: {Path(state.run_directory).name}")
        else:
            st.error("📁 Run Directory: Not set")

        # Training status
        if state.training_active:
            st.info(f"🚀 Training: Active ({state.training_progress:.1%})")
            st.progress(state.training_progress)
        else:
            st.warning("🚀 Training: Inactive")

        st.markdown("---")

        # Notifications
        if state.notifications:
            st.markdown("### 🔔 Recent Activity")
            for notification in state.notifications[-3:]:  # Show last 3
                st.caption(notification)

            if st.button(
                "🗑️ Clear Notifications",
                key="clear_notifications",
                use_container_width=True,
            ):
                state.clear_notifications()
                st.rerun()

        # Quick Actions
        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "🔄 Refresh", key="refresh_app", use_container_width=True
            ):
                st.rerun()

        with col2:
            readiness = state.is_ready_for_training()
            if readiness:
                st.success("✅ Ready")
            else:
                st.error("❌ Not Ready")

        # App info
        st.markdown("---")
        st.caption("CrackSeg v1.0")
        st.caption("AI-Powered Crack Detection")

    return state.current_page


def initialize_session_state() -> None:
    """Initialize session state variables using the new SessionStateManager."""
    SessionStateManager.initialize()


def page_config() -> None:
    """Configuration page content."""
    state = SessionStateManager.get()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Configuration")
        config_input = st.text_input(
            "Config File Path",
            key="config_input",
            value=state.config_path or "",
            help="Path to your YAML configuration file",
        )

        if st.button("Load Configuration"):
            if config_input:
                try:
                    # Validate path exists
                    config_path = Path(config_input)
                    if config_path.exists():
                        state.update_config(config_input, {"loaded": True})
                        state.add_notification(
                            f"Configuration loaded: {config_input}"
                        )
                        st.success(
                            f"Configuration loaded from: {config_input}"
                        )
                    else:
                        st.error("Configuration file does not exist")
                except Exception as e:
                    st.error(f"Error loading configuration: {e}")
            else:
                st.error("Please provide a configuration file path")

    with col2:
        st.subheader("Output Settings")
        run_dir_input = st.text_input(
            "Run Directory",
            key="run_dir_input",
            value=state.run_directory or "",
            help="Directory where outputs will be saved",
        )

        if st.button("Set Run Directory"):
            if run_dir_input:
                try:
                    # Create directory if it doesn't exist
                    run_dir = Path(run_dir_input)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    SessionStateManager.update(
                        {"run_directory": run_dir_input}
                    )
                    state.add_notification(
                        f"Run directory set: {run_dir_input}"
                    )
                    st.success(f"Run directory set to: {run_dir_input}")
                except Exception as e:
                    st.error(f"Error setting run directory: {e}")
            else:
                st.error("Please provide a run directory path")

    # Configuration preview
    st.markdown("---")
    st.subheader("Configuration Preview")

    if state.config_loaded:
        st.success("✅ Configuration loaded successfully")
        if state.config_data:
            st.json(state.config_data)
    else:
        st.info(
            "Configuration preview will be implemented once config loading "
            "is complete"
        )

    # Validation status
    st.markdown("---")
    st.subheader("Validation Status")
    issues = state.validate()
    if issues:
        for issue in issues:
            st.warning(f"⚠️ {issue}")
    else:
        st.success("✅ All validations passed")


def page_architecture() -> None:
    """Architecture visualization page content."""
    state = SessionStateManager.get()

    # Model status
    if state.model_loaded:
        st.success(f"✅ Model loaded: {state.model_architecture}")
    else:
        st.info("No model currently loaded")

    # Placeholder content
    st.info("Architecture visualization will be implemented in future updates")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Components")
        st.markdown(
            """
        - **Encoder**: Feature extraction backbone
        - **Bottleneck**: Feature processing
        - **Decoder**: Segmentation head
        """
        )

    with col2:
        st.subheader("Model Statistics")
        if state.model_parameters:
            st.json(state.model_parameters)
        else:
            st.markdown(
                """
            - **Parameters**: TBD
            - **FLOPs**: TBD
            - **Input Size**: TBD
            """
            )


def page_train() -> None:
    """Training page content."""
    state = SessionStateManager.get()

    if not state.is_ready_for_training():
        st.warning("Please complete configuration setup before training.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "▶️ Start Training", type="primary", disabled=state.training_active
        ):
            state.set_training_active(True)
            state.add_notification("Training started")
            st.success("Training started!")

    with col2:
        if st.button("⏸️ Pause Training", disabled=not state.training_active):
            state.add_notification("Training paused")
            st.info(
                "Pause functionality will be implemented in future updates"
            )

    with col3:
        if st.button("⏹️ Stop Training", disabled=not state.training_active):
            state.set_training_active(False)
            state.add_notification("Training stopped")
            st.info("Training stopped")

    # Training progress
    st.markdown("---")
    st.subheader("Training Progress")

    if state.training_active:
        st.progress(state.training_progress)
        st.info(f"Training in progress: {state.training_progress:.1%}")

        # Mock progress update for demo
        if st.button("Simulate Progress", key="sim_progress"):
            new_progress = min(1.0, state.training_progress + 0.1)
            state.update_training_progress(
                new_progress, {"loss": 0.5 - new_progress * 0.3}
            )
    else:
        st.info("Real-time training metrics will be displayed here")

    # Training metrics
    if state.training_metrics:
        st.markdown("### Current Metrics")
        for metric, value in state.training_metrics.items():
            st.metric(metric.capitalize(), f"{value:.4f}")


def page_results() -> None:
    """Results visualization page content."""
    state = SessionStateManager.get()

    if not state.is_ready_for_results():
        st.warning("Please complete training or set a run directory first.")
        return

    # Results tabs
    tab1, tab2, tab3 = st.tabs(
        ["Metrics", "Visualizations", "Model Comparison"]
    )

    with tab1:
        st.subheader("Training Metrics")
        if state.last_evaluation:
            st.json(state.last_evaluation)
        else:
            st.info(
                "Training metrics visualization will be implemented in future "
                "updates"
            )

    with tab2:
        st.subheader("Segmentation Results")
        st.info(
            "Segmentation visualization will be implemented in future updates"
        )

    with tab3:
        st.subheader("Model Comparison")
        st.info("Model comparison tools will be implemented in future updates")


def main() -> None:
    """Main application entry point with enhanced routing."""
    # Initialize session state
    initialize_session_state()

    # Get current state
    state = SessionStateManager.get()

    # Render sidebar and get current page
    current_page = render_sidebar()

    # Display breadcrumb navigation
    breadcrumbs = PageRouter.get_page_breadcrumbs(current_page)
    st.markdown(f"**Navigation:** {breadcrumbs}")
    st.markdown("---")

    # Use PageRouter to handle page rendering
    PageRouter.route_to_page(current_page, state)


if __name__ == "__main__":
    main()
