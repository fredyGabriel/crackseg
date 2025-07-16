"""
Theme management for the CrackSeg application.

This module provides comprehensive theme support including color schemes,
custom CSS generation, and theme switching utilities for Streamlit.
"""

from dataclasses import dataclass, field
from typing import Any

import streamlit as st


@dataclass
class ColorScheme:
    """Color scheme definition for themes."""

    # Background colors
    primary_bg: str
    secondary_bg: str
    sidebar_bg: str
    card_bg: str

    # Text colors
    primary_text: str
    secondary_text: str
    accent_text: str
    muted_text: str

    # UI elements
    border_color: str
    success_color: str
    warning_color: str
    error_color: str
    info_color: str

    # Interactive elements
    button_bg: str
    button_text: str
    button_hover: str
    link_color: str

    # Status indicators
    active_color: str
    inactive_color: str
    pending_color: str


@dataclass
class ThemeConfig:
    """Complete theme configuration."""

    name: str
    display_name: str
    description: str
    colors: ColorScheme
    logo_style: str = "default"
    custom_css: str = ""
    streamlit_theme: dict[str, Any] = field(default_factory=dict)


class ThemeManager:
    """Centralized theme management system."""

    # Predefined theme configurations
    _themes = {
        "dark": ThemeConfig(
            name="dark",
            display_name="🌙 Dark Mode",
            description="Professional dark theme optimized for extended use",
            colors=ColorScheme(
                # Background colors
                primary_bg="#0E1117",
                secondary_bg="#262730",
                sidebar_bg="#1E1E2E",
                card_bg="#2D2D2D",
                # Text colors
                primary_text="#FAFAFA",
                secondary_text="#E0E0E0",
                accent_text="#00D4FF",
                muted_text="#888888",
                # UI elements
                border_color="#404040",
                success_color="#00C851",
                warning_color="#FFB347",
                error_color="#FF4444",
                info_color="#007AFF",
                # Interactive elements
                button_bg="#007AFF",
                button_text="#FFFFFF",
                button_hover="#0056B3",
                link_color="#00D4FF",
                # Status indicators
                active_color="#00C851",
                inactive_color="#666666",
                pending_color="#FFB347",
            ),
            logo_style="default",
            streamlit_theme={
                "base": "dark",
                "primaryColor": "#007AFF",
                "backgroundColor": "#0E1117",
                "secondaryBackgroundColor": "#262730",
                "textColor": "#FAFAFA",
            },
        ),
        "light": ThemeConfig(
            name="light",
            display_name="☀️ Light Mode",
            description="Clean light theme with high contrast for clarity",
            colors=ColorScheme(
                # Background colors
                primary_bg="#FFFFFF",
                secondary_bg="#F8F9FA",
                sidebar_bg="#F5F5F5",
                card_bg="#FEFEFE",
                # Text colors
                primary_text="#1F1F1F",
                secondary_text="#333333",
                accent_text="#007AFF",
                muted_text="#666666",
                # UI elements
                border_color="#E0E0E0",
                success_color="#28A745",
                warning_color="#FFC107",
                error_color="#DC3545",
                info_color="#17A2B8",
                # Interactive elements
                button_bg="#007AFF",
                button_text="#FFFFFF",
                button_hover="#0056B3",
                link_color="#0066CC",
                # Status indicators
                active_color="#28A745",
                inactive_color="#6C757D",
                pending_color="#FFC107",
            ),
            logo_style="light",
            streamlit_theme={
                "base": "light",
                "primaryColor": "#007AFF",
                "backgroundColor": "#FFFFFF",
                "secondaryBackgroundColor": "#F8F9FA",
                "textColor": "#1F1F1F",
            },
        ),
        "auto": ThemeConfig(
            name="auto",
            display_name="🔄 Auto Mode",
            description="Automatically adapts to system preferences",
            colors=ColorScheme(
                # Will be set dynamically based on system
                primary_bg="#FFFFFF",
                secondary_bg="#F8F9FA",
                sidebar_bg="#F5F5F5",
                card_bg="#FEFEFE",
                primary_text="#1F1F1F",
                secondary_text="#333333",
                accent_text="#007AFF",
                muted_text="#666666",
                border_color="#E0E0E0",
                success_color="#28A745",
                warning_color="#FFC107",
                error_color="#DC3545",
                info_color="#17A2B8",
                button_bg="#007AFF",
                button_text="#FFFFFF",
                button_hover="#0056B3",
                link_color="#0066CC",
                active_color="#28A745",
                inactive_color="#6C757D",
                pending_color="#FFC107",
            ),
            logo_style="default",
            streamlit_theme={
                # Default to light, will be updated dynamically
                "base": "light"
            },
        ),
    }

    @staticmethod
    def get_available_themes() -> list[str]:
        """Get list of available theme names."""
        return list(ThemeManager._themes.keys())

    @staticmethod
    def get_theme_config(theme_name: str) -> ThemeConfig:
        """Get theme configuration by name."""
        return ThemeManager._themes.get(
            theme_name, ThemeManager._themes["dark"]
        )

    @staticmethod
    def get_theme_display_options() -> dict[str, str]:
        """Get theme display options for UI selectors."""
        return {
            config.name: config.display_name
            for config in ThemeManager._themes.values()
        }

    @staticmethod
    def get_current_theme_as_json() -> str:
        """Get the current theme configuration as a JSON string."""
        import json

        current_theme_name = ThemeManager.get_current_theme()
        theme_config = ThemeManager.get_theme_config(current_theme_name)
        # Convert dataclass to dict for serialization
        theme_dict = {
            "name": theme_config.name,
            "display_name": theme_config.display_name,
            "description": theme_config.description,
            "colors": theme_config.colors.__dict__,
            "logo_style": theme_config.logo_style,
            "custom_css": theme_config.custom_css,
            "streamlit_theme": theme_config.streamlit_theme,
        }
        return json.dumps(theme_dict, indent=2)

    @staticmethod
    def import_theme_from_json(json_string: str) -> bool:
        """Import a theme from a JSON string and apply it."""
        import json

        try:
            theme_data = json.loads(json_string)
            theme_name = theme_data.get("name", "custom_theme")

            # Create ThemeConfig from crackseg.data
            colors = ColorScheme(**theme_data["colors"])
            new_config = ThemeConfig(
                name=theme_name,
                display_name=theme_data.get("display_name", "Custom Theme"),
                description=theme_data.get(
                    "description", "A user-imported theme."
                ),
                colors=colors,
                logo_style=theme_data.get("logo_style", "default"),
                custom_css=theme_data.get("custom_css", ""),
                streamlit_theme=theme_data.get("streamlit_theme", {}),
            )

            # Add to themes and switch
            ThemeManager._themes[theme_name] = new_config
            ThemeManager.switch_theme(theme_name)
            return True
        except (json.JSONDecodeError, KeyError) as e:
            st.error(f"Error importing theme: Invalid JSON format. {e}")
            return False

    @staticmethod
    def update_current_theme_config(updates: dict[str, Any]) -> None:
        """Update the configuration of the current theme."""
        current_theme_name = ThemeManager.get_current_theme()
        if current_theme_name in ThemeManager._themes:
            theme_config = ThemeManager._themes[current_theme_name]
            for key, value in updates.items():
                if hasattr(theme_config, key):
                    setattr(theme_config, key, value)
            # Re-apply the theme to reflect changes
            ThemeManager.apply_theme(current_theme_name)

    @staticmethod
    def apply_theme(theme_name: str) -> ThemeConfig:
        """Apply theme and return the configuration."""
        from scripts.gui.utils.session_state import SessionStateManager

        theme_config = ThemeManager.get_theme_config(theme_name)

        # Update session state
        SessionStateManager.update({"theme": theme_name})

        # Apply custom CSS
        ThemeManager._inject_custom_css(theme_config)

        return theme_config

    @staticmethod
    def _inject_custom_css(theme_config: ThemeConfig) -> None:
        """Inject custom CSS for the theme."""
        colors = theme_config.colors

        # Generate comprehensive CSS
        css = f"""
        <style>
        /* Main app theming */
        .stApp {{
            background-color: {colors.primary_bg};
            color: {colors.primary_text};
        }}

        /* Sidebar theming */
        .css-1d391kg {{
            background-color: {colors.sidebar_bg} !important;
        }}

        /* Cards and containers */
        .element-container {{
            background-color: {colors.card_bg};
            border: 1px solid {colors.border_color};
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }}

        /* Text colors */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
            color: {colors.primary_text} !important;
        }}

        .stMarkdown p {{
            color: {colors.secondary_text} !important;
        }}

        /* Buttons */
        .stButton > button {{
            background-color: {colors.button_bg} !important;
            color: {colors.button_text} !important;
            border: none !important;
            border-radius: 6px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 600 !important;
        }}

        .stButton > button:hover {{
            background-color: {colors.button_hover} !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }}

        /* Status indicators */
        .status-active {{
            color: {colors.active_color} !important;
            font-weight: bold;
        }}

        .status-inactive {{
            color: {colors.inactive_color} !important;
        }}

        .status-pending {{
            color: {colors.pending_color} !important;
        }}

        /* Success, warning, error styling */
        .stSuccess {{
            background-color: {colors.success_color}20 !important;
            border-left: 4px solid {colors.success_color} !important;
        }}

        .stWarning {{
            background-color: {colors.warning_color}20 !important;
            border-left: 4px solid {colors.warning_color} !important;
        }}

        .stError {{
            background-color: {colors.error_color}20 !important;
            border-left: 4px solid {colors.error_color} !important;
        }}

        .stInfo {{
            background-color: {colors.info_color}20 !important;
            border-left: 4px solid {colors.info_color} !important;
        }}

        /* Progress bars */
        .stProgress > div > div > div {{
            background-color: {colors.accent_text} !important;
        }}

        /* Metrics */
        .metric-container {{
            background-color: {colors.card_bg};
            border: 1px solid {colors.border_color};
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }}

        /* Links */
        a, .stLink {{
            color: {colors.link_color} !important;
        }}

        /* Navigation items */
        .nav-item {{
            background-color: {colors.card_bg};
            border: 1px solid {colors.border_color};
            border-radius: 6px;
            margin: 4px 0;
            transition: all 0.2s ease;
        }}

        .nav-item:hover {{
            background-color: {colors.accent_text}10;
            border-color: {colors.accent_text};
        }}

        .nav-item.active {{
            background-color: {colors.accent_text}20;
            border-color: {colors.accent_text};
            border-width: 2px;
        }}

        /* Theme switcher */
        .theme-switcher {{
            background-color: {colors.card_bg};
            border: 1px solid {colors.border_color};
            border-radius: 8px;
            padding: 0.5rem;
            margin: 0.5rem 0;
        }}

        /* Custom scrollbar for dark theme */
        {
            (
                "::-webkit-scrollbar"
                if theme_config.name == "dark"
                else "/* Light scrollbar */"
            )
        } {{
            width: 8px;
        }}

        {
            (
                "::-webkit-scrollbar-track"
                if theme_config.name == "dark"
                else "/* Light scrollbar track */"
            )
        } {{
            background: {
            (
                colors.secondary_bg
                if theme_config.name == "dark"
                else colors.primary_bg
            )
        };
        }}

        {
            (
                "::-webkit-scrollbar-thumb"
                if theme_config.name == "dark"
                else "/* Light scrollbar thumb */"
            )
        } {{
            background: {colors.muted_text};
            border-radius: 4px;
        }}

        {
            (
                "::-webkit-scrollbar-thumb:hover"
                if theme_config.name == "dark"
                else "/* Light scrollbar thumb hover */"
            )
        } {{
            background: {colors.secondary_text};
        }}

        /* Additional custom styles */
        {theme_config.custom_css}
        </style>
        """

        # Inject the CSS
        st.markdown(css, unsafe_allow_html=True)

    @staticmethod
    def get_current_theme() -> str:
        """Get current theme name from session state."""
        from scripts.gui.utils.session_state import SessionStateManager

        state = SessionStateManager.get()
        return state.theme or "dark"

    @staticmethod
    def get_current_colors() -> ColorScheme:
        """Get current theme colors."""
        current_theme = ThemeManager.get_current_theme()
        return ThemeManager.get_theme_config(current_theme).colors

    @staticmethod
    def switch_theme(new_theme: str) -> bool:
        """Switch to a new theme."""
        if new_theme not in ThemeManager._themes:
            return False

        from scripts.gui.utils.session_state import SessionStateManager

        # Update session state
        SessionStateManager.update({"theme": new_theme})

        # Add notification
        state = SessionStateManager.get()
        theme_config = ThemeManager.get_theme_config(new_theme)
        state.add_notification(f"Switched to {theme_config.display_name}")

        return True

    @staticmethod
    def render_theme_selector(key: str = "theme_selector") -> str:
        """Render theme selector widget."""
        current_theme = ThemeManager.get_current_theme()
        theme_options = ThemeManager.get_theme_display_options()

        # Create reverse mapping for selectbox
        display_to_name = {v: k for k, v in theme_options.items()}

        selected_display = st.selectbox(
            "🎨 Theme",
            options=list(theme_options.values()),
            index=list(theme_options.keys()).index(current_theme),
            key=key,
            help="Choose your preferred color scheme",
        )

        selected_theme = display_to_name[selected_display]

        # Apply theme if changed
        if selected_theme != current_theme:
            ThemeManager.switch_theme(selected_theme)
            st.rerun()

        return selected_theme

    @staticmethod
    def render_theme_info() -> None:
        """Render current theme information."""
        current_theme = ThemeManager.get_current_theme()
        theme_config = ThemeManager.get_theme_config(current_theme)
        colors = theme_config.colors

        st.markdown("### 🎨 Current Theme")
        st.markdown(f"**{theme_config.display_name}**")
        st.caption(theme_config.description)

        # Color palette preview
        with st.expander("🎭 Color Palette", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Backgrounds**")
                st.color_picker("Primary", colors.primary_bg, disabled=True)
                st.color_picker(
                    "Secondary", colors.secondary_bg, disabled=True
                )
                st.color_picker("Sidebar", colors.sidebar_bg, disabled=True)

            with col2:
                st.markdown("**Text Colors**")
                st.color_picker("Primary", colors.primary_text, disabled=True)
                st.color_picker(
                    "Secondary", colors.secondary_text, disabled=True
                )
                st.color_picker("Accent", colors.accent_text, disabled=True)

            with col3:
                st.markdown("**Status Colors**")
                st.color_picker("Success", colors.success_color, disabled=True)
                st.color_picker("Warning", colors.warning_color, disabled=True)
                st.color_picker("Error", colors.error_color, disabled=True)
