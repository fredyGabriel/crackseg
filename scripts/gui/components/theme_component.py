"""
Theme component for the CrackSeg application.

This module provides UI components for theme selection and management
in the Streamlit interface.
"""

import streamlit as st

from scripts.gui.assets.manager import asset_manager
from scripts.gui.utils.theme import ThemeManager


class ThemeComponent:
    """Theme management UI component."""

    @staticmethod
    def render_theme_selector(
        location: str = "sidebar",
        show_info: bool = False,
        key: str = "theme_selector",
    ) -> str:
        """Render theme selector with enhanced UI.

        Args:
            location: Where to render ('sidebar', 'main', 'expander')
            show_info: Whether to show theme information
            key: Unique key for the selector

        Returns:
            Selected theme name
        """
        current_theme = ThemeManager.get_current_theme()

        if location == "expander":
            with st.expander("🎨 Theme Settings", expanded=False):
                return ThemeComponent._render_selector_content(
                    current_theme, show_info, key
                )
        else:
            return ThemeComponent._render_selector_content(
                current_theme, show_info, key
            )

    @staticmethod
    def _render_selector_content(
        current_theme: str, show_info: bool, key: str
    ) -> str:
        """Render the actual theme selector content."""
        theme_options = ThemeManager.get_theme_display_options()

        # Create reverse mapping for selectbox
        display_to_name = {v: k for k, v in theme_options.items()}

        # Theme selector
        st.markdown("### 🎨 Theme")

        selected_display = st.selectbox(
            "Choose your theme",
            options=list(theme_options.values()),
            index=list(theme_options.keys()).index(current_theme),
            key=key,
            help="Select your preferred color scheme",
            label_visibility="collapsed",
        )

        selected_theme = display_to_name[selected_display]

        # Apply theme if changed
        if selected_theme != current_theme:
            success = ThemeManager.switch_theme(selected_theme)
            if success:
                # Apply the theme CSS
                ThemeManager.apply_theme(selected_theme)
                st.success(f"Switched to {theme_options[selected_theme]}")
                st.rerun()
            else:
                st.error("Failed to switch theme")

        # Show theme info if requested
        if show_info:
            ThemeComponent.render_theme_preview(selected_theme)

        return selected_theme

    @staticmethod
    def render_theme_preview(theme_name: str) -> None:
        """Render a preview of the selected theme."""
        theme_config = ThemeManager.get_theme_config(theme_name)
        colors = theme_config.colors

        with st.expander("🔍 Theme Preview", expanded=False):
            st.markdown(f"**{theme_config.display_name}**")
            st.caption(theme_config.description)

            # Color samples
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Backgrounds**")
                ThemeComponent._render_color_sample(
                    "Primary", colors.primary_bg
                )
                ThemeComponent._render_color_sample(
                    "Secondary", colors.secondary_bg
                )
                ThemeComponent._render_color_sample("Card", colors.card_bg)

            with col2:
                st.markdown("**Text**")
                ThemeComponent._render_color_sample(
                    "Primary", colors.primary_text
                )
                ThemeComponent._render_color_sample(
                    "Secondary", colors.secondary_text
                )
                ThemeComponent._render_color_sample(
                    "Accent", colors.accent_text
                )

            with col3:
                st.markdown("**Status**")
                ThemeComponent._render_color_sample(
                    "Success", colors.success_color
                )
                ThemeComponent._render_color_sample(
                    "Warning", colors.warning_color
                )
                ThemeComponent._render_color_sample(
                    "Error", colors.error_color
                )

    @staticmethod
    def _render_color_sample(name: str, color: str) -> None:
        """Render a small color sample."""
        st.markdown(
            f"""
            <div style='
                display: flex;
                align-items: center;
                margin: 2px 0;
                font-size: 12px;
            '>
                <div style='
                    width: 20px;
                    height: 20px;
                    background-color: {color};
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    margin-right: 8px;
                '></div>
                <span>{name}</span>
                <span style='
                    margin-left: auto;
                    font-family: monospace;
                    color: #666;
                    font-size: 10px;
                '>{color}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_quick_theme_switcher() -> None:
        """Render a compact theme switcher for sidebar."""
        current_theme = ThemeManager.get_current_theme()

        st.markdown("#### 🎨 Theme")

        # Quick toggle buttons
        themes = ["dark", "light", "auto"]
        theme_icons = {"dark": "🌙", "light": "☀️", "auto": "🔄"}

        cols = st.columns(len(themes))

        for i, theme in enumerate(themes):
            with cols[i]:
                is_current = theme == current_theme
                button_type = "primary" if is_current else "secondary"

                if st.button(
                    f"{theme_icons[theme]}",
                    key=f"quick_theme_{theme}",
                    type=button_type,
                    use_container_width=True,
                    help=f"Switch to {theme} theme",
                    disabled=is_current,
                ):
                    if ThemeManager.switch_theme(theme):
                        ThemeManager.apply_theme(theme)
                        st.rerun()

    @staticmethod
    def apply_current_theme() -> None:
        """Apply the current theme's CSS to the interface."""
        current_theme = ThemeManager.get_current_theme()
        ThemeManager.apply_theme(current_theme)

        # Load base CSS assets
        asset_manager.inject_css("base_css")
        asset_manager.inject_css("navigation_css")

    @staticmethod
    def render_theme_status() -> None:
        """Render current theme status in a compact format."""
        current_theme = ThemeManager.get_current_theme()
        theme_config = ThemeManager.get_theme_config(current_theme)

        st.markdown(
            f"""
            <div style='
                display: flex;
                align-items: center;
                padding: 4px 8px;
                background-color: rgba(255,255,255,0.05);
                border-radius: 4px;
                margin: 4px 0;
                font-size: 12px;
            '>
                <span style='margin-right: 6px;'>🎨</span>
                <span style='font-weight: bold;'>
                    {theme_config.display_name}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_advanced_theme_settings() -> None:
        """Render advanced theme settings panel."""
        st.markdown("### 🔧 Advanced Theme Settings")

        current_theme = ThemeManager.get_current_theme()
        theme_config = ThemeManager.get_theme_config(current_theme)

        # Theme customization options
        with st.expander("⚙️ Customization Options", expanded=False):
            st.markdown("**Current Theme Configuration**")
            st.json(theme_config.streamlit_theme)

            # Logo style selection
            logo_styles = ["default", "light", "minimal"]
            current_logo_style = theme_config.logo_style

            selected_logo_style = st.selectbox(
                "Logo Style",
                options=logo_styles,
                index=logo_styles.index(current_logo_style),
                key="logo_style_selector",
                help="Choose the logo style for this theme",
            )

            if selected_logo_style != current_logo_style:
                st.info(
                    f"Logo style will be updated to '{selected_logo_style}' "
                    "on next theme application."
                )

            # Custom CSS input
            st.markdown("**Custom CSS**")
            st.text_area(
                "Additional CSS",
                value=theme_config.custom_css,
                height=100,
                key="custom_css_input",
                help="Add custom CSS rules for this theme",
            )

            if st.button("Apply Custom Settings", key="apply_custom_theme"):
                st.success("Custom theme settings would be applied here")
                st.info("This feature will be implemented in future updates")

        # Theme export/import
        with st.expander("📤 Export/Import Theme", expanded=False):
            # Export current theme
            if st.button("Export Current Theme", key="export_theme"):
                import json

                theme_data = {
                    "name": theme_config.name,
                    "display_name": theme_config.display_name,
                    "description": theme_config.description,
                    "colors": theme_config.colors.__dict__,
                    "logo_style": theme_config.logo_style,
                    "custom_css": theme_config.custom_css,
                    "streamlit_theme": theme_config.streamlit_theme,
                }

                st.download_button(
                    label="Download Theme JSON",
                    data=json.dumps(theme_data, indent=2),
                    file_name=f"crackseg_theme_{theme_config.name}.json",
                    mime="application/json",
                    key="download_theme",
                )

            # Import theme
            uploaded_theme = st.file_uploader(
                "Import Theme JSON",
                type="json",
                key="upload_theme",
                help="Upload a custom theme configuration file",
            )

            if uploaded_theme is not None:
                st.info(
                    "Theme import functionality will be implemented "
                    "in future updates"
                )
