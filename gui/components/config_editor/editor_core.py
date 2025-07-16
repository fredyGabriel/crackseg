"""
Core configuration editor component for YAML editing.

This module provides the main editor rendering functionality
with Ace Editor integration and basic validation.
"""

import logging
from pathlib import Path

import streamlit as st
import yaml
from streamlit_ace import st_ace

from scripts.gui.utils.config import validate_yaml_advanced
from scripts.gui.utils.save_dialog import SaveDialogManager

logger = logging.getLogger(__name__)


class ConfigEditorCore:
    """Core YAML configuration editor functionality."""

    def __init__(self) -> None:
        """Initialize the configuration editor core."""
        self.save_dialog_manager = SaveDialogManager()

    def render_editor(
        self,
        initial_content: str = "",
        key: str = "config_editor",
        height: int = 400,
    ) -> str:
        """Render the Ace editor with YAML configuration.

        Args:
            initial_content: Initial YAML content for the editor
            key: Unique key for the editor component
            height: Editor height in pixels

        Returns:
            Current content of the editor
        """
        col_editor, col_validation = st.columns([2, 1])

        with col_editor:
            st.subheader("📝 YAML Editor")

            # Editor toolbar
            self._render_editor_toolbar(key)

            # Ace editor with YAML syntax highlighting
            content = st_ace(
                value=initial_content,
                language="yaml",
                theme="monokai",
                key=key,
                height=height,
                auto_update=True,
                font_size=14,
                tab_size=2,
                show_gutter=True,
                show_print_margin=True,
                wrap=False,
            )

        with col_validation:
            st.subheader("✅ Real-time Validation")
            self._render_basic_validation(content)

        return content

    def _render_editor_toolbar(self, key: str) -> None:
        """Render the editor toolbar with common actions.

        Args:
            key: Base key for the editor component
        """
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("📄 New", key=f"{key}_new", help="Create new file"):
                self._create_new_config(key)

        with col2:
            if st.button("📂 Load", key=f"{key}_load", help="Load file"):
                self._show_load_dialog(key)

        with col3:
            if st.button("💾 Save", key=f"{key}_save", help="Save file"):
                self._show_save_dialog(key)

        with col4:
            if st.button(
                "📋 Example", key=f"{key}_example", help="Load example"
            ):
                self._load_example_config(key)

    def _render_basic_validation(self, content: str) -> None:
        """Render basic syntax validation feedback.

        Args:
            content: Current YAML content to validate
        """
        if not content.strip():
            st.info("💡 Write YAML to see real-time validation")
            return

        # Basic syntax validation
        try:
            yaml.safe_load(content)
            st.success("✅ Correct YAML syntax")
        except yaml.YAMLError as e:
            st.error("❌ YAML syntax error")
            st.caption(f"Error: {str(e)}")

        # Advanced validation
        is_valid, errors = validate_yaml_advanced(content)
        if is_valid:
            st.success("✅ Configuration valid for CrackSeg")
        else:
            st.error(f"❌ {len(errors)} issues found")
            for error in errors[:3]:  # Show first 3 errors
                st.error(f"• {error}")

    def _create_new_config(self, key: str) -> None:
        """Create a new configuration from template.

        Args:
            key: Base key for the editor component
        """
        template_content = """# CrackSeg Configuration
defaults:
  - data: default
  - model: default
  - training: default

# Model configuration
model:
  _target_: src.model.UNet
  num_classes: 2

# Training configuration
training:
  epochs: 100
  optimizer:
    _target_: torch.optim.Adam
    lr: 0.001

# Data configuration
data:
  root_dir: data/
  batch_size: 16
  image_size: [512, 512]

# Experiment configuration
experiment:
  name: new_experiment
  random_seed: 42
"""
        st.session_state[key] = template_content
        st.success("✅ New file created with basic template")
        st.rerun()

    def _show_load_dialog(self, key: str) -> None:
        """Show dialog for loading configuration files.

        Args:
            key: Base key for the editor component
        """
        with st.expander("📂 Load Configuration File", expanded=True):
            file_path = st.text_input(
                "File path:",
                key=f"{key}_load_path",
                placeholder="configs/model/default.yaml",
            )

            if st.button(
                "📂 Load File",
                key=f"{key}_load_confirm",
                use_container_width=True,
            ):
                if file_path:
                    try:
                        config_path = Path(file_path)
                        if config_path.exists():
                            content = config_path.read_text(encoding="utf-8")
                            st.session_state[key] = content
                            st.success(f"✅ File loaded: {config_path.name}")
                            st.rerun()
                        else:
                            st.error(f"❌ File not found: {file_path}")
                    except Exception as e:
                        st.error(f"❌ Error loading file: {str(e)}")
                else:
                    st.error("⚠️ Please specify a file path")

    def _show_save_dialog(self, key: str) -> None:
        """Show dialog for saving configuration files.

        Args:
            key: Base key for the editor component
        """
        current_content = st.session_state.get(key, "")
        self.save_dialog_manager.render_save_dialog(
            content=current_content,
            key=f"{key}_save_dialog",
            default_name="config",
            show_advanced_options=True,
        )

    def _load_example_config(self, key: str) -> None:
        """Load an example configuration.

        Args:
            key: Base key for the editor component
        """
        examples = {
            "Basic U-Net": """defaults:
  - data: default
  - model: architectures/unet_cnn
  - training: default

experiment:
  name: unet_basic
  random_seed: 42

training:
  epochs: 50
  optimizer:
    lr: 0.001

data:
  batch_size: 8
""",
            "Advanced SwinUNet": """defaults:
  - data: default
  - model: architectures/unet_swin
  - training: default

experiment:
  name: swin_unet_advanced
  random_seed: 42

model:
  encoder:
    pretrained: true
    img_size: 224

training:
  epochs: 100
  use_amp: true

data:
  batch_size: 4
  image_size: [224, 224]
""",
        }

        with st.expander("📋 Load Example", expanded=True):
            example_choice = st.selectbox(
                "Select an example:",
                list(examples.keys()),
                key=f"{key}_example_choice",
            )

            if st.button(
                f"📋 Load '{example_choice}'",
                key=f"{key}_load_example",
                use_container_width=True,
            ):
                st.session_state[key] = examples[example_choice]
                st.success(f"✅ Example loaded: {example_choice}")
                st.rerun()
