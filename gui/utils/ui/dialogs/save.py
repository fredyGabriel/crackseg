"""
Save dialog utilities for YAML configuration files. This module
provides advanced save functionality with Hydra validation, file
naming patterns, and comprehensive error handling.
"""

import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.errors import ConfigCompositionException

from gui.utils.config import validate_yaml_advanced
from gui.utils.session_state import SessionStateManager

logger = logging.getLogger(__name__)


class SaveDialogManager:
    """Advanced save dialog manager for YAML configuration files."""

    def __init__(self, project_root: Path | None = None) -> None:
        """
        Initialize save dialog manager. Args: project_root: Root directory of
        the project. If None, uses current directory.
        """
        self.project_root = project_root or Path.cwd()
        self.save_locations = [
            "generated_configs/",
            "configs/custom/",
            "configs/experiments/",
            "configs/user/",
        ]
        self.default_filename_pattern = "{timestamp}_{name}.yaml"

    def render_save_dialog(
        self,
        content: str,
        key: str,
        default_name: str = "config",
        show_advanced_options: bool = True,
    ) -> bool:
        """
        Render the save dialog UI component. Args: content: YAML content to
        save. key: Unique key for the dialog component. default_name: Default
        filename base. show_advanced_options: Whether to show advanced save
        options. Returns: True if file was saved successfully, False
        otherwise.
        """
        if not content.strip():
            st.warning("⚠️ No content to save")
            return False

        with st.expander("💾 Save Configuration", expanded=True):
            return self._render_save_form(
                content, key, default_name, show_advanced_options
            )

    def _render_save_form(
        self,
        content: str,
        key: str,
        default_name: str,
        show_advanced_options: bool,
    ) -> bool:
        """
        Render the main save form. Args: content: YAML content to save. key:
        Unique key for form components. default_name: Default filename base.
        show_advanced_options: Whether to show advanced options. Returns: True
        if file was saved successfully, False otherwise.
        """
        # Step 1: Validation
        validation_passed = self._render_validation_step(content, key)
        if not validation_passed:
            return False

        # Step 2: File naming and location
        filename, save_location = self._render_file_options(
            key, default_name, show_advanced_options
        )

        # Step 3: Preview and save
        return self._render_save_step(content, filename, save_location, key)

    def _render_validation_step(self, content: str, key: str) -> bool:
        """
        Render validation step with detailed feedback. Args: content: YAML
        content to validate. key: Unique key for validation components.
        Returns: True if validation passed, False otherwise.
        """
        st.markdown("**📋 Step 1: Validation**")

        # Basic YAML syntax validation
        is_syntax_valid, syntax_errors = validate_yaml_advanced(content)

        if not is_syntax_valid:
            st.error(f"❌ YAML has {len(syntax_errors)} syntax errors")
            with st.expander("Show syntax errors", expanded=True):
                for error in syntax_errors:
                    st.error(f"• {error}")
            return False

        st.success("✅ Valid YAML syntax")

        # Hydra validation (optional but recommended)
        hydra_valid, hydra_error = self._validate_with_hydra_compose(
            content, key
        )

        show_hydra_validation = st.checkbox(
            "Validate with Hydra (recommended)",
            value=True,
            key=f"{key}_hydra_validation",
            help="Checks that the configuration is compatible with Hydra",
        )

        if show_hydra_validation:
            if not hydra_valid:
                st.warning(f"⚠️ Hydra Validation: {hydra_error}")
                if not st.checkbox(
                    "Save anyway",
                    key=f"{key}_force_save",
                    help="Save file ignoring Hydra errors",
                ):
                    return False
            else:
                st.success("✅ Hydra-compatible configuration")

        return True

    def _render_file_options(
        self, key: str, default_name: str, show_advanced_options: bool
    ) -> tuple[str, str]:
        """
        Render file naming and location options. Args: key: Unique key for
        option components. default_name: Default filename base.
        show_advanced_options: Whether to show advanced options. Returns:
        Tuple of (filename, save_location).
        """
        st.markdown("**📁 Step 2: Name and Location**")

        col1, col2 = st.columns(2)

        with col1:
            # Filename pattern selection
            if show_advanced_options:
                filename_pattern = st.selectbox(
                    "Naming pattern:",
                    [
                        "{timestamp}_{name}.yaml",
                        "{name}_{timestamp}.yaml",
                        "{name}.yaml",
                        "Custom",
                    ],
                    key=f"{key}_filename_pattern",
                    help="Pattern to generate the file name",
                )
            else:
                filename_pattern = self.default_filename_pattern

            # Generate filename based on pattern
            if filename_pattern == "Custom":
                filename = st.text_input(
                    "File name:",
                    value=f"{default_name}.yaml",
                    key=f"{key}_custom_filename",
                )
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = filename_pattern.format(
                    timestamp=timestamp, name=default_name
                )
                st.text_input(
                    "File name:",
                    value=filename,
                    key=f"{key}_generated_filename",
                    disabled=True,
                )

        with col2:
            # Save location
            save_location = st.selectbox(
                "Save location:",
                self.save_locations,
                key=f"{key}_save_location",
                help="Directory where the file will be saved",
            )

            # Create custom location option
            if show_advanced_options and st.checkbox(
                "Custom location", key=f"{key}_custom_location"
            ):
                custom_location = st.text_input(
                    "Custom path:",
                    key=f"{key}_custom_path",
                    help="Relative path from the project root",
                )
                if custom_location:
                    save_location = custom_location

        # Preview full path
        full_path = self.project_root / save_location / filename
        st.markdown("**Preview:**")
        st.code(str(full_path), language="text")

        return filename, save_location

    def _render_save_step(
        self, content: str, filename: str, save_location: str, key: str
    ) -> bool:
        """
        Render the final save step with confirmation. Args: content: YAML
        content to save. filename: Target filename. save_location: Target
        directory. key: Unique key for save components. Returns: True if file
        was saved successfully, False otherwise.
        """
        st.markdown("**💾 Step 3: Save**")

        # File existence check
        full_path = self.project_root / save_location / filename
        file_exists = full_path.exists()

        if file_exists:
            st.warning(f"⚠️ The file '{filename}' already exists")
            if not st.checkbox(
                "Overwrite existing file",
                key=f"{key}_overwrite",
                help="Mark to replace the existing file",
            ):
                return False

        # Save button
        _, col2, _ = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "💾 Save File",
                key=f"{key}_save_button",
                use_container_width=True,
                type="primary",
            ):
                return self._save_file(content, full_path, key)

        return False

    def _save_file(self, content: str, file_path: Path, key: str) -> bool:
        """
        Save content to file with error handling. Args: content: YAML content
        to save. file_path: Target file path. key: Component key for status
        messages. Returns: True if saved successfully, False otherwise.
        """
        try:
            # Create directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Create backup if file exists
            if file_path.exists():
                backup_path = self._create_backup(file_path)
                st.info(f"📋 Backup created: {backup_path.name}")

            # Save file with atomic write
            self._atomic_write(content, file_path)

            # Success feedback
            st.success(f"✅ File saved successfully: {file_path}")

            # Update session state
            self._update_session_state(file_path)

            return True

        except Exception as e:
            logger.error(f"Error saving file {file_path}: {e}")
            st.error(f"❌ Error saving file: {str(e)}")
            return False

    def _validate_with_hydra_compose(
        self, content: str, key: str
    ) -> tuple[bool, str | None]:
        """
        Validate YAML content using Hydra compose dry-run. Args: content: YAML
        content to validate. key: Component key for temp files. Returns: Tuple
        of (is_valid, error_message).
        """
        temp_path: Path | None = None

        try:
            # Create temporary file for validation
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as temp_file:
                temp_file.write(content)
                temp_path = Path(temp_file.name)

            try:
                # Clear any existing Hydra instance
                GlobalHydra.instance().clear()

                # Try to compose the config
                with initialize_config_dir(
                    config_dir=str(temp_path.parent), version_base=None
                ):
                    compose(config_name=temp_path.stem)

                return True, None

            except ConfigCompositionException as e:
                return False, f"Composition error: {str(e)}"
            except Exception as e:
                return False, f"Validation error: {str(e)}"

        except Exception as e:
            logger.error(f"Hydra validation error: {e}")
            return False, f"Internal error: {str(e)}"

        finally:
            # Cleanup
            try:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
                GlobalHydra.instance().clear()
            except Exception:
                pass  # Ignore cleanup errors

    def _create_backup(self, file_path: Path) -> Path:
        """
        Create backup of existing file. Args: file_path: Path to file to
        backup. Returns: Path to backup file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
        backup_path = file_path.parent / backup_name

        shutil.copy2(file_path, backup_path)
        return backup_path

    def _atomic_write(self, content: str, file_path: Path) -> None:
        """
        Write content to file atomically. Args: content: Content to write.
        file_path: Target file path.
        """
        # Write to temporary file first
        temp_path = file_path.with_suffix(f"{file_path.suffix}.tmp")

        try:
            temp_path.write_text(content, encoding="utf-8")
            # Atomic move
            temp_path.replace(file_path)
        except Exception:
            # Cleanup on error
            temp_path.unlink(missing_ok=True)
            raise

    def _update_session_state(self, file_path: Path) -> None:
        """
        Update session state with save information. Args: file_path: Path to
        saved file.
        """
        try:
            state = SessionStateManager.get()
            state.add_notification(f"Config saved: {file_path.name}")

            # Update the config path if this is the current working config
            state.config_path = str(file_path)

        except Exception as e:
            logger.warning(f"Failed to update session state: {e}")
