"""File upload component for YAML configuration files.

This module provides a reusable Streamlit component for uploading,
validating, and processing YAML configuration files.
"""

from pathlib import Path
from typing import Any

import streamlit as st

from gui.utils.config import (
    ConfigError,
    ValidationError,
    create_upload_progress_placeholder,
    get_upload_file_info,
    update_upload_progress,
    upload_config_file,
)
from gui.utils.config.validation.error_categorizer import (
    ErrorCategorizer,
)
from gui.utils.session_state import SessionStateManager


class FileUploadComponent:
    """Component for uploading and processing YAML configuration files."""

    @staticmethod
    def render_upload_section(
        title: str = "\ud83d\udce4 Upload Configuration File",
        help_text: str | None = None,
        target_directory: str = "generated_configs",
        key_suffix: str = "",
        show_validation: bool = True,
        show_preview: bool = True,
    ) -> tuple[str, dict[str, Any], list[ValidationError]] | None:
        """Render a complete file upload section with validation and feedback.

        Args:
            title: Section title to display.
            help_text: Optional help text to show.
            target_directory: Directory where uploaded files will be saved.
            key_suffix: Suffix for Streamlit component keys to avoid conflicts.
            show_validation: Whether to show validation results.
            show_preview: Whether to show file preview after upload.

        Returns:
            Tuple of (file_path, config_dict, validation_errors) if file
            uploaded, None otherwise.
        """
        with st.expander(title, expanded=False):
            # Help text
            if help_text:
                st.info(help_text)
            else:
                st.markdown(
                    """
                **Upload a YAML configuration file from your computer:**

                - Maximum file size: 10 MB
                - Supported formats: .yaml, .yml
                - Files will be validated automatically
                - Uploaded files are saved to `generated_configs/` with
                timestamp
                """
                )

            # File uploader
            uploaded_file = st.file_uploader(
                "Choose a YAML file",
                type=["yaml", "yml"],
                key=f"config_uploader{key_suffix}",
                help="Select a YAML configuration file from your computer",
            )

            if uploaded_file is not None:
                return FileUploadComponent._process_uploaded_file(
                    uploaded_file,
                    target_directory,
                    show_validation,
                    show_preview,
                    key_suffix,
                )

        return None

    @staticmethod
    def _process_uploaded_file(
        uploaded_file: Any,
        target_directory: str,
        show_validation: bool,
        show_preview: bool,
        key_suffix: str,
    ) -> tuple[str, dict[str, Any], list[ValidationError]] | None:
        """Process the uploaded file with progress indication."""
        # Get file information
        file_info = get_upload_file_info(uploaded_file)

        # Display file information
        st.markdown("### 📄 File Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("File Name", file_info["name"])
        with col2:
            st.metric("File Size", file_info["size_human"])
        with col3:
            extension_color = "🟢" if file_info["is_valid_extension"] else "🔴"
            st.metric(
                "Extension", f"{extension_color} {file_info['extension']}"
            )

        # Check for immediate issues
        issues = []
        if not file_info["is_valid_extension"]:
            issues.append(
                f"❌ Invalid file extension: {file_info['extension']}"
            )
        if not file_info["is_valid_size"]:
            issues.append(
                f"❌ File too large: {file_info['size_human']} "
                f"(max: {file_info['max_size_mb']} MB)"
            )

        if issues:
            st.error("**File validation failed:**")
            for issue in issues:
                st.write(issue)
            return None

        # Process file button
        if st.button(
            f"🚀 Process File: {file_info['name']}",
            use_container_width=True,
            key=f"process_file{key_suffix}",
        ):
            # Create progress placeholder
            progress_placeholder = create_upload_progress_placeholder()

            try:
                # Stage 1: Reading file
                update_upload_progress(
                    progress_placeholder, "reading", 0.1, "Loading file..."
                )

                # Stage 2: Validating
                update_upload_progress(
                    progress_placeholder,
                    "validating",
                    0.5,
                    "Checking YAML syntax...",
                )

                # Stage 3: Saving
                update_upload_progress(
                    progress_placeholder, "saving", 0.8, "Saving to disk..."
                )

                # Upload and process
                file_path, config_dict, validation_errors = upload_config_file(
                    uploaded_file,
                    target_directory=target_directory,
                    validate_on_upload=show_validation,
                )

                # Complete
                update_upload_progress(
                    progress_placeholder,
                    "complete",
                    1.0,
                    f"Saved as {Path(file_path).name}",
                )

                # Show validation results
                if show_validation and validation_errors:
                    FileUploadComponent._show_validation_results(
                        validation_errors
                    )

                # Show preview
                if show_preview:
                    FileUploadComponent._show_config_preview(config_dict)

                # Update session state
                SessionStateManager.update({"config_path": file_path})
                state = SessionStateManager.get()
                state.update_config(file_path, {"loaded": True})
                state.add_notification(
                    f"Configuration uploaded: {Path(file_path).name}"
                )

                st.success(
                    f"✅ File uploaded successfully: {Path(file_path).name}"
                )

                return file_path, config_dict, validation_errors

            except ConfigError as e:
                update_upload_progress(
                    progress_placeholder, "error", 0.0, str(e)
                )
                st.error(f"Upload failed: {e}")

            except Exception as e:
                update_upload_progress(
                    progress_placeholder,
                    "error",
                    0.0,
                    f"Unexpected error: {e}",
                )
                st.error(f"Unexpected error during upload: {e}")

        return None

    @staticmethod
    def _show_validation_results(
        validation_errors: list[ValidationError],
    ) -> None:
        """Show validation results in a user-friendly format."""
        if not validation_errors:
            st.success("\u2705 **Validation passed** - No issues found")
            return

        # Categorize errors using ErrorCategorizer
        categorizer = ErrorCategorizer()
        categorized_errors = categorizer.categorize_errors(validation_errors)
        errors = [
            e for e in categorized_errors if e.severity.value == "critical"
        ]
        warnings = [
            e for e in categorized_errors if e.severity.value == "warning"
        ]
        infos = [
            e
            for e in categorized_errors
            if e.severity.value in ("info", "suggestion")
        ]

        # Show summary
        st.markdown("### \ud83d\udd0d Validation Results")

        if errors:
            st.error(f"\u274c **{len(errors)} critical error(s) found**")
            with st.expander("Show errors", expanded=True):
                for error in errors:
                    st.write(f"**Line {error.line}:** {error.user_message}")

        if warnings:
            st.warning(f"\u26a0\ufe0f **{len(warnings)} warning(s) found**")
            with st.expander("Show warnings", expanded=False):
                for warning in warnings:
                    st.write(
                        f"**Line {warning.line}:** {warning.user_message}"
                    )

        if infos:
            st.info(f"\u2139\ufe0f **{len(infos)} suggestion(s)**")
            with st.expander("Show suggestions", expanded=False):
                for info in infos:
                    st.write(f"**Line {info.line}:** {info.user_message}")

        # Overall status
        if errors:
            st.error(
                "\ud83d\udeab **Configuration has critical issues** - "
                "Please fix errors before using"
            )
        elif warnings:
            st.warning(
                "\u26a1\ufe0f **Configuration usable with warnings** - "
                "Consider fixing warnings"
            )
        else:
            st.success("\u2705 **Configuration ready to use**")

    @staticmethod
    def _show_config_preview(config_dict: dict[str, Any]) -> None:
        """Show a preview of the uploaded configuration."""
        st.markdown("### 👁️ Configuration Preview")

        # Summary statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Top-level keys", len(config_dict.keys()))

        with col2:
            total_values = FileUploadComponent._count_values(config_dict)
            st.metric("Total values", total_values)

        with col3:
            max_depth = FileUploadComponent._calculate_depth(config_dict)
            st.metric("Max depth", max_depth)

        # Show structure
        with st.expander("📋 Configuration Structure", expanded=False):
            FileUploadComponent._show_config_structure(config_dict)

        # Show raw content (limited)
        with st.expander("📄 Raw Content (first 50 lines)", expanded=False):
            import yaml

            config_str = yaml.dump(config_dict, default_flow_style=False)
            lines = config_str.split("\n")
            limited_lines = lines[:50]

            if len(lines) > 50:
                limited_lines.append(f"... ({len(lines) - 50} more lines)")

            st.code("\n".join(limited_lines), language="yaml")

    @staticmethod
    def _count_values(obj: Any, count: int = 0) -> int:
        """Recursively count values in a nested dictionary."""
        if isinstance(obj, dict):
            for value in obj.values():
                count = FileUploadComponent._count_values(value, count)
        elif isinstance(obj, list):
            for item in obj:
                count = FileUploadComponent._count_values(item, count)
        else:
            count += 1
        return count

    @staticmethod
    def _calculate_depth(obj: Any, current_depth: int = 0) -> int:
        """Calculate the maximum depth of a nested dictionary."""
        if not isinstance(obj, dict):
            return current_depth

        if not obj:
            return current_depth

        return max(
            FileUploadComponent._calculate_depth(value, current_depth + 1)
            for value in obj.values()
        )

    @staticmethod
    def _show_config_structure(obj: Any, indent: int = 0) -> None:
        """Show the structure of the configuration in a tree-like format."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                prefix = "  " * indent + "├── " if indent > 0 else ""
                if isinstance(value, dict):
                    type_info = f"(dict) - {len(value)} keys"
                    st.write(f"{prefix}**{key}** {type_info}")
                    if indent < 2:
                        FileUploadComponent._show_config_structure(
                            value, indent + 1
                        )
                elif isinstance(value, list):
                    type_info = f"(list) - {len(value)} items"
                    st.write(f"{prefix}**{key}** {type_info}")
                    if indent < 2:
                        FileUploadComponent._show_config_structure(
                            value, indent + 1
                        )
                else:
                    value_preview = str(value)
                    if len(value_preview) > 50:
                        value_preview = value_preview[:47] + "..."
                    st.write(f"{prefix}**{key}**: `{value_preview}`")
        else:  # Assume obj is a list (already checked above)
            if obj:
                for i, item in enumerate(obj[:3]):  # Show first 3 items
                    prefix = "  " * indent + f"[{i}] "
                    if isinstance(item, dict | list):
                        type_info = f"({type(item).__name__})"
                        st.write(f"{prefix}{type_info}")
                        if indent < 2:
                            FileUploadComponent._show_config_structure(
                                item, indent + 1
                            )
                    else:
                        value_preview = str(item)
                        if len(value_preview) > 50:
                            value_preview = value_preview[:47] + "..."
                        st.write(f"{prefix}`{value_preview}`")
                if len(obj) > 3:
                    st.write(f"{'  ' * indent}... ({len(obj) - 3} more items)")


def render_upload_widget(
    title: str = "Upload Configuration",
    key: str = "default",
) -> str | None:
    """Render a simple upload widget that returns the uploaded file path.

    Args:
        title: Widget title.
        key: Unique key for the widget.

    Returns:
        Path to uploaded file if successful, None otherwise.
    """
    result = FileUploadComponent.render_upload_section(
        title=title,
        key_suffix=f"_{key}",
        show_validation=True,
        show_preview=False,
    )
    return result[0] if result else None


def render_detailed_upload(
    key: str = "detailed",
    target_dir: str = "generated_configs",
) -> tuple[str, dict[str, Any], list[ValidationError]] | None:
    """Render a detailed upload section with full validation and preview.

    Args:
        key: Unique key for the widget.
        target_dir: Target directory for uploaded files.

    Returns:
        Tuple of (file_path, config_dict, validation_errors) if successful,
        None otherwise.
    """
    return FileUploadComponent.render_upload_section(
        title="📤 Upload & Validate Configuration File",
        key_suffix=f"_{key}",
        target_directory=target_dir,
        show_validation=True,
        show_preview=True,
    )


__all__ = [
    "FileUploadComponent",
    "render_detailed_upload",
    "render_upload_widget",
]
