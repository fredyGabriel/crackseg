"""
File browser integration for YAML configuration editor.

This module provides file discovery, loading, and browser functionality
for the configuration editor component.
"""

import logging
from datetime import datetime
from pathlib import Path

import streamlit as st

from scripts.gui.utils.session_state import SessionStateManager

logger = logging.getLogger(__name__)


class FileBrowserIntegration:
    """File browser integration for configuration files."""

    def render_file_browser(self, key: str = "config_browser") -> None:
        """Render file browser integration for configuration files.

        Args:
            key: Unique key for the file browser component
        """
        st.subheader("📁 Explorador de Configuraciones")

        # Scan for configuration files
        config_dirs = ["configs", "generated_configs"]
        all_configs = []

        for config_dir in config_dirs:
            config_path = Path(config_dir)
            if config_path.exists():
                yaml_files = list(config_path.rglob("*.yaml"))
                yml_files = list(config_path.rglob("*.yml"))
                all_configs.extend(yaml_files + yml_files)

        if not all_configs:
            st.info("💡 No se encontraron archivos de configuración")
            return

        # Group files by directory
        files_by_dir: dict[str, list[Path]] = {}
        for config_file in all_configs:
            dir_name = str(config_file.parent)
            if dir_name not in files_by_dir:
                files_by_dir[dir_name] = []
            files_by_dir[dir_name].append(config_file)

        # Display files by directory
        for dir_name, files in files_by_dir.items():
            with st.expander(f"📁 {dir_name}", expanded=True):
                for file_path in sorted(files):
                    self._render_file_item(file_path, key)

    def _render_file_item(self, file_path: Path, key: str) -> None:
        """Render individual file item with metadata and actions.

        Args:
            file_path: Path to the configuration file
            key: Base key for the browser component
        """
        col_file, col_actions = st.columns([3, 1])

        with col_file:
            st.write(f"📄 {file_path.name}")
            # Show file size and modification time
            try:
                stat = file_path.stat()
                size = stat.st_size
                mtime = datetime.fromtimestamp(stat.st_mtime)
                st.caption(
                    f"Tamaño: {size} bytes | Modificado: "
                    f"{mtime.strftime('%Y-%m-%d %H:%M')}"
                )
            except Exception:
                pass

        with col_actions:
            load_key = f"{key}_load_{str(file_path).replace('/', '_')}"
            if st.button(
                "📂 Cargar",
                key=load_key,
                use_container_width=True,
            ):
                self._load_config_file(file_path)

    def _load_config_file(self, file_path: Path) -> None:
        """Load configuration file into the editor.

        Args:
            file_path: Path to the configuration file
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            st.session_state["config_editor"] = content

            # Update session state
            state = SessionStateManager.get()
            state.update_config(str(file_path.absolute()), {"loaded": True})
            state.add_notification(f"Configuración cargada: {file_path.name}")

            st.success(f"✅ Cargado: {file_path.name}")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    def render_advanced_load_dialog(self, key: str) -> None:
        """Render advanced file loading dialog with common files.

        Args:
            key: Base key for the editor component
        """
        with st.expander("📂 Cargar Archivo de Configuración", expanded=True):
            file_path = st.text_input(
                "Ruta del archivo:",
                key=f"{key}_load_path",
                placeholder="configs/model/default.yaml",
            )

            col_load, col_browse = st.columns([3, 1])

            with col_load:
                if st.button(
                    "📂 Cargar Archivo",
                    key=f"{key}_load_confirm",
                    use_container_width=True,
                ):
                    if file_path:
                        self._load_file_by_path(file_path, key)
                    else:
                        st.error("⚠️ Por favor especifica una ruta de archivo")

            with col_browse:
                self._render_quick_load_buttons(key)

    def _load_file_by_path(self, file_path: str, key: str) -> None:
        """Load file by path with error handling.

        Args:
            file_path: Path to the file to load
            key: Base key for the editor component
        """
        try:
            config_path = Path(file_path)
            if config_path.exists():
                content = config_path.read_text(encoding="utf-8")
                st.session_state[key] = content

                # Update session state
                state = SessionStateManager.get()
                state.update_config(
                    str(config_path.absolute()),
                    {"loaded": True},
                )
                state.add_notification(f"Archivo cargado: {config_path.name}")

                st.success(f"✅ Archivo cargado: {config_path.name}")
                st.rerun()
            else:
                st.error(f"❌ Archivo no encontrado: {file_path}")
        except Exception as e:
            st.error(f"❌ Error cargando archivo: {str(e)}")

    def _render_quick_load_buttons(self, key: str) -> None:
        """Render quick load buttons for common files.

        Args:
            key: Base key for the editor component
        """
        st.markdown("**Archivos comunes:**")
        common_files = [
            "configs/model/default.yaml",
            "configs/training/base_training.yaml",
            "configs/complete_config.yaml",
        ]

        for file in common_files:
            if Path(file).exists():
                button_key = f"{key}_quick_{file.replace('/', '_')}"
                if st.button(
                    f"📄 {Path(file).name}",
                    key=button_key,
                    use_container_width=True,
                ):
                    self._load_file_by_path(file, key)
