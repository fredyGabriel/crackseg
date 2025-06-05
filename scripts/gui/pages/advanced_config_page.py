"""
Advanced configuration page with YAML editor for the CrackSeg application.

This module provides an advanced configuration interface with
Ace editor integration, live validation, and file management.
"""

from pathlib import Path

import streamlit as st
import yaml

from scripts.gui.components.config_editor_component import (
    ConfigEditorComponent,
)
from scripts.gui.utils.session_state import SessionStateManager


def page_advanced_config() -> None:
    """Advanced configuration page with YAML editor."""
    state = SessionStateManager.get()

    # Page header
    st.title("⚙️ Editor de Configuración Avanzado")
    st.markdown("**Editor YAML con validación en vivo y syntax highlighting**")

    # Initialize components
    editor_component = ConfigEditorComponent()

    # Create tabs for different functionalities
    tab_editor, tab_browser, tab_templates = st.tabs(
        ["📝 Editor YAML", "📁 Explorador", "📋 Templates"]
    )

    with tab_editor:
        st.markdown("### Editor de Configuración con Validación en Vivo")

        # Load initial content if available
        initial_content = ""
        if state.config_path and Path(state.config_path).exists():
            try:
                initial_content = Path(state.config_path).read_text(
                    encoding="utf-8"
                )
            except Exception as e:
                st.error(f"Error cargando configuración: {str(e)}")

        # Render the Ace editor
        editor_content = editor_component.render_editor(
            initial_content=initial_content,
            key="advanced_config_editor",
            height=500,
        )

        # Quick actions
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Validar", use_container_width=True):
                from scripts.gui.utils.config_io import validate_yaml_advanced

                is_valid, errors = validate_yaml_advanced(editor_content)

                if is_valid:
                    st.success("✅ Configuración válida")
                else:
                    st.error(f"❌ {len(errors)} errores encontrados")

        with col2:
            if st.button("💾 Guardar Rápido", use_container_width=True):
                if state.config_path:
                    try:
                        Path(state.config_path).write_text(
                            editor_content, encoding="utf-8"
                        )
                        st.success("✅ Guardado exitoso")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.warning("No hay archivo configurado")

        with col3:
            if st.button(
                "🏠 Aplicar como Principal", use_container_width=True
            ):
                try:
                    config_data = yaml.safe_load(editor_content)
                    if config_data:
                        state.config_data = config_data
                        state.config_loaded = True
                        st.success("✅ Configuración aplicada")
                    else:
                        st.warning("Configuración vacía")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    with tab_browser:
        st.markdown("### Explorador de Archivos de Configuración")

        # File browser integration
        editor_component.render_file_browser_integration("advanced_browser")

        # File upload
        st.markdown("---")
        st.subheader("📁 Subir Archivo")

        uploaded_file = st.file_uploader(
            "Selecciona archivo YAML:",
            type=["yaml", "yml"],
            key="config_upload",
        )

        if uploaded_file is not None:
            try:
                content = uploaded_file.read().decode("utf-8")
                # Validate before accepting
                yaml.safe_load(content)

                # Save to generated_configs
                save_path = Path("generated_configs") / uploaded_file.name
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_text(content, encoding="utf-8")

                st.success(f"✅ Archivo subido: {uploaded_file.name}")

                # Load into editor
                if st.button("📂 Cargar en Editor"):
                    st.session_state["advanced_config_editor"] = content
                    st.rerun()

            except yaml.YAMLError as e:
                st.error(f"❌ Archivo YAML inválido: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error procesando archivo: {str(e)}")

    with tab_templates:
        st.markdown("### Templates de Configuración")

        templates = {
            "U-Net Básico": """defaults:
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
            "SwinUNet Avanzado": """defaults:
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
  gradient_accumulation_steps: 4

data:
  batch_size: 4
  image_size: [224, 224]
""",
        }

        for template_name, template_content in templates.items():
            with st.expander(f"📋 {template_name}", expanded=False):
                st.code(template_content)

                if st.button(
                    f"📂 Cargar {template_name}",
                    key=f"load_template_{template_name.replace(' ', '_')}",
                    use_container_width=True,
                ):
                    st.session_state["advanced_config_editor"] = (
                        template_content
                    )
                    st.success(f"✅ Template '{template_name}' cargado")
                    st.rerun()

    # Status panel
    st.markdown("---")
    st.subheader("📊 Estado de la Configuración")

    col1, col2, col3 = st.columns(3)

    with col1:
        current_file = (
            Path(state.config_path).name if state.config_path else "Ninguno"
        )
        file_status = "✅ Cargado" if state.config_loaded else "❌ No cargado"
        st.metric("📄 Archivo Actual", current_file, file_status)

    with col2:
        editor_content = st.session_state.get("advanced_config_editor", "")
        content_length = len(editor_content)
        content_status = (
            "✅ Con contenido" if content_length > 0 else "❌ Vacío"
        )
        st.metric("📝 Editor", f"{content_length} caracteres", content_status)

    with col3:
        issues = state.validate()
        validation_status = (
            "✅ Válido" if not issues else f"❌ {len(issues)} problemas"
        )
        st.metric("🔍 Validación", "Sistema", validation_status)
