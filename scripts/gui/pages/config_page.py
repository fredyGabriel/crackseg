"""
Configuration page for the CrackSeg application.

This module contains the configuration page content and functionality
for loading model configurations and setting up the training environment.
"""

from pathlib import Path

import streamlit as st

from scripts.gui.components.theme_component import ThemeComponent
from scripts.gui.utils.session_state import SessionStateManager


def page_config() -> None:
    """Configuration page content."""
    state = SessionStateManager.get()

    # Getting started section
    if not state.config_loaded and not state.run_directory:
        st.info(
            "👋 **Bienvenido a CrackSeg!** Para comenzar, necesitas "
            "configurar los archivos de configuración y directorio de trabajo."
        )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Model Configuration")

        # Show current status
        if state.config_loaded:
            st.success("✅ Configuración cargada")
            if state.config_path:
                st.caption(f"Archivo: {Path(state.config_path).name}")
        else:
            st.warning("⚠️ Configuración no cargada")

        # Help section
        with st.expander(
            "❓ ¿Qué archivo necesito?", expanded=not state.config_loaded
        ):
            st.markdown(
                """
            **Necesitas un archivo de configuración YAML con:**
            - Parámetros del modelo (arquitectura, encoder, decoder)
            - Configuración de entrenamiento (learning rate, epochs, etc.)
            - Rutas de datos y configuración de loss functions

            **Ejemplos de ubicaciones:**
            - `configs/model/default.yaml`
            - `configs/training/experiment_1.yaml`
            - `configs/complete_config.yaml`
            """
            )

            # Example configurations
            example_configs = [
                "configs/model/default.yaml",
                "configs/training/base_training.yaml",
                "configs/complete_config.yaml",
            ]

            st.markdown("**Configuraciones de ejemplo:**")
            for example in example_configs:
                example_path = Path(example)
                if example_path.exists():
                    if st.button(
                        f"📄 Usar {example}",
                        key=f"use_{example.replace('/', '_')}",
                    ):
                        SessionStateManager.update(
                            {"config_path": str(example_path.absolute())}
                        )
                        state.update_config(
                            str(example_path.absolute()), {"loaded": True}
                        )
                        state.add_notification(
                            f"Configuración cargada: {example}"
                        )
                        st.rerun()
                else:
                    st.caption(f"⚪ {example} (no existe)")

        config_input = st.text_input(
            "Ruta del Archivo de Configuración",
            key="config_input",
            value=state.config_path or "",
            help="Ruta al archivo YAML de configuración",
            placeholder="configs/model/default.yaml",
        )

        col_load, col_browse = st.columns([3, 1])
        with col_load:
            if st.button("📂 Cargar Configuración", use_container_width=True):
                if config_input:
                    try:
                        # Validate path exists
                        config_path = Path(config_input)
                        if config_path.exists():
                            state.update_config(
                                str(config_path.absolute()), {"loaded": True}
                            )
                            state.add_notification(
                                f"Configuración cargada: {config_input}"
                            )
                            st.success(
                                f"✅ Configuración cargada: {config_path.name}"
                            )
                            st.rerun()
                        else:
                            st.error(
                                f"❌ El archivo no existe: {config_input}"
                            )
                    except Exception as e:
                        st.error(f"❌ Error cargando configuración: {e}")
                else:
                    st.error("⚠️ Por favor proporciona una ruta de archivo")

        with col_browse:
            if st.button(
                "🔍", help="Explorar archivos", use_container_width=True
            ):
                st.info(
                    "💡 Usa el campo de texto arriba para escribir "
                    "la ruta del archivo"
                )

    with col2:
        st.subheader("📁 Output Settings")

        # Show current status
        if state.run_directory:
            st.success("✅ Directorio configurado")
            st.caption(f"Directorio: {Path(state.run_directory).name}")
        else:
            st.warning("⚠️ Directorio no configurado")

        # Help section
        with st.expander(
            "❓ ¿Qué directorio necesito?", expanded=not state.run_directory
        ):
            st.markdown(
                """
            **El directorio de trabajo guardará:**
            - Modelos entrenados y checkpoints
            - Logs de entrenamiento y métricas
            - Resultados de evaluación
            - Visualizaciones y reportes

            **Ejemplos sugeridos:**
            - `outputs/experiment_1/`
            - `runs/crack_segmentation/`
            - `results/model_v1/`
            """
            )

            # Quick setup buttons
            suggested_dirs = [
                "outputs/experiment_1",
                "runs/current_run",
                "results/latest",
            ]

            st.markdown("**Directorios sugeridos:**")
            for suggested in suggested_dirs:
                if st.button(
                    f"📁 Crear {suggested}",
                    key=f"create_{suggested.replace('/', '_')}",
                ):
                    try:
                        dir_path = Path(suggested)
                        dir_path.mkdir(parents=True, exist_ok=True)
                        SessionStateManager.update(
                            {"run_directory": str(dir_path.absolute())}
                        )
                        state.add_notification(
                            f"Directorio creado: {suggested}"
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error creando directorio: {e}")

        run_dir_input = st.text_input(
            "Directorio de Trabajo",
            key="run_dir_input",
            value=state.run_directory or "",
            help="Directorio donde se guardarán los resultados",
            placeholder="outputs/experiment_1",
        )

        if st.button("📁 Establecer Directorio", use_container_width=True):
            if run_dir_input:
                try:
                    # Create directory if it doesn't exist
                    run_dir = Path(run_dir_input)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    SessionStateManager.update(
                        {"run_directory": str(run_dir.absolute())}
                    )
                    state.add_notification(
                        f"Directorio establecido: {run_dir_input}"
                    )
                    st.success(f"✅ Directorio establecido: {run_dir.name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error estableciendo directorio: {e}")
            else:
                st.error("⚠️ Por favor proporciona una ruta de directorio")

    # Quick setup section
    if not state.config_loaded or not state.run_directory:
        st.markdown("---")
        st.subheader("🚀 Configuración Rápida")

        if st.button(
            "⚡ Configuración Automática (Demo)", use_container_width=True
        ):
            try:
                # Create demo configuration
                demo_config = Path("configs/model")
                demo_config.mkdir(parents=True, exist_ok=True)

                demo_run = Path("outputs/demo_run")
                demo_run.mkdir(parents=True, exist_ok=True)

                # Set demo paths
                SessionStateManager.update(
                    {"run_directory": str(demo_run.absolute())}
                )

                # Add notification for demo setup
                state.add_notification("Configuración demo lista")
                st.success("✅ Configuración demo establecida")
                st.info(
                    "💡 Puedes modificar estos valores según tus necesidades"
                )
                st.rerun()

            except Exception as e:
                st.error(f"Error en configuración automática: {e}")

    # Configuration preview
    st.markdown("---")
    st.subheader("🔍 Configuration Preview")

    if state.config_loaded:
        st.success("✅ Configuración cargada exitosamente")
        if state.config_data:
            with st.expander("Ver configuración completa", expanded=False):
                st.json(state.config_data)
        else:
            st.info(
                "💡 Vista previa disponible después de cargar "
                "archivo de configuración"
            )
    else:
        st.info("📋 Carga una configuración para ver la vista previa")

    # Next steps guidance
    if state.config_loaded and state.run_directory:
        st.markdown("---")
        st.success("🎉 **¡Configuración completa!** Ahora puedes:")
        col_next1, col_next2, col_next3 = st.columns(3)

        with col_next1:
            if st.button("🏗️ Ver Arquitectura", use_container_width=True):
                SessionStateManager.update({"current_page": "Architecture"})
                st.rerun()

        with col_next2:
            if st.button("🚀 Iniciar Entrenamiento", use_container_width=True):
                SessionStateManager.update({"current_page": "Train"})
                st.rerun()

        with col_next3:
            if st.button("📊 Ver Resultados", use_container_width=True):
                SessionStateManager.update({"current_page": "Results"})
                st.rerun()

    # Validation status
    st.markdown("---")
    st.subheader("✅ Estado de Validación")
    issues = state.validate()
    if issues:
        st.error("⚠️ **Problemas encontrados:**")
        for issue in issues:
            st.warning(f"• {issue}")
        st.info("💡 Resuelve estos problemas para continuar")
    else:
        st.success("✅ **Todas las validaciones pasaron** - Sistema listo")

    # Theme configuration
    st.markdown("---")
    st.subheader("🎨 Configuración de Tema")

    col_theme1, col_theme2 = st.columns(2)

    with col_theme1:
        # Theme selector
        ThemeComponent.render_theme_selector(
            location="main", show_info=False, key="config_theme_selector"
        )

        # Theme status
        ThemeComponent.render_theme_status()

    with col_theme2:
        # Advanced theme settings
        with st.expander("🔧 Configuración Avanzada de Tema", expanded=False):
            ThemeComponent.render_advanced_theme_settings()
