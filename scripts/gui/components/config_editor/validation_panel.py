"""
Advanced validation panel for YAML configuration editor.

This module provides detailed validation feedback, error parsing,
and configuration analysis functionality.
"""

import logging
from typing import Any

import streamlit as st
import yaml

from scripts.gui.utils.config import validate_yaml_advanced

logger = logging.getLogger(__name__)


class ValidationPanel:
    """Advanced validation panel for YAML configurations."""

    def render_advanced_validation(self, content: str, key: str) -> None:
        """Render enhanced live validation feedback for YAML content.

        Args:
            content: Current YAML content to validate
            key: Base key for the editor component
        """
        if not content.strip():
            st.info("💡 Escribe YAML para ver validación en tiempo real")
            return

        # Enhanced syntax validation with detailed feedback
        with st.container():
            st.markdown("**🔍 Validación de Sintaxis:**")
            try:
                yaml.safe_load(content)
                st.success("✅ Sintaxis YAML correcta")

                # Additional syntax quality checks
                warnings = self._check_yaml_quality(content)
                if warnings:
                    with st.expander(
                        "⚠️ Advertencias de formato", expanded=False
                    ):
                        for warning in warnings[:3]:  # Show first 3
                            st.warning(f"• {warning}")

            except yaml.YAMLError as e:
                st.error("❌ Error de sintaxis YAML")
                self._render_error_details(e, content)

        # Enhanced advanced validation
        with st.container():
            st.markdown("**🛡️ Validación Avanzada:**")
            is_valid, errors = validate_yaml_advanced(content)

            if is_valid:
                st.success("✅ Configuración válida para CrackSeg")
                self._render_config_metrics(content)
            else:
                st.error(f"❌ {len(errors)} problemas encontrados")
                self._render_categorized_errors(errors)

        # Enhanced configuration preview
        with st.container():
            st.markdown("**📋 Vista Previa Interactiva:**")
            self._render_config_preview(content)

    def _check_yaml_quality(self, content: str) -> list[str]:
        """Check YAML content for quality issues.

        Args:
            content: YAML content to check

        Returns:
            List of warning messages
        """
        warnings = []
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            # Check for common YAML issues
            if (
                line.strip()
                and line.strip().endswith(":")
                and not line.strip().startswith("#")
            ):
                if (
                    i < len(lines)
                    and lines[i].strip()
                    and not lines[i].startswith(" ")
                ):
                    warnings.append(
                        f"Línea {i}: Posible problema de indentación "
                        f"después de '{line.strip()}'"
                    )

            # Check for tabs (should use spaces)
            if "\t" in line:
                warnings.append(f"Línea {i}: Usa espacios en lugar de tabs")

        return warnings

    def _render_error_details(
        self, error: yaml.YAMLError, content: str
    ) -> None:
        """Render detailed error information with suggestions.

        Args:
            error: YAML error to analyze
            content: Original content for context
        """
        error_info = self._parse_yaml_error(error)

        if error_info["line"] and error_info["column"]:
            st.caption(
                f"📍 **Ubicación**: Línea {error_info['line']}, "
                f"Columna {error_info['column']}"
            )

        if error_info["suggestion"]:
            st.info(f"💡 **Sugerencia**: {error_info['suggestion']}")

        # Show context around the error
        if error_info["context"]:
            with st.expander("🔍 Contexto del error", expanded=True):
                st.code(error_info["context"], language="yaml")

    def _parse_yaml_error(
        self, error: yaml.YAMLError
    ) -> dict[str, str | int | None]:
        """Parse YAML error to extract useful information and suggestions.

        Args:
            error: YAML error to parse

        Returns:
            Dictionary with error information and suggestions
        """
        info: dict[str, str | int | None] = {
            "line": None,
            "column": None,
            "suggestion": None,
            "context": None,
        }

        try:
            problem_mark = getattr(error, "problem_mark", None)
            if problem_mark is not None:
                info["line"] = getattr(problem_mark, "line", 0) + 1
                info["column"] = getattr(problem_mark, "column", 0) + 1

            # Generate context around error
            if hasattr(error, "problem_mark") and problem_mark:
                try:
                    snippet = getattr(
                        problem_mark, "get_snippet", lambda: ""
                    )()
                    if snippet:
                        lines = snippet.split("\n")
                        info["context"] = "\n".join(
                            lines[:5]
                        )  # Show 5 lines context
                except Exception:
                    pass

            # Generate suggestions based on error type
            error_str = str(error).lower()
            if (
                "found character" in error_str
                and "that cannot start" in error_str
            ):
                info["suggestion"] = (
                    "Revisa caracteres especiales o comillas mal cerradas"
                )
            elif "mapping values are not allowed" in error_str:
                info["suggestion"] = "Verifica la indentación y el uso de ':'"
            elif "found undefined alias" in error_str:
                info["suggestion"] = (
                    "Alias YAML no definido, revisa referencias"
                )
            elif "expected" in error_str and "but found" in error_str:
                info["suggestion"] = (
                    "Problema de sintaxis, revisa brackets, comillas o "
                    "indentación"
                )
            else:
                info["suggestion"] = (
                    "Revisa la sintaxis YAML cerca de la línea indicada"
                )

        except Exception:
            pass

        return info

    def _render_config_metrics(self, content: str) -> None:
        """Render configuration metrics and statistics.

        Args:
            content: YAML content to analyze
        """
        try:
            config_data = yaml.safe_load(content)
            if config_data:
                metrics = self._calculate_config_metrics(config_data)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Secciones", metrics["sections"])
                with col2:
                    st.metric("Parámetros", metrics["parameters"])
                with col3:
                    st.metric("Profundidad", metrics["max_depth"])
        except Exception:
            pass

    def _calculate_config_metrics(
        self, config_data: dict[str, Any]
    ) -> dict[str, int]:
        """Calculate metrics about the configuration structure.

        Args:
            config_data: Parsed configuration data

        Returns:
            Dictionary with configuration metrics
        """

        def count_items(
            obj: dict[str, Any] | list[Any], depth: int = 0
        ) -> dict[str, int]:
            counts = {"sections": 0, "parameters": 0, "max_depth": depth}

            if isinstance(obj, dict):
                counts["sections"] += 1
                for _key, value in obj.items():
                    if isinstance(value, dict | list):
                        sub_counts = count_items(value, depth + 1)
                        counts["sections"] += sub_counts["sections"]
                        counts["parameters"] += sub_counts["parameters"]
                        counts["max_depth"] = max(
                            counts["max_depth"], sub_counts["max_depth"]
                        )
                    else:
                        counts["parameters"] += 1
            else:  # isinstance(obj, list)
                for item in obj:
                    if isinstance(item, dict | list):
                        sub_counts = count_items(item, depth + 1)
                        counts["sections"] += sub_counts["sections"]
                        counts["parameters"] += sub_counts["parameters"]
                        counts["max_depth"] = max(
                            counts["max_depth"], sub_counts["max_depth"]
                        )
                    else:
                        counts["parameters"] += 1

            return counts

        return count_items(config_data)

    def _render_categorized_errors(self, errors: list[Any]) -> None:
        """Render errors categorized by severity.

        Args:
            errors: List of validation errors
        """
        # Convert errors to strings first
        error_strings = [str(error) for error in errors]
        categorized_errors = self._categorize_validation_errors(error_strings)

        for severity, error_list in categorized_errors.items():
            if not error_list:
                continue

            severity_info = {
                "critical": ("🚨", "error", "Errores críticos"),
                "warning": ("⚠️", "warning", "Advertencias"),
                "info": ("ℹ️", "info", "Información"),
            }

            icon, msg_type, title = severity_info[severity]

            with st.expander(
                f"{icon} {title} ({len(error_list)})", expanded=True
            ):
                for error in error_list[:5]:  # Show first 5 per category
                    if msg_type == "error":
                        st.error(f"• {error}")
                    elif msg_type == "warning":
                        st.warning(f"• {error}")
                    else:
                        st.info(f"• {error}")

                if len(error_list) > 5:
                    st.caption(f"... y {len(error_list) - 5} más")

    def _categorize_validation_errors(
        self, errors: list[str]
    ) -> dict[str, list[str]]:
        """Categorize validation errors by severity.

        Args:
            errors: List of error messages

        Returns:
            Dictionary categorizing errors by severity
        """
        categorized = {"critical": [], "warning": [], "info": []}

        for error in errors:
            error_lower = error.lower()
            if any(
                keyword in error_lower
                for keyword in ["missing", "required", "invalid", "error"]
            ):
                categorized["critical"].append(error)
            elif any(
                keyword in error_lower
                for keyword in ["deprecated", "recommend", "should"]
            ):
                categorized["warning"].append(error)
            else:
                categorized["info"].append(error)

        return categorized

    def _render_config_preview(self, content: str) -> None:
        """Render interactive configuration preview.

        Args:
            content: YAML content to preview
        """
        try:
            config_data = yaml.safe_load(content)
            if config_data:
                # Tabs for different views
                tab1, tab2, tab3 = st.tabs(
                    ["🌳 Estructura", "📊 JSON", "🔍 Resumen"]
                )

                with tab1:
                    self._render_config_tree(config_data)

                with tab2:
                    st.json(config_data)

                with tab3:
                    self._render_config_summary(config_data)

            else:
                st.info("💡 Configuración vacía")
        except Exception as e:
            st.error(f"❌ Error al procesar configuración: {str(e)}")

    def _render_config_tree(self, config_data: dict[str, Any]) -> None:
        """Render configuration as an expandable tree structure.

        Args:
            config_data: Configuration data to render
        """

        def render_tree_node(
            obj: Any, prefix: str = "", key: str = "root"
        ) -> None:
            if isinstance(obj, dict):
                st.markdown(f"{prefix}📁 **{key}**")
                for k, v in obj.items():
                    if isinstance(v, dict | list):
                        render_tree_node(v, prefix + "  ", k)
                    else:
                        st.markdown(f"{prefix}  📄 {k}: `{v}`")
            else:  # isinstance(obj, list)
                st.markdown(
                    f"{prefix}📋 **{key}** (lista con {len(obj)} elementos)"
                )
                for i, item in enumerate(obj[:3]):  # Show first 3 items
                    if isinstance(item, dict | list):
                        render_tree_node(item, prefix + "  ", f"[{i}]")
                    else:
                        st.markdown(f"{prefix}  • `{item}`")
                if len(obj) > 3:
                    st.markdown(
                        f"{prefix}  ... y {len(obj) - 3} elementos más"
                    )

        render_tree_node(config_data)

    def _render_config_summary(self, config_data: dict[str, Any]) -> None:
        """Render a summary of the configuration.

        Args:
            config_data: Configuration data to summarize
        """
        # Main sections summary
        st.markdown("**📊 Resumen de Configuración:**")

        main_sections = list(config_data.keys())
        if main_sections:
            st.markdown(
                f"• **Secciones principales**: {', '.join(main_sections)}"
            )

        # Look for common CrackSeg sections
        common_sections = {
            "model": "🏗️ Modelo",
            "training": "🎯 Entrenamiento",
            "data": "📊 Datos",
            "experiment": "🔬 Experimento",
            "defaults": "⚙️ Configuraciones base",
        }

        found_sections = []
        for section, icon_name in common_sections.items():
            if section in config_data:
                found_sections.append(f"{icon_name}")

        if found_sections:
            st.markdown(
                f"• **Configuraciones detectadas**: "
                f"{', '.join(found_sections)}"
            )

        # Configuration completeness check
        completeness = len(
            [s for s in ["model", "training", "data"] if s in config_data]
        )
        total_expected = 3
        percentage = (completeness / total_expected) * 100

        st.progress(
            percentage / 100,
            text=f"Completitud: {percentage:.0f}% "
            f"({completeness}/{total_expected} secciones básicas)",
        )

        # Warnings or recommendations
        if "defaults" not in config_data:
            st.info(
                "💡 Considera agregar sección 'defaults' para "
                "configuración Hydra"
            )

        if (
            isinstance(config_data.get("model"), dict)
            and "_target_" not in config_data["model"]
        ):
            st.info(
                "💡 Considera agregar '_target_' en la sección del modelo "
                "para Hydra"
            )
