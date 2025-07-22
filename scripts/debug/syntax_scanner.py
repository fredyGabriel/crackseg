#!/usr/bin/env python3
"""
Script para identificar archivos Python con errores críticos de sintaxis.
Escanea todo el proyecto y genera una lista de archivos que necesitan
restauración.
"""

import ast
import sys
from pathlib import Path


def check_syntax(file_path: Path) -> tuple[bool, str]:
    """
    Verifica la sintaxis de un archivo Python.

    Args:
        file_path: Ruta al archivo Python

    Returns:
        Tuple de (is_valid, error_message)
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Intentar parsear el contenido
        ast.parse(content)
        return True, ""

    except UnicodeDecodeError as e:
        return False, f"UnicodeDecodeError: {e}"
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    except Exception as e:
        return False, f"Other error: {e}"


def scan_project_syntax(project_root: Path) -> tuple[list[Path], list[Path]]:
    """
    Escanea todos los archivos Python en el proyecto.

    Args:
        project_root: Directorio raíz del proyecto

    Returns:
        Tuple de (archivos_válidos, archivos_corruptos)
    """
    valid_files = []
    corrupted_files = []

    # Buscar todos los archivos .py
    python_files = list(project_root.rglob("*.py"))

    print(f"🔍 Escaneando {len(python_files)} archivos Python...")

    for file_path in python_files:
        # Omitir archivos en directorios específicos
        skip_dirs = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            "node_modules",
            ".venv",
            "venv",
        }
        if any(part in skip_dirs for part in file_path.parts):
            continue

        is_valid, error = check_syntax(file_path)

        if is_valid:
            valid_files.append(file_path)
            print(f"✅ {file_path.relative_to(project_root)}")
        else:
            corrupted_files.append(file_path)
            print(f"❌ {file_path.relative_to(project_root)} - {error}")

    return valid_files, corrupted_files


def main():
    """Función principal del scanner."""
    project_root = Path.cwd()

    print("=" * 80)
    print("🔍 ESCÁNER DE SINTAXIS - PROYECTO CRACKSEG")
    print("=" * 80)

    valid_files, corrupted_files = scan_project_syntax(project_root)

    print("\n" + "=" * 80)
    print("📊 RESUMEN DEL ESCANEO")
    print("=" * 80)
    print(f"✅ Archivos válidos: {len(valid_files)}")
    print(f"❌ Archivos corruptos: {len(corrupted_files)}")
    print(f"📁 Total escaneados: {len(valid_files) + len(corrupted_files)}")

    if corrupted_files:
        print("\n🚨 ARCHIVOS CON ERRORES CRÍTICOS:")
        print("-" * 50)

        # Generar archivo de lista para git restore
        corrupted_list_file = project_root / "corrupted_files.txt"
        with open(corrupted_list_file, "w", encoding="utf-8") as f:
            for file_path in corrupted_files:
                rel_path = file_path.relative_to(project_root)
                print(f"   {rel_path}")
                f.write(f"{rel_path}\n")

        print(f"\n💾 Lista guardada en: {corrupted_list_file}")
        print("\n🔧 Para restaurar estos archivos, ejecuta:")
        files_to_restore = " ".join(
            str(f.relative_to(project_root)) for f in corrupted_files[:5]
        )
        print(f"   git restore --source=HEAD {files_to_restore}...")

        # Categorizar por directorio
        by_directory = {}
        for file_path in corrupted_files:
            dir_name = str(file_path.relative_to(project_root).parent)
            if dir_name not in by_directory:
                by_directory[dir_name] = []
            by_directory[dir_name].append(file_path)

        print("\n📂 DISTRIBUCIÓN POR DIRECTORIO:")
        print("-" * 50)
        for dir_name, files in sorted(by_directory.items()):
            print(f"   {dir_name}: {len(files)} archivos")

    else:
        print("\n🎉 ¡Todos los archivos tienen sintaxis válida!")

    return 0 if not corrupted_files else 1


if __name__ == "__main__":
    sys.exit(main())
