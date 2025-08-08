#!/usr/bin/env python3
"""
Script para generar automáticamente el manual de usuario de CrackSeg v0.2.0

Este script compila el documento LaTeX y genera el PDF del manual de usuario.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


class ManualGenerator:
    """Generador de manual de usuario para CrackSeg."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.manual_dir = self.project_root / "docs" / "user-manual"
        self.tex_file = self.manual_dir / "manual_usuario.tex"
        self.pdf_file = self.manual_dir / "manual_usuario.pdf"

    def check_dependencies(self) -> bool:
        """Verificar que las dependencias de LaTeX estén instaladas."""
        print("🔍 Verificando dependencias...")

        # Verificar XeLaTeX
        if not shutil.which("xelatex"):
            print("❌ Error: XeLaTeX no está instalado")
            print("💡 Instala una distribución LaTeX completa:")
            print("   - TeX Live: https://www.tug.org/texlive/")
            print("   - MiKTeX: https://miktex.org/")
            return False

        print("✅ XeLaTeX encontrado")
        return True

    def compile_manual(self, clean: bool = True) -> tuple[bool, str]:
        """Compilar el manual de usuario."""

        if not self.tex_file.exists():
            return False, f"❌ Archivo LaTeX no encontrado: {self.tex_file}"

        print(f"🔄 Compilando manual desde: {self.tex_file}")

        # Cambiar al directorio del manual
        original_dir = os.getcwd()
        os.chdir(self.manual_dir)

        try:
            # Primera pasada
            print("📝 Primera pasada de compilación...")
            result = subprocess.run(
                ["xelatex", "-interaction=nonstopmode", "manual_usuario.tex"],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                return False, f"❌ Error en primera pasada:\n{result.stderr}"

            # Segunda pasada para referencias
            print("📝 Segunda pasada para referencias...")
            result = subprocess.run(
                ["xelatex", "-interaction=nonstopmode", "manual_usuario.tex"],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                return False, f"❌ Error en segunda pasada:\n{result.stderr}"

            # Verificar que el PDF se generó
            if not self.pdf_file.exists():
                return False, "❌ PDF no se generó correctamente"

            # Limpiar archivos temporales si se solicita
            if clean:
                self.clean_temp_files()

            return True, f"✅ Manual compilado exitosamente: {self.pdf_file}"

        except Exception as e:
            return False, f"❌ Error durante la compilación: {str(e)}"
        finally:
            os.chdir(original_dir)

    def clean_temp_files(self):
        """Limpiar archivos temporales de LaTeX."""
        print("🧹 Limpiando archivos temporales...")

        temp_extensions = [
            "*.aux",
            "*.log",
            "*.out",
            "*.toc",
            "*.lof",
            "*.lot",
            "*.fls",
            "*.fdb_latexmk",
            "*.synctex.gz",
        ]

        for ext in temp_extensions:
            for file in self.manual_dir.glob(ext):
                try:
                    file.unlink()
                except Exception as e:
                    print(f"⚠️ No se pudo eliminar {file}: {e}")

    def get_pdf_info(self) -> dict | None:
        """Obtener información del PDF generado."""
        if not self.pdf_file.exists():
            return None

        stat = self.pdf_file.stat()
        return {
            "size_mb": stat.st_size / (1024 * 1024),
            "modified": stat.st_mtime,
            "path": str(self.pdf_file),
        }

    def print_stats(self):
        """Imprimir estadísticas del documento."""
        if not self.tex_file.exists():
            print("❌ Archivo LaTeX no encontrado")
            return

        print("📊 Estadísticas del manual:")

        # Contar líneas
        with open(self.tex_file, encoding="utf-8") as f:
            lines = f.readlines()
            total_lines = len(lines)
            code_lines = len(
                [line for line in lines if not line.strip().startswith("%")]
            )

        # Contar palabras (aproximado)
        content = " ".join(lines)
        words = len(content.split())

        # Contar emojis
        emojis = [
            "🎯",
            "🧠",
            "📖",
            "🎪",
            "🚀",
            "🛠️",
            "📋",
            "🔧",
            "📚",
            "🎮",
            "📊",
            "🎨",
            "📞",
            "🌟",
            "✅",
            "❌",
            "⚠️",
            "🔄",
            "📝",
        ]
        emoji_count = sum(content.count(emoji) for emoji in emojis)

        print(f"   📄 Líneas totales: {total_lines}")
        print(f"   📝 Líneas de código: {code_lines}")
        print(f"   📚 Palabras aproximadas: {words:,}")
        print(f"   😊 Emojis utilizados: {emoji_count}")

        # Información del PDF
        pdf_info = self.get_pdf_info()
        if pdf_info:
            print(f"   📄 Tamaño del PDF: {pdf_info['size_mb']:.1f} MB")

    def run(self, clean: bool = True, stats: bool = False) -> bool:
        """Ejecutar el generador de manual."""

        print("🧠 Generador de Manual de Usuario CrackSeg v0.2.0")
        print("=" * 50)

        # Verificar dependencias
        if not self.check_dependencies():
            return False

        # Compilar manual
        success, message = self.compile_manual(clean)
        print(message)

        if not success:
            return False

        # Mostrar estadísticas si se solicita
        if stats:
            print()
            self.print_stats()

        # Información final
        pdf_info = self.get_pdf_info()
        if pdf_info:
            print("\n🎉 Manual generado exitosamente!")
            print(f"📄 Archivo: {pdf_info['path']}")
            print(f"📏 Tamaño: {pdf_info['size_mb']:.1f} MB")

        return True


def main():
    """Función principal del script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generar manual de usuario de CrackSeg v0.2.0"
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="No limpiar archivos temporales",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Mostrar estadísticas del documento",
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Solo verificar dependencias"
    )

    args = parser.parse_args()

    generator = ManualGenerator()

    if args.check_only:
        success = generator.check_dependencies()
        sys.exit(0 if success else 1)

    success = generator.run(clean=not args.no_clean, stats=args.stats)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
