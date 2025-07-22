#!/usr/bin/env python3
"""
Script para restaurar masivamente archivos corruptos usando git restore.
Evita problemas de PowerShell ejecutando comandos git de forma eficiente.
"""

import subprocess
import sys
from pathlib import Path


def restore_files_batch(
    files: list[str], batch_size: int = 10
) -> tuple[int, int]:
    """
    Restaura archivos en lotes usando git restore.

    Args:
        files: Lista de archivos a restaurar
        batch_size: Tamaño del lote

    Returns:
        Tuple de (exitosos, fallidos)
    """
    successful = 0
    failed = 0

    for i in range(0, len(files), batch_size):
        batch = files[i : i + batch_size]

        try:
            # Ejecutar git restore para este lote
            cmd = ["git", "restore"] + batch
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                successful += len(batch)
                print(f"✅ Restaurados {len(batch)} archivos: {batch[0]}...")
            else:
                failed += len(batch)
                print(f"❌ Error en lote: {result.stderr.strip()}")

        except subprocess.TimeoutExpired:
            failed += len(batch)
            print(f"⏰ Timeout en lote: {batch[0]}...")

        except Exception as e:
            failed += len(batch)
            print(f"💥 Error inesperado: {e}")

    return successful, failed


def main():
    """Función principal."""
    corrupted_files_path = Path("corrupted_files.txt")

    if not corrupted_files_path.exists():
        print("❌ No se encontró corrupted_files.txt")
        print("   Ejecuta primero: python scripts/debug/syntax_scanner.py")
        return 1

    # Leer lista de archivos corruptos
    with open(corrupted_files_path, encoding="utf-8") as f:
        files = [line.strip() for line in f if line.strip()]

    print("=" * 80)
    print("🔧 RESTAURACIÓN MASIVA DE ARCHIVOS CORRUPTOS")
    print("=" * 80)
    print(f"📁 Total de archivos a restaurar: {len(files)}")

    # Categorizar archivos por prioridad (usar \ para Windows)
    priority_dirs = ["src\\crackseg\\", "gui\\", "configs\\", "scripts\\"]

    priority_files = []
    other_files = []

    for file in files:
        if any(file.startswith(dir) for dir in priority_dirs):
            priority_files.append(file)
        else:
            other_files.append(file)

    print(f"🎯 Archivos prioritarios: {len(priority_files)}")
    print(f"📦 Otros archivos: {len(other_files)}")

    # Initialize counters
    successful_p = failed_p = successful_o = failed_o = 0

    # Restaurar archivos prioritarios primero
    if priority_files:
        print("\n🚀 FASE 1: Restaurando archivos prioritarios...")
        result_p = restore_files_batch(priority_files, batch_size=5)
        successful_p, failed_p = result_p[0], result_p[1]
        print(f"✅ Exitosos: {successful_p}, ❌ Fallidos: {failed_p}")

    # Restaurar otros archivos
    if other_files:
        print("\n📦 FASE 2: Restaurando otros archivos...")
        result_o = restore_files_batch(other_files, batch_size=10)
        successful_o, failed_o = result_o[0], result_o[1]
        print(f"✅ Exitosos: {successful_o}, ❌ Fallidos: {failed_o}")

    # Resumen final
    total_successful = successful_p + successful_o
    total_failed = failed_p + failed_o

    print("\n" + "=" * 80)
    print("📊 RESUMEN DE RESTAURACIÓN")
    print("=" * 80)
    print(f"✅ Total exitosos: {total_successful}")
    print(f"❌ Total fallidos: {total_failed}")
    print(f"📈 Tasa de éxito: {total_successful / len(files) * 100:.1f}%")

    if total_failed == 0:
        print("\n🎉 ¡Todos los archivos restaurados exitosamente!")
        print("📋 Ejecuta quality gates para verificar:")
        print("   python scripts/debug/syntax_scanner.py")
        return 0
    else:
        print(f"\n⚠️  {total_failed} archivos necesitan atención manual")
        return 1


if __name__ == "__main__":
    sys.exit(main())
