# scripts/

Esta carpeta contiene scripts auxiliares, experimentales y de ejemplo para el proyecto. La organización está pensada para facilitar la localización y el mantenimiento de los scripts según su propósito.

## Estructura

- **experiments/**
  - Scripts de experimentación, pruebas de modelos, benchmarks y prototipos.
  - Ejemplo: `test_swin_encoder.py`, `benchmark_aspp.py`

- **utils/**
  - Utilidades y herramientas para el workspace o el proyecto.
  - Ejemplo: `clean_workspace.py`, `model_summary.py`

- **reports/**
  - Reportes generados, archivos de análisis, PRD de ejemplo y documentación auxiliar.
  - Ejemplo: `task-complexity-report.json`, `prd.txt`, `example_prd.txt`

- **examples/**
  - Ejemplos de integración, uso de APIs y demostraciones.
  - Ejemplo: `factory_registry_integration.py`

## Notas

- Los scripts aquí NO forman parte del core del proyecto y no deben ser importados directamente por el código principal.
- La carpeta `__pycache__` y archivos temporales deben ser eliminados regularmente.
- Si agregas un nuevo script, colócalo en la subcarpeta correspondiente y actualiza este README si es necesario.

---

_Esta organización busca mantener el repositorio limpio, profesional y fácil de navegar para todos los colaboradores._ 