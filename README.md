# Pavement Crack Segmentation Project

## Project Overview

Este proyecto busca lograr un desempeño de vanguardia en segmentación de grietas en pavimentos usando arquitecturas modulares basadas en U-Net. Está diseñado para investigadores, ingenieros y profesionales interesados en inspección automatizada de carreteras y monitoreo de infraestructura.

## Project Structure

- `src/` — Main source code (models, data pipeline, training, utils)
- `tests/` — Unit and integration tests
- `configs/` — Hydra YAML configuration files
- `scripts/` — Scripts auxiliares y de utilidad, organizados en subcarpetas:
  - `experiments/` — Scripts de experimentación, benchmarks y pruebas de modelos.
  - `utils/` — Utilidades y herramientas para el workspace/proyecto.
  - `reports/` — Reportes generados, archivos de análisis, PRD de ejemplo y documentación auxiliar.
  - `examples/` — Ejemplos de integración, uso de APIs y demostraciones.
- `tasks/` — TaskMaster task files

> Nota: Los scripts en `scripts/` no forman parte del core del proyecto y no deben ser importados directamente por el código principal. Elimina regularmente archivos temporales como `__pycache__`.

## Basic Usage

After setting up the environment and configuration files, you can run the main training pipeline (example):

```bash
python src/main.py
```

Para más detalles sobre configuración y uso avanzado, consulta la documentación en la carpeta `docs/` (si está disponible) o los comentarios en los archivos de configuración.

## Training Flow

El proceso de entrenamiento es totalmente modular y está gestionado por la clase `Trainer`. El script principal (`src/main.py`) delega toda la lógica de entrenamiento a esta clase, asegurando una separación clara de responsabilidades y un mantenimiento más sencillo.

**Características clave del flujo de entrenamiento:**
- **Orquestación basada en Trainer:** Toda la lógica de entrenamiento, validación y checkpointing está en `src/training/trainer.py`.
- **Checkpointing:** Los mejores y últimos estados del modelo se guardan automáticamente. Puedes reanudar el entrenamiento desde cualquier checkpoint configurando Hydra (`resume_from_checkpoint`).
- **Early stopping:** El entrenamiento puede detenerse anticipadamente según métricas de validación, configurable en los YAML de Hydra (`early_stopping`).
- **Configuración Hydra:** Todos los parámetros (épocas, optimizador, scheduler, checkpointing, early stopping, etc.) se gestionan vía YAML en `configs/training/`.
- **Sin lógica duplicada:** Todo el código legado de entrenamiento fue removido de `main.py`.

**Para entrenar:**
```bash
python src/main.py
```

**Para reanudar desde un checkpoint:**
- Edita tu config de Hydra (por ejemplo, `configs/training/trainer.yaml`) y pon la ruta en `training.checkpoints.resume_from_checkpoint`.

**Para más detalles:**
- Ver `src/training/trainer.py` para la lógica de orquestación.
- Ver `configs/training/trainer.yaml` para todas las opciones configurables.
- Ver la carpeta `tests/training/` para tests de integración y unidad del flujo de entrenamiento.

## Evaluation Flow

La evaluación final ya no se realiza en `main.py`. Para evaluar tu modelo entrenado en el set de test, usa el script dedicado:

```bash
python src/evaluate.py
```

- Este script carga el mejor o último checkpoint y calcula métricas en el set de test.
- La configuración (rutas, métricas, etc.) se gestiona vía YAML de Hydra, igual que el entrenamiento.
- Ver `src/evaluate.py` y `configs/evaluation/` para detalles.

**¿Por qué este cambio?**
- Esta separación asegura un flujo limpio y modular, evitando mezclar lógica de entrenamiento y evaluación.
- Facilita la automatización de experimentos y el mantenimiento del código.

## How to Contribute

- Por favor, lee las guías en `CONTRIBUTING.md` antes de enviar un pull request.
- Sigue las guías de estilo y modularidad (ver `coding-preferences.mdc`).
- Añade o actualiza tests para tus cambios.
- Actualiza la documentación según sea necesario.

## License

Este proyecto está bajo licencia MIT. Ver el archivo `LICENSE` para detalles.

## Conda Environment

Este proyecto usa un entorno Conda llamado `torch`.

**Para activarlo:**
```bash
conda activate torch
```

**Para instalar dependencias adicionales:**
```bash
conda install <package>
```

**Para reproducir el entorno:**
```bash
conda env create -f environment.yml
```

## Environment Variables

Este proyecto usa variables de entorno para configuración sensible.  
Ver el archivo de ejemplo: `.env.example`

- Copia `.env.example` a `.env` y completa los valores requeridos.
- Nunca subas tu archivo `.env` real al repositorio.

Variables principales:
- `ANTHROPIC_API_KEY`: API key para Anthropic Claude (Task Master)
- `DEBUG`: Activa/desactiva modo debug (`true` o `false`)

## Dependency Management

### Update Verification

El proyecto incluye un script para verificar actualizaciones de dependencias principales:

```bash
python scripts/utils/check_updates.py
```

Este script:
- Verifica versiones actuales en environment.yml
- Compara con las últimas versiones en conda-forge y PyPI
- Muestra un reporte de actualizaciones

### Updating Dependencies

Para actualizar dependencias:

1. Ejecuta el script de verificación
2. Actualiza versiones en environment.yml según sea necesario
3. Aplica las actualizaciones:
   ```bash
   conda env update -f environment.yml --prune
   ```
4. Verifica compatibilidad ejecutando los tests:
   ```bash
   pytest
   ```

### Consideraciones de actualización

- Mantén versiones compatibles de PyTorch y torchvision
- Verifica compatibilidad CUDA si usas GPU
- Documenta cambios significativos en CHANGELOG.md
- Realiza pruebas exhaustivas tras actualizar dependencias críticas

--- 