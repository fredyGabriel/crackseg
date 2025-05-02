# Código Fuente del Proyecto de Segmentación de Grietas

Este directorio contiene el código fuente principal del proyecto de segmentación de grietas en pavimento. La estructura está organizada de manera modular para facilitar el mantenimiento y la escalabilidad.

## Estructura del Directorio

- `data/`: Módulos para el manejo y procesamiento de datos
  - `dataset.py`: Implementación de datasets personalizados
  - `transforms.py`: Transformaciones de datos y aumentación
  - `factory.py`: Fábrica para crear instancias de datasets
  - `memory.py`: Utilidades para manejo eficiente de memoria
  - `sampler.py`: Implementación de samplers personalizados
  - `splitting.py`: Funciones para división de datos
  - `distributed.py`: Soporte para entrenamiento distribuido

- `model/`: Implementación de arquitecturas de redes neuronales
  - `unet.py`: Implementación de la arquitectura U-Net
  - `base.py`: Clases base para modelos
  - `factory.py`: Fábrica para crear instancias de modelos
  - `config.py`: Configuraciones específicas de modelos
  - `encoder/`: Módulos de codificación
  - `decoder/`: Módulos de decodificación
  - `bottleneck/`: Módulos de cuello de botella

- `training/`: Lógica de entrenamiento y evaluación
  - `trainer.py`: Clase principal de entrenamiento
  - `losses.py`: Funciones de pérdida personalizadas
  - `metrics.py`: Métricas de evaluación
  - `factory.py`: Fábrica para componentes de entrenamiento

- `utils/`: Utilidades generales
  - `checkpointing.py`: Manejo de checkpoints
  - `config_*.py`: Utilidades de configuración
  - `device.py`: Gestión de dispositivos (CPU/GPU)
  - `early_stopping.py`: Implementación de early stopping
  - `env.py`: Variables de entorno
  - `factory.py`: Fábrica genérica
  - `loggers.py`: Configuración de logging
  - `paths.py`: Gestión de rutas
  - `seeds.py`: Control de semillas aleatorias

## Archivo Principal

- `main.py`: Punto de entrada principal del proyecto que orquesta el entrenamiento y la evaluación

## Características Principales

- Arquitectura modular y extensible
- Configuración basada en Hydra
- Soporte para experimentación
- Logging detallado
- Manejo eficiente de recursos
- Compatibilidad con entrenamiento distribuido

## Uso

El proyecto utiliza Hydra para la gestión de configuraciones. Para ejecutar el entrenamiento:

```bash
python main.py
```

Para modificar la configuración, usar los archivos en el directorio `configs/` o sobrescribir parámetros desde la línea de comandos:

```bash
python main.py data.batch_size=32 training.epochs=100
``` 