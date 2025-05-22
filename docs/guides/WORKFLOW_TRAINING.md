# Guía de Flujo de Trabajo para Entrenamiento

Este documento proporciona una guía paso a paso para configurar y ejecutar el entrenamiento de modelos de segmentación de grietas en pavimentos usando nuestro framework modular.

## Contenido

- [Requisitos Previos](#requisitos-previos)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Configuración](#configuración)
- [Ejecución del Entrenamiento](#ejecución-del-entrenamiento)
- [Evaluación de Modelos](#evaluación-de-modelos)
- [Resolución de Problemas](#resolución-de-problemas)

## Requisitos Previos

Antes de comenzar, asegúrate de tener lo siguiente:

1. **Entorno Conda**: Configura el entorno usando:

   ```bash
   conda env create -f environment.yml
   conda activate torch
   ```

2. **Datos**: Coloca tus datos en la estructura adecuada:

   ```txt
   data/
   ├── train/
   │   ├── images/
   │   └── masks/
   ├── val/
   │   ├── images/
   │   └── masks/
   └── test/
       ├── images/
       └── masks/
   ```

3. **Variables de Entorno**: Copia y configura el archivo `.env`:

   ```bash
   cp .env.example .env
   # Editar .env según sea necesario
   ```

## Estructura del Proyecto

La estructura modular del proyecto permite flexibilidad en la selección de componentes:

- **Arquitecturas**: `configs/model/architectures/`
- **Codificadores**: `configs/model/encoder/`
- **Decodificadores**: `configs/model/decoder/`
- **Funciones de Pérdida**: `configs/training/loss/`
- **Métricas**: `configs/training/metric/`
- **Planificadores de LR**: `configs/training/lr_scheduler/`

Todos estos componentes se combinan en la configuración principal.

## Configuración

### Configuración Principal

El archivo principal de configuración es `configs/config.yaml`, que incluye:

```yaml
defaults:
  - data: dataloader/default
  - model: architectures/unet
  - training: default
  - evaluation: default
  - _self_
```

Puedes anular cualquier configuración directamente desde la línea de comandos.

### Ejemplos de Configuración

1. **Entrenamiento básico con U-Net**:

   ```bash
   python run.py model=architectures/unet
   ```

2. **Cambiar arquitectura a DeepLabV3+**:

   ```bash
   python run.py model=architectures/deeplabv3plus
   ```

3. **Modificar función de pérdida**:

   ```bash
   python run.py training.loss=dice_bce
   ```

4. **Cambiar tamaño de lote y épocas**:

   ```bash
   python run.py data.batch_size=16 training.epochs=100
   ```

5. **Configuración completa personalizada**:

   ```bash
   python run.py model=architectures/swin_unet \
                 model.encoder.embed_dim=96 \
                 data.batch_size=8 \
                 training.optimizer.lr=0.0005 \
                 training.loss=dice_focal \
                 training.epochs=150
   ```

## Ejecución del Entrenamiento

### Entrenamiento Básico

Para iniciar un entrenamiento con la configuración predeterminada:

```bash
python run.py
```

### Monitoreo

El entrenamiento genera logs y métricas en:

```txt
outputs/
└── experiments/
    └── {timestamp}-{config_name}/
        ├── checkpoints/  # Modelos guardados
        ├── logs/         # Logs de TensorBoard
        ├── metrics/      # Métricas CSV
        └── results/      # Predicciones de validación
```

Para visualizar las métricas de entrenamiento:

```bash
tensorboard --logdir outputs/experiments/
```

## Evaluación de Modelos

Para evaluar un modelo entrenado:

```bash
python -m src.evaluation \
    model.checkpoint_path=outputs/experiments/{timestamp}-{config_name}/checkpoints/best_model.pth \
    evaluation.save_predictions=True
```

Esto generará métricas e imágenes de predicciones en el directorio de resultados.

## Resolución de Problemas

### Problemas Comunes

1. **Error de Memoria CUDA**: Reduzca el tamaño de lote o resolución de imagen

   ```python
   python run.py data.batch_size=4 data.image_size=[384,384]
   ```

2. **Gradientes Explotando**: Ajuste la tasa de aprendizaje o active el recorte de gradiente

   ```python
   python run.py training.optimizer.lr=0.0001 training.clip_grad_norm=1.0
   ```

3. **Métricas Estancadas**: Pruebe diferentes funciones de pérdida o aumentaciones de datos

   ```python
   python run.py training.loss=dice_focal data.augmentation=strong
   ```

### Optimización de Rendimiento

Para mejorar el rendimiento de entrenamiento:

1. **Precarga de Datos**: Aumente `num_workers` para el cargador de datos

   ```python
   python run.py data.num_workers=4
   ```

2. **Precisión Mixta**: Active el entrenamiento con precisión mixta

   ```python
   python run.py training.mixed_precision=true
   ```

---

Para cualquier pregunta adicional, consulte la documentación detallada en la carpeta `docs/` o abra un issue en el repositorio.

**Nota**: Esta guía se actualizará periódicamente con nuevas funciones y optimizaciones.
