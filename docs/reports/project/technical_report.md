# Informe Técnico del Algoritmo de Segmentación de Fisuras - CrackSeg

## 1. Resumen Ejecutivo

El proyecto CrackSeg desarrolla un sistema avanzado para la segmentación automática de fisuras en
pavimento asfáltico, empleando técnicas de Deep Learning con PyTorch. Su arquitectura modular,
basada en modelos encoder-decoder configurables mediante Hydra, permite experimentar con diferentes
variantes de red, facilitando la investigación y la adaptación a distintos escenarios de detección
de fisuras. El sistema es funcional, ejecutable y está respaldado por una interfaz gráfica
profesional (Streamlit) que simplifica la configuración, entrenamiento y análisis de resultados.
Su impacto radica en la mejora de la precisión y reproducibilidad en la detección de fisuras,
abordando retos como el desbalance de clases y la detección de estructuras delgadas, aspectos
críticos en aplicaciones de ingeniería civil.

## 2. Introducción

Las fisuras en pavimentos representan un problema relevante para la seguridad y durabilidad de
infraestructuras viales. La detección manual es costosa y propensa a errores, por lo que una
solución automatizada basada en Deep Learning ofrece ventajas significativas: mayor objetividad,
escalabilidad y capacidad de adaptación a grandes volúmenes de datos. CrackSeg tiene como objetivo
proporcionar una herramienta robusta y reproducible para la segmentación precisa de fisuras,
integrando buenas prácticas de ingeniería de software y aprendizaje profundo.

## 3. Arquitectura del Algoritmo

### 3.1. Visión General de la Arquitectura

El núcleo del sistema es un modelo encoder-decoder tipo U-Net, donde el encoder puede ser un Swin
Transformer V2 preentrenado ("swinv2_tiny_window16_256") y el decoder es una red convolucional. La
arquitectura es altamente configurable mediante archivos YAML y un sistema de factorías que permite
intercambiar componentes (encoder, bottleneck, decoder) de forma flexible. El uso de Swin
Transformer como encoder permite capturar características multi-escala y mejorar la detección de
fisuras delgadas.

### 3.2. Componentes Clave del Modelo

- **Encoder**: SwinTransformerEncoder, preentrenado en ImageNet, configurable en tamaño de entrada,
  patch size y salidas intermedias.
- **Bottleneck**: CNNBottleneckBlock, recibe la última característica del encoder (768 canales) y la
  expande a 1024 canales.
- **Decoder**: CNNDecoder, recibe la salida del bottleneck y utiliza skip connections de las capas
  intermedias del encoder ([384, 192, 96] canales) para reconstruir la máscara de segmentación.
- **Activación final**: Sigmoid para segmentación binaria.

### 3.3. Configuración de Parámetros

- Tamaño de entrada: 256x256 píxeles (configurable)
- Número de clases: 1 (segmentación binaria)
- Optimizer: AdamW, lr=0.0001, weight_decay=0.01
- Preentrenamiento: Sí (ImageNet)
- Patch size encoder: 4
- Skip connections: [384, 192, 96] canales

## 4. Preparación y Preprocesamiento de Datos

### 4.1. Adquisición y Colección de Datos

El sistema espera los datos organizados en carpetas `data/train`, `data/val` y `data/test`, cada una
con subcarpetas `images/` y `masks/`. Las imágenes y máscaras deben estar emparejadas por nombre de
archivo. El tamaño y formato de las imágenes es configurable, y el sistema soporta múltiples fuentes
de datos (archivos, PIL, numpy arrays).

### 4.2. Anotación de Datos

Las máscaras de segmentación son imágenes binarias (0/1) en formato PNG, donde los píxeles positivos
representan fisuras. El sistema valida y normaliza automáticamente las máscaras, asegurando
compatibilidad y calidad en el entrenamiento.

### 4.3. Preprocesamiento de Imágenes

El preprocesamiento se realiza mediante Albumentations y es configurable por split (train/val/test):

- **Train**: Resize (256x256), flips horizontales/verticales, rotación aleatoria, color jitter,
  normalización y conversión a tensor.
- **Val/Test**: Resize (256x256), normalización y conversión a tensor.

## 5. Entrenamiento del Modelo

### 5.1. Entorno de Entrenamiento

- Python 3.12+
- PyTorch >=2.5.0
- Entrenamiento acelerado por GPU (CUDA, recomendado RTX 3070 Ti 8GB)
- Configuración reproducible mediante Hydra y archivos YAML

### 5.2. Función de Pérdida (Loss Function)

Se utiliza una función combinada BCE + Dice Loss (`BCEDiceLoss`), configurable en pesos relativos y
parámetros de suavizado. Esta combinación es adecuada para segmentación de fisuras por su capacidad
para manejar desbalance de clases y optimizar tanto la precisión global como la de bordes.

### 5.3. Optimizador y Tasa de Aprendizaje

- Optimizador: AdamW
- Learning rate: 0.0001
- Weight decay: 0.01

### 5.4. Estrategia de Entrenamiento

- Tamaño de batch: 4 (ajustable según VRAM)
- Número de épocas: configurable
- Validación periódica y guardado de checkpoints con metadatos completos (época, métricas,
  configuración, versión de PyTorch, etc.)
- Reproducibilidad garantizada mediante control de semillas y almacenamiento de configuración

### 5.5. Métricas de Evaluación en Entrenamiento

- IoU (Intersection over Union)
- F1-Score
- Dice Coefficient
- Accuracy, Precision, Recall (según configuración)

## 6. Resultados y Evaluación

### 6.1. Métricas de Rendimiento

Las métricas de rendimiento (IoU, F1, Dice) se registran durante el entrenamiento y validación. Si
no hay resultados explícitos, estos deben ser completados manualmente tras la ejecución de experimentos.

### 6.2. Ejemplos de Segmentación

*Placeholder*: Esta sección debe complementarse con imágenes visuales de ejemplos de segmentación
(original, máscara real, predicción).

### 6.3. Discusión de Resultados

El uso de Swin Transformer como encoder y la combinación de pérdidas BCE+Dice permite abordar los
principales retos de la segmentación de fisuras: desbalance de clases, detección de estructuras
delgadas y preservación de bordes. El sistema es modular, reproducible y fácilmente extensible. Las
limitaciones actuales incluyen la dependencia de la calidad de anotaciones y la necesidad de
ajustar hiperparámetros para distintos datasets.

## 7. Implementación y Usabilidad

### 7.1. Estructura del Código

El repositorio sigue una estructura modular:

- `src/`: Código fuente principal (modelos, entrenamiento, datos, utilidades)
- `configs/`: Configuraciones Hydra (modelos, entrenamiento, datos, métricas)
- `data/`: Datos organizados por split y tipo (imágenes, máscaras)
- `scripts/`: Scripts y utilidades (incluye la GUI en `gui/`)
- `docs/`: Documentación, guías y reportes
- `tests/`: Pruebas unitarias e integración

### 7.2. Dependencias

Principales dependencias (ver `requirements.txt`):

- hydra-core >=1.3.2
- omegaconf >=2.3.0
- torch >=2.5.0
- torchvision >=0.20.0
- albumentations >=2.0.0
- timm >=1.0.0
- opencv-python >=4.11.0
- Pillow >=10.4.0
- streamlit >=1.45.0
- pytest, black, ruff, basedpyright (desarrollo y calidad)

### 7.3. Instrucciones de Ejecución

**Configuración del entorno:**

```bash
conda env create -f environment.yml
conda activate crackseg
```

**Entrenamiento (CLI):**

```bash
python run.py
# O con configuración específica:
python run.py model=architectures/unet_swin data.batch_size=4 training.loss=bce_dice
```

**Entrenamiento (GUI):**

```bash
conda activate crackseg && streamlit run gui/app.py
```

**Evaluación:**

```bash
python -m src.evaluation.evaluate --checkpoint_path ... --config_path ...
```

### 7.4. Consideraciones de Rendimiento (Inferencia)

- Requiere GPU para inferencia en tiempo real (recomendado RTX 3070 Ti 8GB)
- El batch size y la resolución de entrada deben ajustarse según la memoria disponible
- Esta sección puede completarse tras pruebas de rendimiento específicas

## 8. Conclusiones y Trabajo Futuro

### 8.1. Conclusiones

CrackSeg proporciona una base sólida y profesional para la segmentación de fisuras en pavimento,
integrando buenas prácticas de ingeniería, reproducibilidad y flexibilidad experimental. Su
arquitectura modular y la integración de una GUI facilitan tanto la investigación como la aplicación
práctica.

### 8.2. Limitaciones Actuales

- Dependencia de la calidad y cantidad de datos anotados
- Ajuste manual de hiperparámetros para nuevos datasets
- Requiere GPU para entrenamiento eficiente

### 8.3. Futuras Mejoras y Expansiones

- Integración de nuevos encoders y decoders (e.g., variantes de Vision Transformers)
- Incorporación de técnicas de auto-rotulación y aprendizaje semi-supervisado
- Optimización para dispositivos edge y despliegue en campo
- Mejora de la visualización de resultados y reportes automáticos

## 9. Referencias (Opcional)

- [Hydra Documentation](https://hydra.cc/)
- [Albumentations](https://albumentations.ai/)
- [PyTorch](https://pytorch.org/)
- [Swin Transformer](https://arxiv.org/abs/2103.14030)

## 10. Apéndices (Opcional)

- Diagrama de arquitectura técnica (ver `docs/guides/architecture/TECHNICAL_ARCHITECTURE.md`)
- Ejemplo de configuración YAML (`configs/model/architectures/unet_swin.yaml`)
- Especificación de checkpoints (`docs/guides/specifications/checkpoint_format_specification.md`)
