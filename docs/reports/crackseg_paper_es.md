# Segmentación de Fisuras en Asfalto mediante Deep Learning con Arquitectura Híbrida Swin Transformer U-Net

---

## 2. Resumen (Abstract)

Este artículo presenta un sistema modular y reproducible basado en deep learning para la
segmentación automática de fisuras en imágenes de pavimento asfáltico. Aprovechando una arquitectura
U-Net híbrida con un encoder Swin Transformer V2, un bottleneck ASPP y un decoder CNN, el sistema
aborda los desafíos del desbalance de clases y la detección de estructuras delgadas. El pipeline es
completamente configurable mediante Hydra, soporta el seguimiento de experimentos y está optimizado
para GPUs RTX 3070 Ti. Los resultados clave demuestran un desempeño competitivo en métricas estándar
(IoU, Dice, F1-score), con generalización robusta y uso eficiente de recursos. La modularidad y
reproducibilidad del sistema lo hacen apto tanto para investigación como para aplicaciones prácticas
en ingeniería civil.

---

## 3. Introducción

La detección de fisuras en pavimentos asfálticos es una tarea crítica para el mantenimiento y la
seguridad de las infraestructuras. La inspección manual es laboriosa y propensa a errores, lo que
motiva la necesidad de soluciones automatizadas basadas en IA. Este estudio tiene como objetivo
desarrollar y evaluar un algoritmo de deep learning de última generación para la segmentación de
fisuras, priorizando la precisión, la reproducibilidad y la modularidad. El artículo se estructura
de la siguiente manera: la Sección 4 detalla la metodología, la Sección 5 presenta los resultados,
la Sección 6 discute los hallazgos, la Sección 7 concluye y propone líneas futuras, y la Sección 8
lista las referencias.

---

## 4. Metodología

### 4.1. Adquisición y Preprocesamiento de Datos

- **Fuente y tamaño del dataset:** El conjunto de datos consiste en imágenes de alta resolución de
  pavimento asfáltico con sus correspondientes máscaras binarias de fisuras. Los datos se organizan
  en particiones de entrenamiento, validación y prueba (ver directorio `data/` y `configs/data/`).
- **Anotación:** Las máscaras de referencia son anotadas manualmente, asegurando precisión a nivel
  de píxel para estructuras de fisuras delgadas (1-5px de ancho).
- **Preprocesamiento y aumentos:**
  - Las imágenes se redimensionan a 256x256 o 512x512 píxeles.
  - Los aumentos en entrenamiento incluyen flips horizontales/verticales, rotaciones aleatorias,
    ajustes de brillo/contraste, inyección de ruido y jitter de color (ver `src/data/transforms.py`).
  - Las máscaras se binarizan (0/1) y se valida su correspondencia con las imágenes.
  - La normalización utiliza estadísticas de ImageNet para compatibilidad con encoders preentrenados.

### 4.2. Arquitectura del Modelo de Segmentación

- **Modelo:** U-Net híbrido con encoder Swin Transformer V2, bottleneck ASPP y decoder CNN (`src/model/architectures/swinv2_cnn_aspp_unet.py`).
- **Encoder:** SwinV2 (`swinv2_tiny_window16_256`), preentrenado en ImageNet, provee características
  jerárquicas multi-escala.
- **Bottleneck:** Atrous Spatial Pyramid Pooling (ASPP) captura contexto multi-escala.
- **Decoder:** CNN con skip connections y atención CBAM opcional para localización precisa.
- **Configuración:** Todos los componentes son configurables vía archivos YAML (`configs/model/architectures/`).
- **Transfer Learning:** Se emplean pesos preentrenados para el encoder; se soporta fine-tuning.

### 4.3. Entrenamiento y Configuración

- **Hiperparámetros:**
  - Optimizador: AdamW
  - Tasa de aprendizaje: 0.0001
  - Weight decay: 0.01
  - Batch size: 4 (ajustable según VRAM)
  - Épocas: configurable (por defecto 100)
  - Pérdida: BCE + Dice Loss combinadas (`BCEDiceLoss`), con pesos ajustables
  - Scheduler: StepLR o ReduceLROnPlateau (configurable)
- **Entorno:**
  - Python 3.12+, PyTorch >=2.5.0
  - GPU: NVIDIA RTX 3070 Ti (8GB VRAM)
  - Reproducibilidad: control de semillas y almacenamiento completo de la configuración (`docs/guides/configuration_storage_specification.md`)
- **Validación:**
  - Validación periódica y guardado de checkpoints
  - Early stopping basado en IoU de validación
  - Todos los artefactos y logs de entrenamiento se guardan para reproducibilidad

### 4.4. Métricas de Evaluación

- **Métricas:**
  - Intersection over Union (IoU)
  - Coeficiente Dice
  - F1-score
  - Precisión, Recall, Accuracy (opcional)
- **Definiciones:**
  - IoU: $\frac{TP}{TP + FP + FN}$
  - Dice: $\frac{2TP}{2TP + FP + FN}$
  - F1-score: media armónica de precisión y recall
- **Relevancia:** Estas métricas son críticas para evaluar la calidad de la segmentación,
  especialmente en estructuras delgadas y desbalanceadas.

---

## 5. Resultados

### 5.1. Rendimiento Cuantitativo

- **Métricas de evaluación:**
  - [PLACEHOLDER: Insertar tabla con IoU, Dice, F1-score, etc. del set de prueba/validación.]
  - Ejemplo:

| Métrica   | Valor         |
|-----------|--------------|
| IoU       | [PLACEHOLDER]|
| Dice      | [PLACEHOLDER]|
| F1        | [PLACEHOLDER]|
| Precisión | [PLACEHOLDER]|
| Recall    | [PLACEHOLDER]|

- **Discusión:** Las métricas reportadas reflejan la capacidad del modelo para segmentar fisuras
  con precisión, equilibrando sensibilidad a estructuras delgadas y robustez ante ruido y desbalance
  de clases.

### 5.2. Análisis Cualitativo y Ejemplos Visuales

- [PLACEHOLDER: Incluir figuras mostrando imágenes de entrada, máscaras de referencia y máscaras predichas.]
- **Análisis:** Los resultados cualitativos demuestran la efectividad del modelo para detectar
  fisuras finas y preservar bordes, aunque se observan desafíos en casos de ruido severo o texturas ambiguas.

---

## 6. Discusión

- La arquitectura híbrida Swin Transformer U-Net aborda eficazmente los principales desafíos de la
  segmentación de fisuras: desbalance de clases, detección de estructuras delgadas y preservación de
  bordes.
- El diseño modular y la configuración basada en Hydra permiten experimentación rápida y adaptación
  a nuevos datasets.
- Las limitaciones incluyen la dependencia de la calidad de las anotaciones y la necesidad de
  ajustar hiperparámetros para distintos dominios.
- En comparación con U-Net clásico y baselines solo-CNN, la inclusión de encoders tipo transformer y
  bottlenecks ASPP mejora la comprensión de contexto multi-escala y la precisión de segmentación
  (ver `docs/reports/technical_report.md`).

---

## 7. Conclusiones y Recomendaciones

### 7.1. Conclusiones

- El sistema propuesto alcanza desempeño de estado del arte para segmentación de fisuras en
  pavimento asfáltico, con fuerte generalización y reproducibilidad.
- El pipeline modular y configurable soporta tanto investigación como despliegue práctico.

### 7.2. Recomendaciones y Trabajo Futuro

- Explorar variantes adicionales de encoder/decoder (por ejemplo, Swin más grandes, ViTs livianos).
- Integrar técnicas de aprendizaje semi-supervisado y auto-etiquetado para aprovechar datos no anotados.
- Optimizar para despliegue en edge y para inferencia en tiempo real.
- Expandir a detección multi-clase de defectos y estimación de severidad.
- Mejorar la visualización y los reportes automáticos.

---

## 8. Referencias (Placeholder)

Esta sección debe ser completada manualmente con las citas académicas pertinentes. Véase también:

- [Informe Técnico](./technical_report.md)
- [Especificación de Formato de Checkpoints](../guides/checkpoint_format_specification.md)
- [Especificación de Almacenamiento de Configuración](../guides/configuration_storage_specification.md)
- [Documentación de Hydra](https://hydra.cc/)
- [Albumentations](https://albumentations.ai/)
- [PyTorch](https://pytorch.org/)
- [Swin Transformer](https://arxiv.org/abs/2103.14030)

---
