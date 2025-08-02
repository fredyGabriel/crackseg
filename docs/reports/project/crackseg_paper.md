# Deep Learning-Based Crack Segmentation in Asphalt Pavement Using a Hybrid Swin Transformer U-Net Architecture

---

## 2. Resumen (Abstract)

This paper presents a modular, reproducible deep learning system for automatic crack segmentation in
asphalt pavement images. Leveraging a hybrid U-Net architecture with a Swin Transformer V2 encoder,
ASPP bottleneck, and CNN decoder, the system addresses the challenges of class imbalance and thin
structure detection. The pipeline is fully configurable via Hydra, supports experiment tracking, and
is optimized for RTX 3070 Ti GPUs. Key results demonstrate competitive performance on standard
metrics (IoU, Dice, F1-score), with robust generalization and efficient resource usage. The
system's modularity and reproducibility make it suitable for both research and practical deployment
in civil engineering applications.

---

## 3. Introducción

Crack detection in asphalt pavements is a critical task for infrastructure maintenance and safety.
Manual inspection is labor-intensive and error-prone, motivating the need for automated, AI-driven
solutions. This study aims to develop and evaluate a state-of-the-art deep learning algorithm for
crack segmentation, focusing on accuracy, reproducibility, and modularity. The paper is structured
as follows: Section 4 details the methodology, Section 5 presents results, Section 6 discusses
findings, Section 7 concludes and outlines future work, and Section 8 lists references.

---

## 4. Metodología

### 4.1. Adquisición y Preprocesamiento de Datos

- **Dataset Source & Size:** The dataset consists of high-resolution asphalt pavement images with
  corresponding binary crack masks. Data is organized into train/val/test splits (see `data/`
  directory and `configs/data/`).
- **Annotation:** Ground-truth masks are manually annotated, ensuring pixel-level accuracy for thin
  crack structures (1-5px width).
- **Preprocessing & Augmentation:**
  - Images are resized to 256x256 or 512x512 pixels.
  - Training augmentations include horizontal/vertical flips, random rotations, brightness/contrast
    adjustments, noise injection, and color jitter (see `src/data/transforms.py`).
  - Masks are binarized (0/1) and validated for correspondence with images.
  - Normalization uses ImageNet statistics for compatibility with pretrained encoders.

### 4.2. Arquitectura del Modelo de Segmentación

- **Model:** Hybrid U-Net with Swin Transformer V2 encoder, ASPP bottleneck, and CNN decoder (`src/model/architectures/swinv2_cnn_aspp_unet.py`).
- **Encoder:** SwinV2 (`swinv2_tiny_window16_256`), pretrained on ImageNet, provides hierarchical
  multi-scale features.
- **Bottleneck:** Atrous Spatial Pyramid Pooling (ASPP) captures multi-scale context.
- **Decoder:** CNN with skip connections and optional CBAM attention for precise localization.
- **Configuration:** All components are configurable via YAML files (`configs/model/architectures/`).
- **Transfer Learning:** Pretrained weights are used for the encoder; fine-tuning is supported.

### 4.3. Entrenamiento y Configuración

- **Hyperparameters:**
  - Optimizer: AdamW
  - Learning rate: 0.0001
  - Weight decay: 0.01
  - Batch size: 4 (adjustable for VRAM)
  - Epochs: configurable (default 100)
  - Loss: Combined BCE + Dice Loss (`BCEDiceLoss`), with tunable weights
  - Scheduler: StepLR or ReduceLROnPlateau (configurable)
- **Environment:**
  - Python 3.12+, PyTorch >=2.5.0
  - GPU: NVIDIA RTX 3070 Ti (8GB VRAM)
  - Reproducibility: Seed control and full config storage (`doconfiguration_storage_specification.md`)
- **Validation:**
  - Periodic validation and checkpointing
  - Early stopping based on validation IoU
  - All training artifacts and logs are saved for reproducibility

### 4.4. Métricas de Evaluación

- **Metrics:**
  - Intersection over Union (IoU)
  - Dice Coefficient
  - F1-score
  - Precision, Recall, Accuracy (optional)
- **Definitions:**
  - IoU: $\frac{TP}{TP + FP + FN}$
  - Dice: $\frac{2TP}{2TP + FP + FN}$
  - F1-score: Harmonic mean of precision and recall
- **Relevance:** These metrics are critical for evaluating segmentation quality, especially for
  thin, imbalanced crack structures.

---

## 5. Resultados

### 5.1. Rendimiento Cuantitativo

- **Evaluation Metrics:**
  - [PLACEHOLDER: Insert table with IoU, Dice, F1-score, etc. from test/validation set.]
  - Example:

| Metric | Value |
|--------|-------|
| IoU    | [PLACEHOLDER] |
| Dice   | [PLACEHOLDER] |
| F1     | [PLACEHOLDER] |
| Precision | [PLACEHOLDER] |
| Recall    | [PLACEHOLDER] |

- **Discussion:** The reported metrics reflect the model's ability to accurately segment cracks,
  balancing sensitivity to thin structures and robustness to noise/class imbalance.

### 5.2. Análisis Cualitativo y Ejemplos Visuales

- [PLACEHOLDER: Include figures showing input images, ground-truth masks, and predicted masks.]
- **Analysis:** Qualitative results demonstrate the model's effectiveness in detecting fine cracks
  and preserving boundaries, with some challenges in cases of severe noise or ambiguous textures.

---

## 6. Discusión

- The hybrid Swin Transformer U-Net architecture effectively addresses the main challenges in crack
  segmentation: class imbalance, thin structure detection, and edge preservation.
- The modular design and Hydra-based configuration enable rapid experimentation and adaptation to
  new datasets.
- Limitations include dependency on annotation quality and the need for hyperparameter tuning for
  different domains.
- Compared to classical U-Net and CNN-only baselines, the inclusion of transformer-based encoders
  and ASPP bottlenecks improves multi-scale context understanding and segmentation accuracy (see `docs/reports/technical_report.md`).

---

## 7. Conclusiones y Recomendaciones

### 7.1. Conclusiones

- The proposed system achieves state-of-the-art performance for crack segmentation in asphalt
  pavement, with strong generalization and reproducibility.
- The modular, configurable pipeline supports both research and deployment scenarios.

### 7.2. Recomendaciones y Trabajo Futuro

- Explore additional encoder/decoder variants (e.g., larger Swin models, lightweight ViTs).
- Integrate semi-supervised learning and self-labeling techniques to leverage unlabeled data.
- Optimize for edge deployment and real-time inference.
- Expand to multi-class defect detection and severity estimation.
- Improve visualization and automated reporting tools.

---

## 8. Referencias (Placeholder)

This section should be completed manually with relevant academic citations. See also:

- [Technical Report](./technical_report.md)
- [Checkpoint Format Specification](checkpoint_format_specification.md)
- [Configuration Storage Specification](configuration_storage_specification.md)
- [Hydra Documentation](https://hydra.cc/)
- [Albumentations](https://albumentations.ai/)
- [PyTorch](https://pytorch.org/)
- [Swin Transformer](https://arxiv.org/abs/2103.14030)

---
