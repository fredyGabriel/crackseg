# Data Processing Scripts

Este directorio contiene scripts para el procesamiento de datos en el proyecto CrackSeg, organizados
por tipo de tarea.

## Estructura del Directorio

```txt
data_processing/
├── mask_verification/          # Verificación de máscaras de segmentación
├── format_conversion/          # Conversión entre formatos de datos
├── image_processing/           # Procesamiento de imágenes
└── README.md                   # Este archivo
```

## Subdirectorios

### 📋 `mask_verification/`

Scripts para verificar la precisión de las máscaras de segmentación mediante superposición visual.

**Archivos:**

- `segmentation_mask_verifier.py` - Clase principal para verificación de máscaras
- `run_verification.py` - Interfaz de línea de comandos
- `demo_verification.py` - Script de demostración
- `example_verification.py` - Ejemplo programático
- `README_verification.md` - Documentación completa
- `VERIFICATION_SYSTEM_SUMMARY.md` - Resumen ejecutivo

**Uso:**

```bash
# Verificar una sola imagen
python run_verification.py --single-image 125

# Verificar múltiples imágenes
python run_verification.py --max-samples 10

# Ejecutar demostración
python demo_verification.py
```

### 🔄 `format_conversion/`

Scripts para convertir entre diferentes formatos de datos (segmentación ↔ detección de objetos).

**Archivos:**

- `segmentation_to_detection.py` - Conversor principal
- `convert_crackseg_dataset.py` - Conversión de datasets completos
- `README_segmentation_to_detection.md` - Documentación

**Formatos soportados:**

- YOLO
- COCO
- Pascal VOC

### 🖼️ `image_processing/`

Scripts para procesamiento básico de imágenes.

**Archivos:**

- `crop_crack_images.py` - Recorte de imágenes de grietas
- `README_crop_crack_images.md` - Documentación

## Estándares de Calidad

Todos los scripts siguen los estándares del proyecto:

- **Type Hints**: Anotaciones completas de tipos (Python 3.12+)
- **Documentación**: Docstrings en inglés para todas las APIs públicas
- **Logging**: Uso de `logging` para salidas informativas
- **Manejo de Errores**: Excepciones apropiadas y mensajes claros
- **Testing**: Cobertura de pruebas >80% en funcionalidad crítica

## Salidas

**IMPORTANTE**: Todas las salidas se guardan en `artifacts/`, no en `outputs/`.

- `artifacts/verification_results/` - Resultados de verificación de máscaras
- `artifacts/verification_demo/` - Demostraciones de verificación
- `artifacts/batch_verification_demo/` - Verificaciones en lote

## Ejemplos de Uso

### Verificación de Máscaras

```python
from mask_verification.segmentation_mask_verifier import SegmentationMaskVerifier

verifier = SegmentationMaskVerifier(
    images_dir=Path("data/PY-CrackBD/Segmentation/Original image"),
    masks_dir=Path("data/PY-CrackBD/Segmentation/Ground truth"),
    output_dir=Path("artifacts/verification_results")
)

result = verifier.verify_single_pair("sample_image")
```

### Conversión de Formatos

```python
from format_conversion.segmentation_to_detection import MaskToDetectionConverter

converter = MaskToDetectionConverter(
    image_dir=Path("data/images"),
    mask_dir=Path("data/masks"),
    output_dir=Path("data/detection"),
    format_type="yolo"
)

converter.convert_dataset()
```

## Contribución

Al agregar nuevos scripts:

1. **Organizar por tipo de tarea** en el subdirectorio apropiado
2. **Seguir convenciones de nombres** consistentes
3. **Incluir documentación** completa
4. **Usar `artifacts/`** para todas las salidas
5. **Mantener compatibilidad** con estándares del proyecto

## Troubleshooting

### Problemas Comunes

**Error**: "Images directory not found"

- Verificar que las rutas sean relativas al directorio raíz del proyecto
- Asegurar que los directorios de datos existan

**Error**: "No matching image-mask pairs found"

- Verificar que los nombres de archivos coincidan (sin extensión)
- Asegurar que las extensiones sean correctas (.jpg para imágenes, .png para máscaras)

**Error**: "Output directory not writable"

- Verificar permisos en `artifacts/`
- Crear directorios de salida si no existen

## Referencias

- [Coding Standards](/.cursor/rules/coding-standards.mdc)
- [ML PyTorch Standards](/.cursor/rules/ml-pytorch-standards.mdc)
- [Testing Standards](/.cursor/rules/testing-standards.mdc)
