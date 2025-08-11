# Segmentation Mask Verification System

Este sistema implementa la verificación de precisión de máscaras de segmentación mediante un
proceso de revisión cruzada y superposición visual, tal como se requiere en el proyecto CrackSeg.

## Descripción

El sistema cumple con el requisito: "The accuracy of segmentation masks is verified through a
cross-review process and by visually superimposing the masks onto the original images to confirm alignment"

### Características Principales

- ✅ **Verificación Visual**: Crea superposiciones visuales de máscaras sobre imágenes originales
- ✅ **Proceso de Revisión Cruzada**: Valida la precisión mediante inspección visual
- ✅ **Estadísticas Detalladas**: Calcula métricas de cobertura y características de grietas
- ✅ **Procesamiento por Lotes**: Verifica múltiples pares imagen-máscara automáticamente
- ✅ **Interfaz de Línea de Comandos**: Fácil uso desde terminal
- ✅ **Documentación Completa**: Código bien documentado siguiendo estándares del proyecto

## Archivos del Sistema

### Scripts Principales

1. **`segmentation_mask_verifier.py`** - Clase principal para verificación
2. **`demo_verification.py`** - Script de demostración
3. **`run_verification.py`** - Interfaz de línea de comandos
4. **`example_verification.py`** - Ejemplo de uso programático

### Estructura de Datos

```bash
data/PY-CrackBD/Segmentation/
├── Original image/     # Imágenes originales (.jpg)
└── Ground truth/       # Máscaras de segmentación (.png)
```

## Uso Rápido

### 1. Verificar una Imagen Específica

```bash
cd scripts/data_processing
python run_verification.py --single-image 125
```

**Resultado**: Crea una imagen con tres paneles:

- Imagen original (izquierda)
- Máscara de segmentación (centro)
- Superposición (derecha) - máscara sobrepuesta en rojo

### 2. Listar Imágenes Disponibles

```bash
python run_verification.py --list-available
```

### 3. Verificar Múltiples Imágenes

```bash
# Verificar hasta 10 imágenes
python run_verification.py --max-samples 10

# Verificar todas las imágenes disponibles
python run_verification.py
```

### 4. Usar Directorio Personalizado

```bash
python run_verification.py --output-dir ./mis_resultados --max-samples 5
```

## Ejemplo de Salida

```txt
🔍 Verifying single image: 125
✅ Verification completed successfully!

📊 RESULTS:
Image: 125
Image shape: (500, 351, 3)
Mask shape: (500, 351)
Crack coverage: 1.77%
Crack pixels: 3,102
Number of crack regions: 156
Verification result saved to: outputs/verification_results

🎯 VERIFICATION PROCESS COMPLETED
The verification process has created visual overlays showing:
1. Original image
2. Segmentation mask
3. Superposition (mask overlaid on original image)
```

## Uso Programático

### Verificación de Imagen Individual

```python
from segmentation_mask_verifier import SegmentationMaskVerifier

# Inicializar verificador
verifier = SegmentationMaskVerifier(
    images_dir="data/PY-CrackBD/Segmentation/Original image",
    masks_dir="data/PY-CrackBD/Segmentation/Ground truth",
    output_dir="outputs/verification_results"
)

# Verificar imagen específica
result = verifier.verify_single_pair("125")

if result["success"]:
    stats = result["statistics"]
    print(f"Cobertura de grietas: {stats['coverage_percentage']:.2f}%")
    print(f"Píxeles de grietas: {stats['crack_pixels']:,}")
```

### Verificación por Lotes

```python
# Verificar múltiples imágenes
results = verifier.verify_dataset(max_samples=10)

print(f"Total procesadas: {results['total_pairs']}")
print(f"Exitosas: {results['successful_verifications']}")
print(f"Tasa de éxito: {results['success_rate']:.2%}")
```

## Estadísticas Calculadas

### Para Cada Imagen

- **Cobertura de Grietas**: Porcentaje de píxeles que contienen grietas
- **Píxeles de Grietas**: Número total de píxeles de grietas
- **Regiones de Grietas**: Número de componentes conectados de grietas
- **Dimensiones**: Ancho y alto de la imagen

### Estadísticas Globales

- **Cobertura Promedio**: Promedio de cobertura de grietas
- **Rango de Cobertura**: Mínimo y máximo de cobertura
- **Píxeles Totales**: Suma de todos los píxeles de grietas

## Estándares de Calidad

### Código

- ✅ **Type Annotations**: Python 3.12+ con tipos completos
- ✅ **Documentación**: Docstrings estilo Google
- ✅ **Manejo de Errores**: Excepciones específicas
- ✅ **Logging**: Sistema de logging configurado
- ✅ **Validación**: Verificación de archivos y directorios

### Proceso de Verificación

- ✅ **Validación de Formato**: Verifica formatos de imagen y máscara
- ✅ **Normalización**: Convierte máscaras a binario (0/255)
- ✅ **Superposición Visual**: Crea overlays con color configurable
- ✅ **Estadísticas**: Calcula métricas de precisión
- ✅ **Visualización**: Genera imágenes de verificación

## Estructura de Salida

### Directorio de Resultados

```txt
outputs/verification_results/
├── 125_verification.png    # Imagen de verificación
├── 132_verification.png    # Imagen de verificación
└── ...
```

### Formato de Imagen de Verificación

Cada imagen de verificación contiene tres paneles:

1. **Imagen Original** (izquierda)
   - Imagen RGB sin procesar
   - Muestra la superficie de pavimento con grietas

2. **Máscara de Segmentación** (centro)
   - Imagen binaria en escala de grises
   - Píxeles blancos = grietas detectadas
   - Píxeles negros = fondo

3. **Superposición** (derecha)
   - Imagen original con máscara sobrepuesta en rojo
   - Permite verificar la precisión de la segmentación
   - Confirma el alineamiento entre imagen y máscara

## Casos de Uso

### 1. Control de Calidad

```bash
# Verificar precisión de máscaras generadas por modelo
python run_verification.py --single-image 125
```

### 2. Validación de Dataset

```bash
# Verificar todo el dataset de entrenamiento
python run_verification.py
```

### 3. Análisis de Precisión

```bash
# Analizar estadísticas de cobertura
python run_verification.py --max-samples 50
```

### 4. Desarrollo de Modelos

```python
# Integrar en pipeline de entrenamiento
verifier = SegmentationMaskVerifier(...)
results = verifier.verify_dataset()
if results['success_rate'] < 0.95:
    print("⚠️ Dataset quality issues detected")
```

## Requisitos

### Dependencias

```python
matplotlib>=3.5.0
numpy>=1.21.0
Pillow>=8.3.0
scipy>=1.7.0
```

### Estructura de Datos

- Imágenes originales: formato `.jpg`
- Máscaras de segmentación: formato `.png`
- Nombres de archivo deben coincidir entre directorios

## Troubleshooting

### Problemas Comunes

1. **Directorio no encontrado**

   ```txt
   ❌ Images directory not found: data/PY-CrackBD/Segmentation/Original image
   ```

   **Solución**: Verificar que las rutas sean correctas

2. **No hay pares imagen-máscara**

   ```txt
   ❌ No matching image-mask pairs found
   ```

   **Solución**: Verificar que los nombres de archivo coincidan

3. **Error de formato de imagen**

   ```txt
   ❌ Failed to load image: Invalid format
   ```

   **Solución**: Verificar que las imágenes sean válidas

### Verificación de Instalación

```bash
# Probar con imagen de ejemplo
python demo_verification.py

# Verificar dependencias
python -c "import matplotlib, numpy, PIL; print('✅ Dependencies OK')"
```

## Contribución

### Estándares de Código

- Seguir estándares de codificación del proyecto CrackSeg
- Usar type annotations de Python 3.12+
- Documentar todas las funciones públicas
- Incluir tests para nuevas funcionalidades

### Extensibilidad

El sistema está diseñado para ser extensible:

- **Nuevos Formatos**: Agregar soporte para otros formatos de imagen
- **Métricas Adicionales**: Implementar nuevas métricas de precisión
- **Visualizaciones**: Agregar diferentes tipos de superposición
- **Integración**: Conectar con sistemas de entrenamiento de modelos

## Referencias

- **Proyecto CrackSeg**: Sistema de segmentación de grietas en pavimento
- **Estándares de Codificación**: `coding-standards.mdc`
- **Workflow de Desarrollo**: `development-workflow.mdc`
- **Estándares de ML**: `ml-pytorch-standards.mdc`

---

**Nota**: Este sistema cumple con los requisitos de verificación de precisión de máscaras de
segmentación mediante superposición visual, proporcionando una herramienta robusta para el control
de calidad en el proyecto CrackSeg.
