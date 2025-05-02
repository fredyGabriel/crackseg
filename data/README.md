# Directorio de Datos

Este directorio contiene los datos utilizados para el entrenamiento, validación y prueba del modelo de segmentación de grietas en pavimento.

## Estructura del Directorio

```
data/
├── examples/          # Example data and dummy files for testing
├── train/            # Training dataset
├── val/              # Validation dataset
└── test/             # Test dataset
```

## Formato de Datos

### Imágenes
- Formato: PNG o JPG
- Dimensiones: Variables (se redimensionan durante el preprocesamiento)
- Canales: RGB o escala de grises
- Nomenclatura: `imagen_XXX.{png|jpg}`

### Máscaras de Segmentación
- Formato: PNG
- Dimensiones: Iguales a las imágenes correspondientes
- Canales: 1 (binario)
- Valores: 0 (fondo) y 255 (grieta)
- Nomenclatura: `imagen_XXX_mask.png`

## División de Datos

- **Entrenamiento (train/)**: 70% del conjunto de datos
- **Validación (val/)**: 15% del conjunto de datos
- **Prueba (test/)**: 15% del conjunto de datos

## Preprocesamiento

Las imágenes se preprocesan durante el entrenamiento según la configuración en `configs/data/transform/`.

Transformaciones típicas incluyen:
- Redimensionamiento
- Normalización
- Aumentación de datos (solo en entrenamiento)
  - Rotaciones
  - Volteos
  - Ajustes de brillo/contraste
  - Ruido
  - Recortes aleatorios

## Gestión de Datos

- No incluir datos en el control de versiones
- Mantener una copia de respaldo de los datos
- Documentar cualquier preprocesamiento manual realizado
- Verificar la integridad de los pares imagen-máscara

## Notas Importantes

1. Asegurarse de que las imágenes y máscaras correspondientes tengan nombres coincidentes
2. Mantener la consistencia en el formato y estructura de los datos
3. Verificar la calidad de las máscaras de segmentación
4. Documentar cualquier característica especial del conjunto de datos

## Example Data

The `examples/` directory contains dummy data files that can be used for:
- Testing the data pipeline
- Verifying the model's basic functionality
- Running quick experiments

## Data Configuration

The data loading configuration can be found in:
- `configs/data/default.yaml`: Main data configuration
- `configs/data/transform.yaml`: Data augmentation settings
- `configs/data/dataloader.yaml`: DataLoader parameters 