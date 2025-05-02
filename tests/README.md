# Tests del Proyecto

Este directorio contiene las pruebas unitarias y de integración para el proyecto de segmentación de grietas.

## Estructura del Directorio

La estructura de pruebas refleja la estructura del código fuente:

```
tests/
├── data/           # Pruebas para módulos de datos
├── model/          # Pruebas para arquitecturas de modelos
├── training/       # Pruebas para lógica de entrenamiento
├── utils/          # Pruebas para utilidades
└── conftest.py     # Fixtures compartidos de pytest
```

## Tipos de Pruebas

### Pruebas Unitarias
- Pruebas de componentes individuales
- Mocking de dependencias
- Verificación de comportamiento aislado

### Pruebas de Integración
- Pruebas de interacción entre componentes
- Verificación de flujos completos
- Pruebas de configuración

## Ejecución de Pruebas

### Ejecutar todas las pruebas:
```bash
pytest
```

### Ejecutar pruebas específicas:
```bash
# Ejecutar pruebas de un módulo
pytest tests/data/

# Ejecutar una prueba específica
pytest tests/model/test_unet.py

# Ejecutar con cobertura
pytest --cov=src tests/
```

## Fixtures

Los fixtures comunes se encuentran en `conftest.py`:
- Datos de prueba
- Configuraciones mock
- Utilidades compartidas

## Buenas Prácticas

1. Mantener pruebas independientes
2. Usar nombres descriptivos para las pruebas
3. Documentar casos de prueba complejos
4. Mantener datos de prueba pequeños y representativos
5. Actualizar pruebas al modificar código

## Cobertura de Código

Se recomienda mantener una cobertura mínima del 80% en:
- Módulos de datos
- Arquitecturas de modelos
- Lógica de entrenamiento
- Utilidades críticas

## Organización de Pruebas

- Cada módulo debe tener su conjunto de pruebas
- Usar clases de prueba para agrupar casos relacionados
- Mantener pruebas enfocadas y específicas
- Documentar configuraciones especiales

## Ejemplos

```python
# Ejemplo de prueba unitaria
def test_dataset_loading():
    dataset = CrackDataset(...)
    assert len(dataset) > 0
    assert dataset[0]['image'].shape == (3, 512, 512)

# Ejemplo de prueba de integración
def test_training_loop():
    trainer = Trainer(...)
    trainer.train()
    assert trainer.metrics['val_loss'] < initial_loss
``` 