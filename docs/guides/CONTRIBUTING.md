# Guía de Contribución

Gracias por tu interés en contribuir al proyecto CrackSeg. Este documento proporciona directrices para contribuir al proyecto de manera efectiva, asegurando la calidad y la coherencia del código.

## Índice

- [Configuración del Entorno](#configuración-del-entorno)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Flujo de Desarrollo](#flujo-de-desarrollo)
- [Estilo de Código](#estilo-de-código)
- [Pruebas](#pruebas)
- [Documentación](#documentación)
- [Envío de Cambios](#envío-de-cambios)
- [Revisión de Código](#revisión-de-código)

## Configuración del Entorno

1. **Clonar el Repositorio**:

   ```bash
   git clone https://github.com/tu-usuario/crackseg.git
   cd crackseg
   ```

2. **Crear Entorno Conda**:

   ```bash
   conda env create -f environment.yml
   conda activate torch
   ```

3. **Configuración para Desarrollo**:

   ```bash
   # Instalar dependencias en modo desarrollo
   pip install -e .

   # Configurar herramientas de linting
   python scripts/utils/lint_manager.py hooks
   ```

## Estructura del Proyecto

El proyecto está organizado de manera modular para facilitar la extensibilidad:

```txt
crackseg/
├── configs/             # Configuraciones Hydra
│   ├── data/            # Configuración del dataset y dataloaders
│   ├── model/           # Configuración de arquitecturas y componentes
│   └── training/        # Configuración de entrenamiento y evaluación
├── data/                # Directorio para datos (git-ignorado)
├── docs/                # Documentación
├── scripts/             # Scripts de utilidad
├── src/                 # Código fuente principal
│   ├── data/            # Módulos de datos y transformaciones
│   ├── model/           # Implementación de modelos y componentes
│   │   ├── architectures/  # Arquitecturas completas
│   │   ├── encoder/     # Módulos codificadores
│   │   ├── decoder/     # Módulos decodificadores
│   │   └── bottleneck/  # Módulos de cuello de botella
│   ├── training/        # Módulos de entrenamiento
│   └── utils/           # Utilidades comunes
└── tests/               # Pruebas unitarias e integración
```

### Principios de Organización

1. **Modularidad**: Cada componente debe tener una responsabilidad única y bien definida
2. **Extensibilidad**: Facilitar la adición de nuevos componentes sin modificar el código existente
3. **Configurabilidad**: Todos los parámetros deben ser configurables a través de archivos YAML

## Flujo de Desarrollo

### 1. Planificación

Antes de comenzar a codificar:

- Revisa los issues existentes y elige uno para trabajar o crea uno nuevo
- Discute el enfoque con el equipo si es un cambio significativo
- Define claramente el alcance del cambio

### 2. Desarrollo

Sigue estos pasos para cada contribución:

1. **Crear una Rama**:

   ```bash
   # Para nuevas características
   git checkout -b feature/nombre-caracteristica

   # Para correcciones de errores
   git checkout -b fix/nombre-error
   ```

2. **Implementación**:
   - Escribe código limpio y bien documentado
   - Sigue las convenciones de estilo (ver [Estilo de Código](#estilo-de-código))
   - Incluye pruebas para la nueva funcionalidad

3. **Pruebas Locales**:

   ```bash
   # Ejecutar pruebas
   pytest

   # Verificar estilo de código
   python scripts/utils/lint_manager.py full
   ```

### 3. Envío de Cambios

1. **Commit de Cambios**:

   ```bash
   git add .
   git commit -m "tipo(alcance): descripción corta"

   ```txt
   Seguimos la convención de [Commits Convencionales](https://www.conventionalcommits.org/):
   - `feat`: Nueva característica
   - `fix`: Corrección de error
   - `docs`: Cambios en documentación
   - `style`: Cambios que no afectan el significado del código
   - `refactor`: Refactorización del código
   - `test`: Adición o corrección de pruebas
   - `chore`: Cambios en el proceso de construcción o herramientas auxiliares

2. **Actualizar la Rama Principal**:

   ```bash
   git fetch origin
   git rebase origin/main
   ```

3. **Enviar Cambios**:

   ```bash
   git push origin nombre-rama
   ```

4. **Crear Pull Request**:
   - Usa el formato: `tipo(alcance): descripción`
   - Incluye una descripción detallada de los cambios
   - Enlaza el issue relacionado usando `Fixes #issueNum`

## Estilo de Código

Seguimos una versión personalizada de PEP 8 con algunas especificaciones adicionales:

### Generales

- **Longitud de línea**: Máximo 79 caracteres
- **Indentación**: 4 espacios (no tabs)
- **Longitud máxima de archivo**: 400 líneas
- **Longitud máxima de función**: 40 líneas

### Nomenclatura

- **Módulos**: `snake_case`
- **Clases**: `CamelCase`
- **Funciones/Variables**: `snake_case`
- **Constantes**: `MAYUSCULAS_CON_GUIONES`

### Docstrings

Usamos docstrings en formato NumPy para documentar todos los módulos, clases y funciones:

```python
def function_example(param1, param2):
    """
    Breve descripción en una línea.

    Descripción más detallada que puede
    abarcar múltiples líneas.

    Parameters
    ----------
    param1 : type
        Descripción del parámetro 1
    param2 : type
        Descripción del parámetro 2

    Returns
    -------
    type
        Descripción del valor de retorno

    Raises
    ------
    ExceptionType
        Descripción de cuándo se lanza la excepción
    """
    # Implementación
```

### Herramientas de Linting

Usamos varias herramientas para mantener la calidad del código:

1. **Pylint**: Para análisis estático de código

   ```bash
   python scripts/utils/lint_manager.py critical  # Verificar errores críticos
   python scripts/utils/lint_manager.py full      # Verificación completa
   ```

2. **Black**: Para formateo automático

   ```bash
   python scripts/utils/lint_manager.py format
   ```

3. **isort**: Para ordenar importaciones

   ```bash
   isort .
   ```

## Pruebas

Todos los cambios deben incluir pruebas adecuadas:

### Tipos de Pruebas

1. **Pruebas Unitarias**: Para componentes individuales
2. **Pruebas de Integración**: Para interacciones entre componentes
3. **Pruebas End-to-End**: Para flujos completos (opcional)

### Ejecución de Pruebas

```bash
# Todas las pruebas
pytest

# Pruebas específicas
pytest tests/unit/
pytest tests/integration/

# Con cobertura
pytest --cov=src
```

### Pautas para Pruebas

- Cada módulo debe tener al menos 70% de cobertura
- Las pruebas deben ser independientes (no depender de otras pruebas)
- Utiliza fixtures y mocks para datos y dependencias
- Crea casos de prueba para caminos felices y casos de borde

## Documentación

La documentación es fundamental para la mantenibilidad del proyecto:

### Tipos de Documentación

1. **Docstrings**: Para API interna y uso programático
2. **Archivos Markdown**: Para guías de usuario y desarrollador
3. **Ejemplos de Código**: Para demostrar el uso

### Pautas para Documentación

- Actualiza la documentación junto con los cambios de código
- Usa lenguaje claro y conciso
- Incluye ejemplos prácticos cuando sea posible
- Escribe documentación pensando en nuevos usuarios

## Envío de Cambios

Para enviar contribuciones al proyecto:

1. Asegúrate de que todas las pruebas pasen
2. Actualiza la documentación si es necesario
3. Crea un pull request con una descripción clara

## Revisión de Código

Todas las contribuciones pasan por un proceso de revisión:

### Proceso de Revisión

1. **Verificación Automática**:
   - Las pruebas deben pasar
   - El análisis de código debe ser aprobado
   - La cobertura de pruebas debe cumplir con los umbrales

2. **Revisión Manual**:
   - Al menos un revisor debe aprobar el PR
   - Se pueden solicitar cambios antes de la aprobación

### Criterios de Revisión

- **Funcionalidad**: ¿El código hace lo que se supone que debe hacer?
- **Calidad**: ¿El código es limpio, eficiente y mantenible?
- **Pruebas**: ¿Hay pruebas adecuadas para los cambios?
- **Documentación**: ¿Los cambios están bien documentados?

---

Agradecemos tu contribución al proyecto CrackSeg. Si tienes alguna pregunta o sugerencia sobre estas directrices, por favor abre un issue o contacta al equipo de mantenimiento.
