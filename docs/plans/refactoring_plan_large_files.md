# Plan de Refactorización - Archivos de Código Fuente con >500 líneas

## **Estado del Proyecto: PLANIFICACIÓN**

**Fecha de Creación:** 2025-01-27
**Responsable:** Equipo de Desarrollo CrackSeg
**Prioridad:** ALTA
**Relacionado con:** Sistema de Artefactos (artifact-system)

---

## **Resumen Ejecutivo**

Este plan aborda la refactorización de 29 archivos de código fuente que exceden las 500 líneas de
código, violando las reglas del proyecto que establecen límites de 300-400 líneas por archivo. El
objetivo es mejorar la mantenibilidad, legibilidad y modularidad del código sin comprometer la
funcionalidad existente.

### **Objetivos Principales**

- ✅ **Reducir complejidad**: Dividir archivos grandes en módulos cohesivos
- ✅ **Mejorar mantenibilidad**: Facilitar futuras modificaciones y debugging
- ✅ **Preservar funcionalidad**: Mantener todas las características existentes
- ✅ **Optimizar rendimiento**: Mejorar la eficiencia del código
- ✅ **Facilitar testing**: Hacer el código más testeable

### **Métricas de Éxito**

- [ ] **Límite de líneas**: Todos los archivos < 400 líneas
- [ ] **Cobertura de tests**: Mantener >90% de cobertura
- [ ] **Funcionalidad**: 100% de tests pasando
- [ ] **Performance**: Sin degradación de rendimiento
- [ ] **Calidad de código**: Pasar todas las quality gates

---

## **Análisis de Prioridades**

### **🔴 Alta Prioridad (>800 líneas)**

Estos archivos representan el mayor riesgo y deben ser refactorizados primero:

#### **1. `src/crackseg/utils/deployment/orchestration.py` (882 líneas)**

- **Importancia**: Sistema crítico de despliegue
- **Cohesión**: Baja - mezcla múltiples responsabilidades
- **Riesgo**: Alto - cambios pueden afectar despliegues en producción
- **Estrategia**: Dividir en módulos especializados

#### **2. `src/crackseg/model/decoder/cnn_decoder.py` (878 líneas)**

- **Importancia**: Componente central del modelo
- **Cohesión**: Media - lógica de decodificación compleja
- **Riesgo**: Alto - cambios pueden afectar rendimiento del modelo
- **Estrategia**: Separar por tipos de decodificadores

#### **3. `gui/utils/process/manager_backup.py` (802 líneas)**

- **Importancia**: Gestión de procesos de GUI
- **Cohesión**: Baja - múltiples responsabilidades mezcladas
- **Riesgo**: Medio - afecta interfaz de usuario
- **Estrategia**: Dividir por funcionalidades específicas

### **🟡 Media Prioridad (700-800 líneas)**

#### **4. `src/crackseg/data/dataset.py` (799 líneas)**

- **Importancia**: Pipeline de datos crítico
- **Cohesión**: Media - lógica de dataset compleja
- **Riesgo**: Alto - afecta entrenamiento de modelos
- **Estrategia**: Separar por tipos de datasets

#### **5. `src/crackseg/utils/deployment/artifact_optimizer.py` (742 líneas)**

- **Importancia**: Optimización para producción
- **Cohesión**: Media - múltiples técnicas de optimización
- **Riesgo**: Medio - afecta preparación para despliegue
- **Estrategia**: Dividir por técnicas de optimización

#### **6. `src/crackseg/model/core/unet.py` (698 líneas)**

- **Importancia**: Arquitectura central del modelo
- **Cohesión**: Alta - lógica cohesiva de UNet
- **Riesgo**: Alto - cambios pueden afectar rendimiento
- **Estrategia**: Mantener cohesión, optimizar estructura interna

#### **7. `src/crackseg/utils/deployment/validation_pipeline.py` (687 líneas)**

- **Importancia**: Validación de despliegues
- **Cohesión**: Baja - múltiples tipos de validación
- **Riesgo**: Medio - afecta calidad de despliegues
- **Estrategia**: Separar por tipos de validación

#### **8. `src/main.py` (673 líneas)**

- **Importancia**: Punto de entrada principal
- **Cohesión**: Baja - orquesta múltiples componentes
- **Riesgo**: Alto - afecta toda la aplicación
- **Estrategia**: Extraer lógica a módulos especializados

### **🟢 Baja Prioridad (500-700 líneas)**

Los 21 archivos restantes se refactorizarán después de completar los de alta y media prioridad.

---

## **Estrategias de Refactorización**

### **Patrón 1: División por Responsabilidades**

**Aplicable a:** `orchestration.py`, `manager_backup.py`, `validation_pipeline.py`

**Estrategia:**

1. Identificar responsabilidades distintas
2. Crear módulos especializados
3. Mantener interfaz unificada
4. Implementar tests de integración

**Ejemplo para `orchestration.py`:**

```bash
orchestration/
├── __init__.py
├── deployment_manager.py      # Gestión de despliegues
├── rollback_manager.py        # Gestión de rollbacks
├── alert_manager.py          # Sistema de alertas
├── performance_monitor.py    # Monitoreo de performance
└── strategies/              # Estrategias de despliegue
    ├── __init__.py
    ├── blue_green.py
    ├── canary.py
    ├── rolling.py
    └── recreate.py
```

### **Patrón 2: División por Tipos**

**Aplicable a:** `cnn_decoder.py`, `dataset.py`, `artifact_optimizer.py`

**Estrategia:**

1. Identificar tipos distintos de funcionalidad
2. Crear módulos por tipo
3. Mantener abstracciones comunes
4. Implementar factory patterns

**Ejemplo para `cnn_decoder.py`:**

```bash
decoder/
├── __init__.py
├── base_decoder.py           # Clase base abstracta
├── cnn_decoder.py           # Decodificador CNN principal
├── attention_decoder.py      # Decodificador con atención
├── skip_connection.py       # Gestión de conexiones skip
└── upsampling.py           # Técnicas de upsampling
```

### **Patrón 3: Extracción de Utilidades**

**Aplicable a:** `main.py`, `core_validator.py`, `error_handling.py`

**Estrategia:**

1. Identificar funciones utilitarias
2. Extraer a módulos de utilidades
3. Mantener lógica principal en archivo original
4. Implementar imports limpios

### **Patrón 4: Optimización de Estructura**

**Aplicable a:** `unet.py`, `trainer.py`, `transforms.py`

**Estrategia:**

1. Reorganizar métodos por funcionalidad
2. Extraer constantes y configuraciones
3. Optimizar imports y dependencias
4. Mantener cohesión de la clase principal

---

## **Plan de Implementación Detallado**

### **Fase 1: Preparación (Semana 1)**

#### **Tareas:**

1. **Análisis detallado de cada archivo**
   - [ ] Identificar responsabilidades principales
   - [ ] Mapear dependencias internas
   - [ ] Identificar puntos de acoplamiento
   - [ ] Documentar funcionalidades críticas

2. **Crear tests de regresión**
   - [ ] Tests unitarios para cada funcionalidad
   - [ ] Tests de integración para flujos completos
   - [ ] Tests de performance para operaciones críticas
   - [ ] Tests de edge cases

3. **Establecer baseline de métricas**
   - [ ] Medir tiempo de ejecución actual
   - [ ] Medir uso de memoria
   - [ ] Documentar cobertura de tests
   - [ ] Establecer métricas de calidad

#### **Criterios de Éxito:**

- [ ] Análisis completo de los 29 archivos
- [ ] Tests de regresión implementados
- [ ] Baseline de métricas establecido
- [ ] Plan detallado para cada archivo

### **Fase 2: Refactorización de Alta Prioridad (Semanas 2-3)**

#### **Semana 2: Archivos >800 líneas**

**2.1. `orchestration.py` (882 líneas)**

- **Objetivo**: Dividir en 5-6 módulos especializados
- **Estrategia**: Patrón 1 - División por responsabilidades
- **Módulos resultantes**:
  - `deployment_manager.py` (~200 líneas)
  - `rollback_manager.py` (~150 líneas)
  - `alert_manager.py` (~120 líneas)
  - `performance_monitor.py` (~150 líneas)
  - `strategies/` (~100 líneas)
  - `orchestration.py` (~60 líneas - interfaz unificada)

**2.2. `cnn_decoder.py` (878 líneas)**

- **Objetivo**: Dividir en módulos por tipos de decodificación
- **Estrategia**: Patrón 2 - División por tipos
- **Módulos resultantes**:
  - `base_decoder.py` (~150 líneas)
  - `cnn_decoder.py` (~300 líneas)
  - `attention_decoder.py` (~200 líneas)
  - `skip_connection.py` (~100 líneas)
  - `upsampling.py` (~128 líneas)

**2.3. `manager_backup.py` (802 líneas)**

- **Objetivo**: Dividir por funcionalidades de GUI
- **Estrategia**: Patrón 1 - División por responsabilidades
- **Módulos resultantes**:
  - `process_manager.py` (~200 líneas)
  - `backup_manager.py` (~150 líneas)
  - `state_manager.py` (~120 líneas)
  - `ui_manager.py` (~150 líneas)
  - `manager_backup.py` (~182 líneas - coordinación)

#### **Semana 3: Archivos 700-800 líneas**

**3.1. `dataset.py` (799 líneas)**

- **Objetivo**: Dividir por tipos de datasets
- **Estrategia**: Patrón 2 - División por tipos
- **Módulos resultantes**:
  - `base_dataset.py` (~150 líneas)
  - `crack_dataset.py` (~200 líneas)
  - `augmented_dataset.py` (~150 líneas)
  - `validation_dataset.py` (~100 líneas)
  - `dataset_factory.py` (~100 líneas)
  - `dataset.py` (~99 líneas - interfaz unificada)

**3.2. `artifact_optimizer.py` (742 líneas)**

- **Objetivo**: Dividir por técnicas de optimización
- **Estrategia**: Patrón 2 - División por tipos
- **Módulos resultantes**:
  - `base_optimizer.py` (~120 líneas)
  - `quantization_optimizer.py` (~150 líneas)
  - `pruning_optimizer.py` (~120 líneas)
  - `compression_optimizer.py` (~100 líneas)
  - `benchmark_optimizer.py` (~150 líneas)
  - `artifact_optimizer.py` (~102 líneas - coordinación)

**3.3. `unet.py` (698 líneas)**

- **Objetivo**: Optimizar estructura manteniendo cohesión
- **Estrategia**: Patrón 4 - Optimización de estructura
- **Módulos resultantes**:
  - `unet.py` (~400 líneas - clase principal)
  - `unet_blocks.py` (~150 líneas - bloques especializados)
  - `unet_config.py` (~148 líneas - configuraciones)

**3.4. `validation_pipeline.py` (687 líneas)**

- **Objetivo**: Dividir por tipos de validación
- **Estrategia**: Patrón 1 - División por responsabilidades
- **Módulos resultantes**:
  - `functional_validator.py` (~150 líneas)
  - `performance_validator.py` (~120 líneas)
  - `security_validator.py` (~100 líneas)
  - `compatibility_validator.py` (~120 líneas)
  - `validation_pipeline.py` (~197 líneas - coordinación)

**3.5. `main.py` (673 líneas)**

- **Objetivo**: Extraer lógica a módulos especializados
- **Estrategia**: Patrón 3 - Extracción de utilidades
- **Módulos resultantes**:
  - `main.py` (~200 líneas - punto de entrada)
  - `pipeline_orchestrator.py` (~150 líneas)
  - `config_manager.py` (~120 líneas)
  - `experiment_runner.py` (~203 líneas)

#### **Criterios de Éxito Fase 2:**

- [ ] Todos los archivos >700 líneas refactorizados
- [ ] Tests de regresión pasando al 100%
- [ ] Performance mantenida o mejorada
- [ ] Cobertura de tests >90%
- [ ] Quality gates pasando

### **Fase 3: Refactorización de Media Prioridad (Semanas 4-5)**

#### **Semana 4: Archivos 600-700 líneas**

**4.1. `validation.py` (643 líneas)**

- **Objetivo**: Dividir por tipos de validación de datos
- **Estrategia**: Patrón 2 - División por tipos

**4.2. `core_validator.py` (639 líneas)**

- **Objetivo**: Dividir por esquemas de validación
- **Estrategia**: Patrón 1 - División por responsabilidades

**4.3. `environment_configurator.py` (633 líneas)**

- **Objetivo**: Dividir por tipos de configuración
- **Estrategia**: Patrón 2 - División por tipos

**4.4. `factory.py` (626 líneas)**

- **Objetivo**: Dividir por tipos de factories
- **Estrategia**: Patrón 2 - División por tipos

**4.5. `advanced_validation.py` (607 líneas)**

- **Objetivo**: Dividir por tipos de validación avanzada
- **Estrategia**: Patrón 1 - División por responsabilidades

#### **Semana 5: Archivos 500-600 líneas**

**5.1. `validation_reporter.py` (602 líneas)**

- **Objetivo**: Dividir por tipos de reportes
- **Estrategia**: Patrón 2 - División por tipos

**5.2. `error_handling.py` (600 líneas)**

- **Objetivo**: Dividir por tipos de manejo de errores
- **Estrategia**: Patrón 1 - División por responsabilidades

**5.3. `coverage_monitor.py` (575 líneas)**

- **Objetivo**: Dividir por tipos de monitoreo
- **Estrategia**: Patrón 2 - División por tipos

**5.4. `swinv2_cnn_aspp_unet.py` (574 líneas)**

- **Objetivo**: Optimizar estructura manteniendo cohesión
- **Estrategia**: Patrón 4 - Optimización de estructura

**5.5. `core.py` (574 líneas)**

- **Objetivo**: Dividir por funcionalidades de checkpointing
- **Estrategia**: Patrón 1 - División por responsabilidades

### **Fase 4: Refactorización de Baja Prioridad (Semanas 6-7)**

#### **Semana 6: Archivos 500-550 líneas**

Refactorización de los archivos restantes siguiendo los patrones establecidos.

#### **Semana 7: Validación y Documentación**

- [ ] Validación final de todos los cambios
- [ ] Documentación de la nueva estructura
- [ ] Guías de migración
- [ ] Actualización de documentación técnica

---

## **Criterios de Éxito por Fase**

### **Criterios Generales:**

- [ ] **Límite de líneas**: Todos los archivos < 400 líneas
- [ ] **Funcionalidad**: 100% de tests pasando
- [ ] **Performance**: Sin degradación de rendimiento
- [ ] **Calidad**: Pasar todas las quality gates
- [ ] **Cobertura**: Mantener >90% de cobertura de tests

### **Criterios Específicos por Fase:**

#### **Fase 1 - Preparación:**

- [ ] Análisis completo de los 29 archivos
- [ ] Tests de regresión implementados
- [ ] Baseline de métricas establecido
- [ ] Plan detallado para cada archivo

#### **Fase 2 - Alta Prioridad:**

- [ ] 8 archivos >700 líneas refactorizados
- [ ] Tests de regresión pasando al 100%
- [ ] Performance mantenida o mejorada
- [ ] Cobertura de tests >90%

#### **Fase 3 - Media Prioridad:**

- [ ] 10 archivos 600-700 líneas refactorizados
- [ ] Integración funcional completa
- [ ] Documentación de cambios
- [ ] Guías de migración

#### **Fase 4 - Baja Prioridad:**

- [ ] 11 archivos 500-600 líneas refactorizados
- [ ] Validación final completa
- [ ] Documentación actualizada
- [ ] Código listo para producción

---

## **Riesgos y Mitigaciones**

### **Riesgos Identificados:**

1. **Pérdida de funcionalidad** durante la refactorización
2. **Degradación de performance** por overhead de imports
3. **Conflicto con desarrollo activo** del sistema de artefactos
4. **Complejidad de testing** con nueva estructura modular

### **Estrategias de Mitigación:**

1. **Desarrollo incremental** con validación continua
2. **Tests exhaustivos** antes y después de cada cambio
3. **Coordinación con equipo** de desarrollo de artefactos
4. **Documentación detallada** de cambios y migraciones

---

## **Recursos Requeridos**

### **Desarrollo:**

- 1 Ingeniero Senior (7 semanas)
- 1 Ingeniero de Testing (3 semanas)
- 1 Técnico de Documentación (1 semana)

### **Infraestructura:**

- Entorno de testing aislado
- Herramientas de profiling de performance
- Sistema de CI/CD para validación continua

---

## **Cronograma Detallado**

| Semana | Fase | Archivos Objetivo | Entregables |
|--------|------|-------------------|-------------|
| 1 | Preparación | Análisis de 29 archivos | Plan detallado, tests baseline |
| 2 | Alta Prioridad | 3 archivos >800 líneas | Módulos refactorizados |
| 3 | Alta Prioridad | 5 archivos 700-800 líneas | Módulos refactorizados |
| 4 | Media Prioridad | 5 archivos 600-700 líneas | Módulos refactorizados |
| 5 | Media Prioridad | 5 archivos 500-600 líneas | Módulos refactorizados |
| 6 | Baja Prioridad | 11 archivos restantes | Módulos refactorizados |
| 7 | Validación | Todos los archivos | Documentación, guías |

---

## **Integración con Task Master**

### **Tag de Proyecto:** `refactoring-large-files`

### **Estructura de Tareas Sugerida:**

| ID | Tarea | Dependencias | Estado |
|----|-------|--------------|--------|
| 1 | Preparación y Análisis | None | ⏳ Pending |
| 2 | Refactorización Alta Prioridad | 1 | ⏳ Pending |
| 3 | Refactorización Media Prioridad | 2 | ⏳ Pending |
| 4 | Refactorización Baja Prioridad | 3 | ⏳ Pending |
| 5 | Validación y Documentación | 4 | ⏳ Pending |

### **Comandos de Seguimiento:**

```bash
# Crear tareas de refactorización
task-master add --prompt="Preparación y análisis de archivos grandes" --file="C:/Users/fgrv/Dev/CursorProjects/crackseg/.taskmaster/tasks/tasks.json"

# Ver estado actual
task-master list --file="C:/Users/fgrv/Dev/CursorProjects/crackseg/.taskmaster/tasks/tasks.json"

# Marcar tarea como en progreso
task-master set-status --id=1 --status=in-progress --file="C:/Users/fgrv/Dev/CursorProjects/crackseg/.taskmaster/tasks/tasks.json"
```

---

## **Aprobaciones**

| Rol | Nombre | Fecha | Firma |
|-----|--------|-------|-------|
| Product Owner | [Por definir] | [Fecha] | [Firma] |
| Tech Lead | [Por definir] | [Fecha] | [Firma] |
| Project Manager | [Por definir] | [Fecha] | [Firma] |

---

**Documento creado:** 2025-01-27
**Última actualización:** 2025-01-27
**Versión:** 1.0
**Estado:** PLANIFICACIÓN
