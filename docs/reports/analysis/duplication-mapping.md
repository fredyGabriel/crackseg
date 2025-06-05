# Mapeo Exacto de Duplicaciones

**Fase 1 de Consolidación**: Identificación precisa de contenido duplicado

## 🔍 **Duplicaciones en `always_applied_workspace_rules`**

### 1. **Estándares de Calidad de Código** (DUPLICACIÓN COMPLETA)

**Origen**: `coding-preferences.mdc` líneas 1-200+
**Duplicado en**: `always_applied_workspace_rules` ~300 líneas

**Contenido duplicado exacto**:

- Sección "Code Quality Standards (Mandatory)"
- Reglas de Type checking: `basedpyright .`
- Formatting: `black .`
- Linting: `ruff .`
- "Type annotations are mandatory for all code"
- Sección completa "Modern Generic Type Syntax (Python 3.12+ PEP 695)"
- "Built-in Generic Types (Python 3.9+ PEP 585)"
- Ejemplos de código idénticos
- "Pre-commit workflow" con comandos bash exactos
- Sección "Code Structure and Organization"
- "Documentation and Comments"
- "Reliability and Error Handling"
- "Configuration and Dependencies"
- "Advanced Type Patterns for Python 3.12+"

### 2. **Development Workflow Guidelines** (DUPLICACIÓN MASIVA)

**Origen**: `workflow-preferences.mdc` líneas 1-125
**Duplicado en**: `always_applied_workspace_rules` ~200 líneas

**Contenido duplicado exacto**:

- "Development Workflow Guidelines" título y descripción
- "Planning and Analysis" sección completa
- "Analyze Three Solution Options" princicipio
- "Implementation Principles" completo
- "Quality Assurance" sección
- "Project Structure and Documentation"
- "Error Resolution and Communication"
- "Workflow Integration" con Pre-Commit Checklist
- "Professional Solution Analysis" párrafo completo

### 3. **Task Master Development Workflow** (DUPLICACIÓN MEGA)

**Origen**: `dev_workflow.mdc` líneas 1-210
**Duplicado en**: `always_applied_workspace_rules` ~400 líneas

**Contenido duplicado exacto**:

- "Task Master Development Workflow" título completo
- "Primary Interaction: MCP Server vs. CLI" sección entera
- "Standard Development Workflow Process" lista completa de ~20 elementos
- "Task Complexity Analysis" sección
- "Task Breakdown Process"
- "Implementation Drift Handling"
- "Task Status Management"
- "Task Structure Fields" con todos los ejemplos
- "Environment Variables Configuration" lista completa
- "Determining the Next Task"
- "Viewing Specific Task Details"
- "Managing Task Dependencies"
- "Iterative Subtask Implementation" proceso de 10 pasos

### 4. **Cursor Rule Creation Guidelines** (DUPLICACIÓN PARCIAL)

**Origen**: `cursor_rules.mdc` líneas 1-144
**Duplicado en**: `always_applied_workspace_rules` ~100 líneas

**Contenido duplicado**:

- "Cursor Rule Creation Guidelines" principios
- "Required Rule Structure" estándares
- "Content Guidelines" con ejemplos de código
- "Rule Categories and Organization"

## 📊 **Métricas de Duplicación**

| Archivo Origen | Líneas Origen | Líneas Duplicadas | % Duplicación | Impacto |
|----------------|---------------|-------------------|---------------|---------|
| `coding-preferences.mdc` | 244 | ~300 | 123% | CRÍTICO |
| `workflow-preferences.mdc` | 125 | ~200 | 160% | ALTO |
| `dev_workflow.mdc` | 210 | ~400 | 190% | MEGA |
| `cursor_rules.mdc` | 144 | ~100 | 69% | MODERADO |
| **TOTAL** | **723** | **~1000** | **138%** | **MASIVO** |

## 🎯 **Contenido que DEBE permanecer en `always_applied_workspace_rules`**

### Reglas Críticas Mínimas (3-4 líneas máximo)

```
- **Código Python**: Debe pasar basedpyright, black, y ruff antes del commit
- **Type annotations**: Obligatorias usando generics Python 3.12+ modernos
- **Referencias**: Ver [consolidated-workspace-rules.mdc](mdc:.cursor/rules/consolidated-workspace-rules.mdc)
- **Responder en español**: Usar inglés solo para código
```

### Todo lo demás → ELIMINAR y referenciar archivos específicos

## 🚀 **Plan de Eliminación**

1. **Reducir `always_applied_workspace_rules`** de ~1000 líneas a ~50 líneas
2. **Eliminar duplicaciones** internas entre archivos .mdc
3. **Optimizar referencias** en `consolidated-workspace-rules.mdc`
4. **Validar navegación** completa

**Reducción objetivo**: **95% menos contenido duplicado**
