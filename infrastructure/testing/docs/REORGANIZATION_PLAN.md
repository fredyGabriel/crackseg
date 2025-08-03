# Plan de Reorganización de Infraestructura Docker

## Análisis de la Situación Actual

### Ubicación Actual

- **Carpeta**: `/crackseg/docker/`
- **Propósito**: Infraestructura Docker para testing, deployment y health checks
- **Contenido**: 28 archivos, 5 subdirectorios, 69,000+ líneas de código

### Problemas Identificados

1. **Ubicación Inadecuada**: `docker/` en raíz no refleja el propósito completo
2. **Organización Interna**: Mezcla de responsabilidades (testing, deployment, health checks)
3. **Nomenclatura**: No sigue convenciones modernas de proyectos ML
4. **Escalabilidad**: Estructura actual dificulta agregar nuevos tipos de infraestructura

## Propuesta de Reorganización

### Nueva Estructura Recomendada

```bash
crackseg/
├── infrastructure/                    # Nueva carpeta principal
│   ├── testing/                      # Infraestructura de testing
│   │   ├── docker/                   # Dockerfiles y configuraciones
│   │   │   ├── Dockerfile.test-runner
│   │   │   ├── Dockerfile.streamlit
│   │   │   ├── docker-compose.test.yml
│   │   │   └── .dockerignore
│   │   ├── scripts/                  # Scripts de testing
│   │   │   ├── run-test-runner.sh
│   │   │   ├── e2e-test-orchestrator.sh
│   │   │   ├── browser-manager.sh
│   │   │   └── artifact-manager.sh
│   │   ├── config/                   # Configuraciones de testing
│   │   │   ├── pytest.ini
│   │   │   ├── test-runner.config
│   │   │   ├── grid-config.json
│   │   │   └── browser-capabilities.json
│   │   ├── health_check/             # Sistema de health checks
│   │   │   ├── __init__.py
│   │   │   ├── analytics/
│   │   │   ├── checkers/
│   │   │   ├── cli/
│   │   │   ├── models/
│   │   │   ├── orchestration/
│   │   │   └── persistence/
│   │   └── docs/                     # Documentación de testing
│   │       ├── README.md
│   │       ├── README-ARCHITECTURE.md
│   │       ├── README-USAGE.md
│   │       └── README-TROUBLESHOOTING.md
│   ├── deployment/                   # Infraestructura de deployment
│   │   ├── docker/                   # Dockerfiles de deployment
│   │   ├── kubernetes/               # Manifests de Kubernetes
│   │   ├── scripts/                  # Scripts de deployment
│   │   └── config/                   # Configuraciones de deployment
│   ├── monitoring/                   # Infraestructura de monitoreo
│   │   ├── scripts/
│   │   ├── config/
│   │   └── dashboards/
│   └── shared/                       # Recursos compartidos
│       ├── scripts/                  # Scripts utilitarios
│       │   ├── setup-local-dev.sh
│       │   ├── setup-env.sh
│       │   ├── system-monitor.sh
│       │   └── network-manager.sh
│       ├── config/                   # Configuraciones compartidas
│       │   ├── env.local.template
│       │   ├── env.test.template
│       │   ├── env.staging.template
│       │   └── env.production.template
│       └── docs/                     # Documentación compartida
│           ├── README-LOCAL-DEV.md
│           ├── README.environment-management.md
│           └── README.network-setup.md
```

## Justificación de la Reorganización

### 1. **Separación de Responsabilidades**

**Testing (`infrastructure/testing/`)**:

- Dockerfiles específicos para testing
- Scripts de ejecución de tests
- Configuraciones de Selenium Grid
- Sistema de health checks para testing

**Deployment (`infrastructure/deployment/`)**:

- Dockerfiles de producción
- Manifests de Kubernetes
- Scripts de deployment
- Configuraciones de entorno

**Monitoring (`infrastructure/monitoring/`)**:

- Scripts de monitoreo
- Dashboards y métricas
- Alertas y notificaciones

**Shared (`infrastructure/shared/`)**:

- Scripts utilitarios comunes
- Configuraciones de entorno
- Documentación compartida

### 2. **Ventajas de la Nueva Estructura**

#### **Escalabilidad**

- Fácil agregar nuevos tipos de infraestructura
- Separación clara por dominio
- Reutilización de componentes compartidos

#### **Mantenibilidad**

- Responsabilidades bien definidas
- Documentación organizada por dominio
- Scripts agrupados por propósito

#### **Profesionalismo**

- Sigue convenciones de proyectos ML modernos
- Estructura intuitiva para nuevos desarrolladores
- Facilita CI/CD y automatización

### 3. **Migración de Archivos**

#### **Fase 1: Crear Nueva Estructura**

```bash
# Crear directorios principales
mkdir -p infrastructure/{testing,deployment,monitoring,shared}

# Crear subdirectorios
mkdir -p infrastructure/testing/{docker,scripts,config,health_check,docs}
mkdir -p infrastructure/deployment/{docker,kubernetes,scripts,config}
mkdir -p infrastructure/monitoring/{scripts,config,dashboards}
mkdir -p infrastructure/shared/{scripts,config,docs}
```

#### **Fase 2: Migrar Archivos por Categoría**

**Testing (`infrastructure/testing/`)**:

```bash
# Dockerfiles y configuraciones
mv docker/Dockerfile.test-runner infrastructure/testing/docker/
mv docker/Dockerfile.streamlit infrastructure/testing/docker/
mv docker/docker-compose.test.yml infrastructure/testing/docker/
mv docker/.dockerignore infrastructure/testing/docker/

# Scripts de testing
mv docker/scripts/run-test-runner.sh infrastructure/testing/scripts/
mv docker/scripts/e2e-test-orchestrator.sh infrastructure/testing/scripts/
mv docker/scripts/browser-manager.sh infrastructure/testing/scripts/
mv docker/scripts/artifact-manager.sh infrastructure/testing/scripts/

# Configuraciones
mv docker/pytest.ini infrastructure/testing/config/
mv docker/test-runner.config infrastructure/testing/config/
mv docker/grid-config.json infrastructure/testing/config/
mv docker/browser-capabilities.json infrastructure/testing/config/

# Health checks
mv docker/health_check/ infrastructure/testing/health_check/

# Documentación
mv docker/README.md infrastructure/testing/docs/
mv docker/README-ARCHITECTURE.md infrastructure/testing/docs/
mv docker/README-USAGE.md infrastructure/testing/docs/
mv docker/README-TROUBLESHOOTING.md infrastructure/testing/docs/
mv docker/README-DOCKER-TESTING.md infrastructure/testing/docs/
```

**Shared (`infrastructure/shared/`)**:

```bash
# Scripts utilitarios
mv docker/scripts/setup-local-dev.sh infrastructure/shared/scripts/
mv docker/scripts/setup-env.sh infrastructure/shared/scripts/
mv docker/scripts/system-monitor.sh infrastructure/shared/scripts/
mv docker/scripts/network-manager.sh infrastructure/shared/scripts/
mv docker/scripts/docker-stack-manager.sh infrastructure/shared/scripts/
mv docker/scripts/health-check-manager.sh infrastructure/shared/scripts/

# Configuraciones de entorno
mv docker/env.local.template infrastructure/shared/config/
mv docker/env.test.template infrastructure/shared/config/
mv docker/env.staging.template infrastructure/shared/config/
mv docker/env.production.template infrastructure/shared/config/

# Documentación compartida
mv docker/README-LOCAL-DEV.md infrastructure/shared/docs/
mv docker/README.environment-management.md infrastructure/shared/docs/
mv docker/README.network-setup.md infrastructure/shared/docs/
```

#### **Fase 3: Actualizar Referencias**

```bash
# Actualizar rutas en scripts
find infrastructure/ -name "*.sh" -exec sed -i 's|tests/docker/|infrastructure/testing/|g' {} \;
find infrastructure/ -name "*.sh" -exec sed -i 's|docker/|infrastructure/testing/docker/|g' {} \;

# Actualizar documentación
find infrastructure/ -name "*.md" -exec sed -i 's|tests/docker/|infrastructure/testing/|g' {} \;
```

### 4. **Actualización de CI/CD**

#### **GitHub Actions**

```yaml
# Actualizar rutas en workflows
- name: Run E2E Tests
  run: |
    cd infrastructure/testing
    ./scripts/run-test-runner.sh run
```

#### **Documentación**

```markdown
# Actualizar referencias en README principal
- [Testing Infrastructure](infrastructure/testing/docs/README.md)
- [Deployment Infrastructure](infrastructure/deployment/docs/README.md)
- [Monitoring Infrastructure](infrastructure/monitoring/docs/README.md)
```

## Beneficios Esperados

### 1. **Organización Profesional**

- Estructura clara y intuitiva
- Separación de responsabilidades
- Facilita onboarding de nuevos desarrolladores

### 2. **Escalabilidad**

- Fácil agregar nuevos tipos de infraestructura
- Reutilización de componentes compartidos
- Configuración modular

### 3. **Mantenibilidad**

- Scripts organizados por propósito
- Documentación estructurada
- Configuraciones centralizadas

### 4. **Integración CI/CD**

- Rutas claras y consistentes
- Scripts especializados por entorno
- Configuraciones de entorno separadas

## Plan de Implementación

### **Semana 1: Preparación**

1. Crear nueva estructura de directorios
2. Documentar plan de migración
3. Crear scripts de migración automatizada

### **Semana 2: Migración**

1. Migrar archivos por categoría
2. Actualizar referencias en scripts
3. Actualizar documentación

### **Semana 3: Validación**

1. Ejecutar tests completos
2. Validar CI/CD pipelines
3. Actualizar documentación del proyecto

### **Semana 4: Limpieza**

1. Eliminar carpeta `docker/` original
2. Actualizar `project_tree.md`
3. Comunicar cambios al equipo

## Conclusión

La reorganización propuesta transformará la infraestructura Docker de una estructura monolítica a
una arquitectura modular y profesional, siguiendo las mejores prácticas de proyectos ML modernos.
Esta estructura facilitará el mantenimiento, la escalabilidad y la integración con sistemas CI/CD.

---

**Nota**: Este plan debe ejecutarse de manera incremental para minimizar el riesgo y permitir
rollback si es necesario.
