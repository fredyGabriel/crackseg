# CrackSeg Docker Infrastructure

> **⚠️ IMPORTANTE: Esta carpeta será reorganizada según el [Plan de Reorganización](REORGANIZATION_PLAN.md)**
>
> La infraestructura Docker actual será migrada a una estructura más profesional bajo `infrastructure/`

## Overview

La carpeta `docker/` contiene la infraestructura Docker completa para el proyecto CrackSeg,
incluyendo testing, deployment, health checks y monitoreo. Esta infraestructura proporciona un
entorno de testing E2E con Selenium Grid, scripts de deployment y sistemas de monitoreo.

## Estructura Actual

```bash
docker/
├── scripts/                    # Scripts de gestión y automatización
│   ├── run-test-runner.sh     # Ejecutor principal de tests
│   ├── e2e-test-orchestrator.sh # Orquestador de tests E2E
│   ├── browser-manager.sh     # Gestión de navegadores
│   ├── artifact-manager.sh    # Gestión de artefactos
│   ├── docker-stack-manager.sh # Gestión del stack Docker
│   ├── system-monitor.sh      # Monitoreo del sistema
│   ├── network-manager.sh     # Gestión de redes
│   ├── health-check-manager.sh # Gestión de health checks
│   ├── setup-local-dev.sh     # Configuración de desarrollo local
│   ├── setup-env.sh          # Configuración de entorno
│   ├── start-test-env.sh     # Inicio del entorno de testing
│   ├── manage-grid.sh        # Gestión de Selenium Grid
│   ├── run-e2e-tests.sh      # Ejecución de tests E2E
│   ├── ci-setup.sh           # Configuración para CI/CD
│   └── browser-manager.sh    # Gestión de navegadores
├── health_check/              # Sistema de health checks
│   ├── __init__.py           # Inicialización del módulo
│   ├── analytics/            # Análisis de métricas
│   ├── checkers/             # Verificadores de salud
│   ├── cli/                  # Interfaz de línea de comandos
│   ├── models/               # Modelos de datos
│   ├── orchestration/        # Orquestación de health checks
│   └── persistence/          # Persistencia de datos
├── Dockerfile.test-runner    # Dockerfile especializado para testing
├── Dockerfile.streamlit      # Dockerfile para aplicación Streamlit
├── docker-compose.test.yml   # Configuración de testing con Selenium Grid
├── pytest.ini               # Configuración de pytest
├── test-runner.config        # Configuración del test runner
├── grid-config.json          # Configuración de Selenium Grid
├── browser-capabilities.json # Capacidades de navegadores
├── mobile-browser-config.json # Configuración de navegadores móviles
├── health_check_system.py    # Sistema principal de health checks
├── env_utils.py              # Utilidades de entorno
├── env_config.py             # Configuración de entorno
├── env_manager.py            # Gestor de entorno
├── env.local.template        # Template de entorno local
├── env.test.template         # Template de entorno de testing
├── env.staging.template      # Template de entorno de staging
├── env.production.template   # Template de entorno de producción
├── env-test.yml              # Configuración de entorno de testing
├── docker-entrypoint.sh      # Script de entrada de Docker
├── .dockerignore             # Archivos ignorados por Docker
└── docs/                     # Documentación
    ├── README-ARCHITECTURE.md # Arquitectura técnica
    ├── README-USAGE.md       # Guía de uso
    ├── README-TROUBLESHOOTING.md # Solución de problemas
    ├── README-DOCKER-TESTING.md # Testing con Docker
    ├── README-LOCAL-DEV.md   # Desarrollo local
    ├── README.environment-management.md # Gestión de entornos
    ├── README.network-setup.md # Configuración de redes
    ├── README.cross-browser-testing.md # Testing multi-navegador
    ├── README.artifact-management.md # Gestión de artefactos
    └── selenium-grid-guide.md # Guía de Selenium Grid
```

## Propósito y Responsabilidades

### 1. **Testing Infrastructure**

- **Selenium Grid**: Entorno de testing multi-navegador
- **Test Runner**: Contenedor especializado para ejecución de tests
- **Cross-Browser Testing**: Soporte para Chrome, Firefox y Edge
- **Mobile Emulation**: Simulación de dispositivos móviles
- **Artifact Management**: Gestión de resultados y reportes

### 2. **Deployment Infrastructure**

- **Docker Containers**: Contenedores para diferentes entornos
- **Environment Management**: Gestión de configuraciones por entorno
- **Health Checks**: Verificación de salud de servicios
- **Network Management**: Configuración de redes Docker

### 3. **Monitoring & Health Checks**

- **System Monitoring**: Monitoreo de recursos y servicios
- **Health Check System**: Verificación automática de salud
- **Analytics**: Análisis de métricas y performance
- **Persistence**: Almacenamiento de datos de monitoreo

### 4. **Development Support**

- **Local Development**: Configuración para desarrollo local
- **CI/CD Integration**: Integración con pipelines de CI/CD
- **Environment Setup**: Scripts de configuración automática
- **Troubleshooting**: Herramientas de diagnóstico

## Arquitectura del Sistema

### **Multi-Network Architecture**

- **Frontend Network** (172.20.0.0/24): Servicios públicos
- **Backend Network** (172.21.0.0/24): Procesamiento interno
- **Management Network** (172.22.0.0/24): Administración y monitoreo

### **Container Specialization**

- **test-runner**: Contenedor optimizado para testing
- **streamlit-app**: Aplicación principal
- **selenium-hub**: Coordinación de Selenium Grid
- **browser-nodes**: Nodos de navegadores
- **health-monitor**: Monitoreo de salud

## Uso Rápido

### **Testing E2E**

```bash
# Ejecutar tests con configuración por defecto
./scripts/run-test-runner.sh run

# Ejecutar con navegador específico
./scripts/run-test-runner.sh run --browser firefox

# Ejecutar con cobertura y reportes
./scripts/run-test-runner.sh run --coverage --html-report
```

### **Desarrollo Local**

```bash
# Configurar entorno de desarrollo
./scripts/setup-local-dev.sh

# Iniciar servicios
./scripts/docker-stack-manager.sh start

# Ejecutar tests
./scripts/run-test-runner.sh run
```

### **Monitoreo**

```bash
# Verificar salud del sistema
./scripts/health-check-manager.sh status

# Monitorear recursos
./scripts/system-monitor.sh dashboard

# Ver logs de servicios
./scripts/docker-stack-manager.sh logs
```

## Configuración de Entornos

### **Variables de Entorno**

- `ENVIRONMENT`: Entorno de ejecución (local, test, staging, production)
- `BROWSER`: Navegador para testing (chrome, firefox, edge)
- `PARALLEL_WORKERS`: Número de workers paralelos
- `VIDEO_RECORDING`: Habilitar grabación de video
- `COVERAGE`: Habilitar cobertura de código

### **Configuraciones por Entorno**

- **Local**: Desarrollo y testing local
- **Test**: Entorno de testing automatizado
- **Staging**: Entorno de pre-producción
- **Production**: Entorno de producción

## Documentación Especializada

### **Guías Principales**

- Arquitectura Técnica: ver `infrastructure/testing/docs/README-ARCHITECTURE.md`
- Guía de Uso: ver `infrastructure/testing/docs/README-USAGE.md`
- Solución de Problemas: ver `infrastructure/testing/docs/README-TROUBLESHOOTING.md`
- Testing con Docker: ver `infrastructure/testing/docs/README-DOCKER-TESTING.md`

### **Guías Especializadas**

- Desarrollo Local: ver `infrastructure/testing/docs/README-LOCAL-DEV.md`
- Gestión de Entornos: ver `infrastructure/testing/docs/README.environment-management.md`
- Configuración de Redes: ver `infrastructure/testing/docs/README.network-setup.md`
- Testing Multi-Navegador: ver `infrastructure/testing/docs/README.cross-browser-testing.md`
- Gestión de Artefactos: ver `infrastructure/testing/docs/README.artifact-management.md`
- Guía de Selenium Grid: ver `infrastructure/testing/docs/selenium-grid-guide.md`

## Plan de Reorganización

### **Estado Actual**

Esta carpeta contiene una infraestructura Docker compleja y bien desarrollada, pero su organización
actual no sigue las mejores prácticas de proyectos ML modernos.

### **Problemas Identificados**

1. **Ubicación**: `docker/` en raíz no refleja el propósito completo
2. **Organización**: Mezcla de responsabilidades (testing, deployment, health checks)
3. **Escalabilidad**: Estructura actual dificulta agregar nuevos tipos de infraestructura
4. **Mantenibilidad**: Scripts y configuraciones dispersos

### **Solución Propuesta**

La reorganización transformará esta estructura en:

```bash
infrastructure/
├── testing/          # Infraestructura de testing
├── deployment/       # Infraestructura de deployment
├── monitoring/       # Infraestructura de monitoreo
└── shared/          # Recursos compartidos
```

### **Beneficios Esperados**

- **Organización Profesional**: Estructura clara y intuitiva
- **Escalabilidad**: Fácil agregar nuevos tipos de infraestructura
- **Mantenibilidad**: Scripts organizados por propósito
- **Integración CI/CD**: Rutas claras y consistentes

## Contribución

### **Antes de la Reorganización**

- Documentar cualquier dependencia específica
- Verificar que todos los scripts funcionen correctamente
- Actualizar referencias en CI/CD si es necesario

### **Después de la Reorganización**

- Seguir la nueva estructura modular
- Mantener separación de responsabilidades
- Documentar cambios en la nueva ubicación

---

**Nota**: Esta infraestructura Docker es fundamental para el proyecto CrackSeg y proporciona un
entorno de testing robusto y profesional. La reorganización propuesta mejorará significativamente su
mantenibilidad y escalabilidad.
