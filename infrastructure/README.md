# CrackSeg Infrastructure

> **Infraestructura modular y profesional para el proyecto CrackSeg**
>
> Esta carpeta contiene toda la infraestructura del proyecto organizada por dominio y responsabilidad.

## Estructura de Infraestructura

```bash
infrastructure/
├── testing/          # Infraestructura de testing E2E
├── deployment/       # Infraestructura de deployment
├── monitoring/       # Infraestructura de monitoreo
└── shared/          # Recursos compartidos
```

## Componentes

### **Testing (`testing/`)**

Infraestructura completa para testing end-to-end con Selenium Grid:

- **Dockerfiles**: Contenedores especializados para testing
- **Scripts**: Orquestación de tests y gestión de navegadores
- **Configuraciones**: Selenium Grid, pytest, capacidades de navegadores
- **Health Checks**: Sistema de verificación de salud para testing
- **Documentación**: Guías completas de testing

**Uso Rápido:**

```bash
cd infrastructure/testing
./scripts/run-test-runner.sh run
```

### **Deployment (`deployment/`)**

Infraestructura completa para deployment en producción:

- **Packages**: Paquetes de deployment específicos (test, test-package, test-crackseg-model)
- **Dockerfiles**: Contenedores de producción optimizados
- **Kubernetes**: Manifests para orquestación
- **Scripts**: Automatización de deployment y rollback
- **Configuraciones**: Entornos de deployment

### **Monitoring (`monitoring/`)**

Sistema de monitoreo y observabilidad:

- **Scripts**: Monitoreo de recursos y servicios
- **Dashboards**: Métricas y visualizaciones
- **Configuraciones**: Alertas y notificaciones

### **Shared (`shared/`)**

Recursos compartidos entre todos los componentes:

- **Scripts**: Utilidades comunes (setup, network, system)
- **Configuraciones**: Templates de entorno
- **Documentación**: Guías compartidas

## Migración Completada

✅ **Archivos Migrados**: 28 archivos y 5 subdirectorios
✅ **Estructura Creada**: Organización modular por dominio
✅ **Responsabilidades Separadas**: Testing, deployment, monitoring, shared
✅ **Carpeta Original Eliminada**: `/docker/` removida exitosamente

## Beneficios de la Nueva Estructura

### **Organización Profesional**

- Separación clara de responsabilidades
- Estructura intuitiva para nuevos desarrolladores
- Facilita onboarding y mantenimiento

### **Escalabilidad**

- Fácil agregar nuevos tipos de infraestructura
- Reutilización de componentes compartidos
- Configuración modular

### **Mantenibilidad**

- Scripts organizados por propósito
- Documentación estructurada
- Configuraciones centralizadas

### **Integración CI/CD**

- Rutas claras y consistentes
- Scripts especializados por entorno
- Configuraciones de entorno separadas

## Documentación Especializada

### **Testing**

- [Guía Principal](testing/docs/README.md)
- [Arquitectura](testing/docs/README-ARCHITECTURE.md)
- [Uso](testing/docs/README-USAGE.md)
- [Solución de Problemas](testing/docs/README-TROUBLESHOOTING.md)

### **Shared**

- [Desarrollo Local](shared/docs/README-LOCAL-DEV.md)
- [Gestión de Entornos](shared/docs/README.environment-management.md)
- [Configuración de Redes](shared/docs/README.network-setup.md)

## Próximos Pasos

1. **Actualizar CI/CD**: Modificar workflows de GitHub Actions
2. **Actualizar Documentación**: Referencias en README principal del proyecto
3. **Validar Funcionamiento**: Ejecutar tests completos
4. **Validar Deployments**: Probar deployments consolidados
5. **Comunicar Cambios**: Informar al equipo sobre la nueva estructura

## Comandos de Verificación

```bash
# Verificar estructura
ls infrastructure/

# Verificar testing
ls infrastructure/testing/

# Verificar shared
ls infrastructure/shared/

# Ejecutar test de verificación
cd infrastructure/testing
./scripts/run-test-runner.sh run --browser chrome
```

---

**Nota**: Esta reorganización transformó la infraestructura Docker de una estructura monolítica a
una arquitectura modular y profesional, siguiendo las mejores prácticas de proyectos ML modernos.
