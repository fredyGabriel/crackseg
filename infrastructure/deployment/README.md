# CrackSeg Deployment Infrastructure

> **Infraestructura de deployment profesional para el proyecto CrackSeg**
>
> Esta carpeta contiene toda la infraestructura de deployment, desde paquetes de aplicación hasta
> configuraciones de infraestructura.

## Estructura de Deployment

```bash
infrastructure/deployment/
├── packages/              # Paquetes de deployment específicos
│   ├── test/             # Test deployment configurations
│   ├── test-package/     # Package testing environment
│   ├── test-crackseg-model/  # Model-specific deployment
│   └── README.md         # Documentación de paquetes
├── docker/               # Dockerfiles de deployment
├── kubernetes/           # Manifests de Kubernetes
├── scripts/              # Scripts de deployment
└── config/               # Configuraciones de deployment
```

## Componentes

### **Packages (`packages/`)**

Paquetes de deployment específicos para diferentes aplicaciones:

#### **Test Deployments (`packages/test/`)**

- **Propósito**: Entornos de desarrollo y testing para validación de deployment
- **Casos de Uso**: Testing local, validación de CI/CD, testing de scripts
- **Contenido**: Configuraciones de entorno de testing

#### **Package Template (`packages/test-package/`)**

- **Propósito**: Template estandarizado de paquete de deployment
- **Componentes**:
  - **Docker**: Multi-stage Dockerfile con imágenes optimizadas
  - **Kubernetes**: Manifests completos (deployment, service, ingress, HPA)
  - **Helm Charts**: Deployments de Kubernetes templatizados
  - **Código de Aplicación**: Templates de FastAPI y Streamlit
  - **Gestión de Configuración**: Configs específicas por entorno
  - **Scripts de Deployment**: Automatización de deployment y rollback

#### **Model-Specific Deployments (`packages/test-crackseg-model/`)**

- **Propósito**: Deployments especializados para el modelo ML CrackSeg
- **Características**:
  - Endpoints de model serving
  - Optimización de inferencia
  - Versionado de modelos
  - Monitoreo de performance
  - Health checks

### **Docker (`docker/`)**

Infraestructura Docker para deployment en producción:

- **Dockerfiles**: Contenedores de producción optimizados
- **Docker Compose**: Configuraciones multi-servicio
- **Optimizaciones**: Imágenes base optimizadas para ML

### **Kubernetes (`kubernetes/`)**

Manifests y configuraciones para orquestación:

- **Deployments**: Configuraciones de deployment
- **Services**: Configuraciones de servicio
- **Ingress**: Configuraciones de entrada
- **HPA**: Horizontal Pod Autoscaler
- **ConfigMaps**: Configuraciones de aplicación

### **Scripts (`scripts/`)**

Automatización de deployment:

- **Deployment Scripts**: Scripts de deployment automatizado
- **Rollback Scripts**: Scripts de rollback en caso de problemas
- **Health Check Scripts**: Verificación de salud de deployments
- **Monitoring Scripts**: Scripts de monitoreo de deployments

### **Config (`config/`)**

Configuraciones de deployment:

- **Environment Configs**: Configuraciones por entorno
- **Deployment Configs**: Configuraciones específicas de deployment
- **Resource Configs**: Configuraciones de recursos

## Pipeline de Deployment

### **1. Selección de Artefactos**

El sistema de deployment selecciona automáticamente los artefactos de modelo apropiados basándose en:

- Métricas de performance del modelo
- Estado de completitud del entrenamiento
- Compatibilidad de versiones
- Requisitos del entorno

### **2. Generación de Paquetes**

Cada paquete de deployment incluye:

- **Código de Aplicación**: Endpoints de model serving y API
- **Dependencias**: Requirements optimizados para producción
- **Configuración**: Settings específicos del entorno

### **3. Deployment Automatizado**

- **Docker Build**: Construcción de contenedores optimizados
- **Kubernetes Deployment**: Despliegue orquestado
- **Health Checks**: Verificación de salud del deployment
- **Monitoring**: Monitoreo continuo

## Uso Rápido

### **Deployment Local**

```bash
cd infrastructure/deployment/packages/test-package
docker-compose up -d
```

### **Deployment en Kubernetes**

```bash
cd infrastructure/deployment
kubectl apply -f kubernetes/
```

### **Deployment del Modelo CrackSeg**

```bash
cd infrastructure/deployment/packages/test-crackseg-model
./scripts/deploy.sh
```

## Beneficios de la Consolidación

### **Organización Profesional**

- ✅ Todo deployment en una ubicación coherente
- ✅ Separación clara entre paquetes e infraestructura
- ✅ Estructura intuitiva para nuevos desarrolladores

### **Escalabilidad**

- ✅ Fácil agregar nuevos paquetes de deployment
- ✅ Reutilización de infraestructura común
- ✅ Configuración modular

### **Mantenibilidad**

- ✅ Scripts organizados por propósito
- ✅ Documentación estructurada
- ✅ Configuraciones centralizadas

### **Integración CI/CD**

- ✅ Rutas claras y consistentes
- ✅ Scripts especializados por entorno
- ✅ Configuraciones de entorno separadas

## Documentación Especializada

### **Packages**

- [Guía de Paquetes](packages/README.md)
- [Test Deployments](packages/test/)
- [Package Template](packages/test-package/)
- [Model Deployment](packages/test-crackseg-model/)

### **Infraestructura**

- [Docker Configuration](docker/)
- [Kubernetes Manifests](kubernetes/)
- [Deployment Scripts](scripts/)
- [Configuration Management](config/)

## Próximos Pasos

1. **Validar Deployments**: Ejecutar deployments de prueba
2. **Actualizar CI/CD**: Modificar workflows para nuevas rutas
3. **Documentar Procesos**: Crear guías de deployment
4. **Optimizar Configuraciones**: Ajustar configs para producción

---

**Nota**: Esta consolidación unifica toda la infraestructura de deployment en una estructura
profesional y escalable, eliminando duplicaciones y mejorando la organización del proyecto.
