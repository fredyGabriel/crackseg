# Análisis de Impacto - Movimientos de Archivos

## Resumen Ejecutivo

⚠️ **PROBLEMAS IDENTIFICADOS**: Se encontraron múltiples archivos con rutas hardcodeadas que
necesitan actualización después de los movimientos de `docker/` y `deployments/`.

## Problemas Críticos Identificados

### **1. Rutas Hardcodeadas en Código Python**

#### **Archivos Afectados:**

**`tests/integration/utils/test_packaging_system.py`**

```python
# Línea 234: Ruta hardcodeada
package_dir = Path("deployments/test-package/package")

# Línea 314: Ruta hardcodeada
lambda: Path("deployments/test").mkdir(exist_ok=True),
```

**`scripts/packaging_example.py`**

```python
# Línea 260: Ruta hardcodeada
package_dir = Path("deployments/sample-crackseg-model/package")
```

**`scripts/deployment_example.py`**

```python
# Línea 37: Ruta hardcodeada
output_dir = Path("deployments")
```

**`src/crackseg/utils/deployment/artifact_optimizer.py`**

```python
# Línea 780: Ruta hardcodeada
output_dir = Path(f"deployments/{artifact.artifact_id}")
```

**`src/crackseg/utils/deployment/packaging/core.py`**

```python
# Línea 181: Ruta hardcodeada
package_dir = Path(f"deployments/{artifact_id}/package")
```

**`src/crackseg/utils/deployment/environment_configurator.py`**

```python
# Múltiples líneas con rutas hardcodeadas
output_dir = Path(f"deployments/{env_config.environment_name}")
```

### **2. Rutas Hardcodeadas en Scripts**

#### **Archivos Afectados:**

**`infrastructure/testing/scripts/run-test-runner.sh`**

```bash
# Línea 14: Ruta hardcodeada
DOCKER_DIR="$PROJECT_ROOT/tests/docker"
```

**`infrastructure/testing/scripts/artifact-manager.sh`**

```bash
# Línea 15: Ruta hardcodeada
DOCKER_DIR="$PROJECT_ROOT/tests/docker"
```

**`infrastructure/shared/scripts/manage-grid.sh`**

```bash
# Línea 17: Ruta hardcodeada
readonly DOCKER_DIR="${PROJECT_ROOT}/tests/docker"
```

**`infrastructure/shared/scripts/ci-setup.sh`**

```bash
# Líneas 67, 81: Rutas hardcodeadas
"$PROJECT_ROOT/tests/docker"
"$PROJECT_ROOT/tests/docker/docker-compose.test.yml"
```

### **3. GitHub Actions Workflows**

#### **Archivos Afectados:**

**`.github/workflows/e2e-testing.yml`**

```yaml
# Líneas 8-9: Paths hardcodeados
- 'tests/docker/**'
- 'tests/docker/**'

# Línea 49: Comando hardcodeado
cd tests/docker
docker-compose -f docker-compose.test.yml config > /dev/null
```

**`.github/workflows/test-e2e.yml`**

```yaml
# Línea 81: Ruta hardcodeada en cache key
CACHE_KEY="docker-${{ runner.os }}-$(sha256sum tests/docker/docker-compose.test.yml tests/docker/Dockerfile.* | sha256sum | cut -d' ' -f1)"

# Línea 97: Comando hardcodeado
if ! docker-compose -f docker-compose.test.yml config > /dev/null 2>&1; then
```

### **4. Documentación Desactualizada**

#### **Archivos Afectados:**

**`infrastructure/deployment/packages/README.md`**

```markdown
# Múltiples referencias a rutas antiguas
cd deployments/test-package/package
cd deployments/test-crackseg-model/package
```

**`docs/reports/project_tree.md`**

```markdown
# Referencias a estructura antigua
├── deployments/
```

**Múltiples archivos en `infrastructure/testing/docs/`**

```markdown
# Referencias a tests/docker/ en lugar de infrastructure/testing/
./tests/docker/scripts/manage-grid.sh
docker-compose -f tests/docker/docker-compose.test.yml
```

## Soluciones Requeridas

### **1. Actualizar Código Python (PRIORIDAD ALTA)**

#### **Archivos a Corregir:**

1. **`tests/integration/utils/test_packaging_system.py`**

   ```python
   # Cambiar de:
   package_dir = Path("deployments/test-package/package")
   # A:
   package_dir = Path("infrastructure/deployment/packages/test-package/package")
   ```

2. **`scripts/packaging_example.py`**

   ```python
   # Cambiar de:
   package_dir = Path("deployments/sample-crackseg-model/package")
   # A:
   package_dir = Path("infrastructure/deployment/packages/test-crackseg-model/package")
   ```

3. **`src/crackseg/utils/deployment/artifact_optimizer.py`**

   ```python
   # Cambiar de:
   output_dir = Path(f"deployments/{artifact.artifact_id}")
   # A:
   output_dir = Path(f"infrastructure/deployment/packages/{artifact.artifact_id}")
   ```

### **2. Actualizar Scripts (PRIORIDAD ALTA)**

#### **Archivos a Corregir:**

1. **`infrastructure/testing/scripts/run-test-runner.sh`**

   ```bash
   # Cambiar de:
   DOCKER_DIR="$PROJECT_ROOT/tests/docker"
   # A:
   DOCKER_DIR="$PROJECT_ROOT/infrastructure/testing"
   ```

2. **`infrastructure/shared/scripts/manage-grid.sh`**

   ```bash
   # Cambiar de:
   readonly DOCKER_DIR="${PROJECT_ROOT}/tests/docker"
   # A:
   readonly DOCKER_DIR="${PROJECT_ROOT}/infrastructure/testing"
   ```

### **3. Actualizar GitHub Actions (PRIORIDAD CRÍTICA)**

#### **Archivos a Corregir:**

1. **`.github/workflows/e2e-testing.yml`**

   ```yaml
   # Cambiar paths de:
   - 'tests/docker/**'
   # A:
   - 'infrastructure/testing/**'

   # Cambiar comandos de:
   cd tests/docker
   # A:
   cd infrastructure/testing
   ```

2. **`.github/workflows/test-e2e.yml`**

   ```yaml
   # Cambiar cache key de:
   CACHE_KEY="docker-${{ runner.os }}-$(sha256sum tests/docker/docker-compose.test.yml tests/docker/Dockerfile.* | sha256sum | cut -d' ' -f1)"
   # A:
   CACHE_KEY="docker-${{ runner.os }}-$(sha256sum infrastructure/testing/docker/docker-compose.test.yml infrastructure/testing/docker/Dockerfile.* | sha256sum | cut -d' ' -f1)"
   ```

### **4. Actualizar Documentación (PRIORIDAD MEDIA)**

#### **Archivos a Corregir:**

1. **`infrastructure/deployment/packages/README.md`**
2. **`docs/reports/project_tree.md`**
3. **Todos los archivos en `infrastructure/testing/docs/`**

## Plan de Corrección

### **Fase 1: Correcciones Críticas (Inmediato)**

1. **Actualizar GitHub Actions**
   - Modificar workflows para nuevas rutas
   - Probar que CI/CD funcione correctamente

2. **Actualizar Código Python**
   - Corregir rutas hardcodeadas en archivos Python
   - Ejecutar tests para verificar funcionamiento

3. **Actualizar Scripts**
   - Corregir rutas en scripts de infraestructura
   - Probar scripts de testing y deployment

### **Fase 2: Correcciones de Documentación (Próxima Semana)**

1. **Actualizar READMEs**
   - Corregir referencias en documentación
   - Actualizar ejemplos de uso

2. **Actualizar Project Tree**
   - Reflejar nueva estructura en documentación
   - Actualizar diagramas si existen

### **Fase 3: Validación Completa (Próxima Semana)**

1. **Ejecutar Tests Completos**
   - Verificar que todos los tests pasen
   - Probar deployments consolidados

2. **Validar CI/CD**
   - Confirmar que workflows funcionen
   - Verificar que builds sean exitosos

## Comandos de Verificación

```bash
# Verificar rutas hardcodeadas
grep -r "deployments/" src/ tests/ scripts/ --include="*.py"
grep -r "tests/docker" infrastructure/ --include="*.sh"

# Verificar GitHub Actions
grep -r "tests/docker" .github/ --include="*.yml"

# Ejecutar tests después de correcciones
python -m pytest tests/integration/utils/test_packaging_system.py -v
```

## Riesgos Identificados

### **Riesgos Críticos**

1. **CI/CD Roto**: Workflows de GitHub Actions fallarán
2. **Tests Fallidos**: Tests con rutas hardcodeadas fallarán
3. **Scripts Inoperativos**: Scripts de infraestructura no funcionarán

### **Riesgos Medios**

1. **Documentación Confusa**: Usuarios seguirán rutas incorrectas
2. **Deployments Fallidos**: Scripts de deployment no encontrarán archivos

### **Mitigaciones**

1. ✅ **Identificación Completa**: Todos los problemas identificados
2. ✅ **Plan de Corrección**: Pasos claros para resolver
3. ✅ **Priorización**: Correcciones críticas primero

## Conclusión

Los movimientos de archivos crearon una estructura más profesional, pero dejaron rutas hardcodeadas
que necesitan corrección inmediata. El plan de corrección debe ejecutarse prioritariamente para
evitar fallos en CI/CD y funcionalidad del proyecto.

**Estado**: ⚠️ **CORRECCIONES REQUERIDAS**
**Prioridad**: 🔴 **CRÍTICA**
**Tiempo Estimado**: 2-3 horas para correcciones críticas
