# Análisis del Patrón Singleton - Implementación Actual vs Estándares

## 🔍 **Problema Identificado: Patrón Singleton Defectuoso**

### **Implementación Actual (Problemática)**

```python
# En registry_setup.py líneas 30-68
_encoder_registry = None
_bottleneck_registry = None
_decoder_registry = None
_architecture_registry = None

def get_encoder_registry():
    global _encoder_registry
    if _encoder_registry is None:
        _encoder_registry = Registry(nn.Module, "Encoder")
    return _encoder_registry

# ⚠️ PROBLEMA CRÍTICO: Se ejecuta al nivel del módulo
encoder_registry = get_encoder_registry()
bottleneck_registry = get_bottleneck_registry()
decoder_registry = get_decoder_registry()
architecture_registry = get_architecture_registry()
```

### **Problemas Identificados:**

1. **Ejecución a Nivel de Módulo**: Las instancias se crean inmediatamente al importar el módulo
2. **No Thread-Safe**: No hay protección contra condiciones de carrera
3. **Imports Circulares**: El patrón no maneja imports circulares correctamente
4. **Múltiples Instancias**: Se crean nuevas instancias en cada import del módulo

## ✅ **Patrones Singleton Estándar de Python**

### **Opción 1: Singleton con `__new__` (Recomendado)**

```python
class RegistrySingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, base_class, name):
        if not hasattr(self, '_initialized'):
            self._base_class = base_class
            self._name = name
            self._components = {}
            self._tags = {}
            self._initialized = True
```

### **Opción 2: Singleton con Decorador**

```python
def singleton(cls):
    instances = {}
    lock = threading.Lock()

    def get_instance(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class Registry:
    def __init__(self, base_class, name):
        self._base_class = base_class
        self._name = name
        self._components = {}
        self._tags = {}
```

### **Opción 3: Singleton con Metaclass**

```python
class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Registry(metaclass=SingletonMeta):
    def __init__(self, base_class, name):
        self._base_class = base_class
        self._name = name
        self._components = {}
        self._tags = {}
```

## 🎯 **Recomendación: Opción 1 con `__new__`**

### **Ventajas:**

- **Thread-safe** con locks apropiados
- **Lazy initialization** - solo se crea cuando se necesita
- **Manejo de imports circulares** - no se ejecuta al nivel del módulo
- **Compatibilidad** con el código existente
- **Simplicidad** - fácil de entender y mantener

### **Implementación Propuesta:**

```python
class Registry[T]:
    _instances = {}
    _lock = threading.RLock()

    def __new__(cls, base_class: type[T], name: str) -> Registry[T]:
        # Usar tupla como clave para evitar problemas con tipos mutables
        key = (base_class, name)

        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    cls._instances[key] = super().__new__(cls)
                    # Inicialización diferida en __init__
        return cls._instances[key]

    def __init__(self, base_class: type[T], name: str) -> None:
        # Solo inicializar si no se ha hecho antes
        if not hasattr(self, '_initialized'):
            self._base_class = base_class
            self._name = name
            self._components: dict[str, type[T]] = {}
            self._tags: dict[str, list[str]] = {}
            self._lock = threading.RLock()
            self._initialized = True
```

## 📊 **Comparación de Soluciones**

| Aspecto | Implementación Actual | Singleton con `__new__` | Singleton con Decorador | Singleton con Metaclass |
|---------|----------------------|-------------------------|-------------------------|-------------------------|
| **Thread-safety** |  No | ✅ Sí | ✅ Sí | ✅ Sí |
| **Lazy initialization** |  No | ✅ Sí | ✅ Sí | ✅ Sí |
| **Imports circulares** |  Problemas | ✅ Maneja correctamente | ✅ Maneja correctamente | ✅ Maneja correctamente |
| **Compatibilidad** |  Roto | ✅ Alta | ⚠️ Media | ⚠️ Media |
| **Simplicidad** |  Confuso | ✅ Simple | ✅ Simple | ⚠️ Complejo |
| **Performance** |  Múltiples instancias | ✅ Óptimo | ✅ Óptimo | ✅ Óptimo |

## 🚨 **Problemas Específicos de la Implementación Actual**

### **1. Ejecución Inmediata**

```python
#  Se ejecuta al importar el módulo
encoder_registry = get_encoder_registry()
```

### **2. No Thread-Safe**

```python
#  Condición de carrera posible
if _encoder_registry is None:
    _encoder_registry = Registry(nn.Module, "Encoder")
```

### **3. Imports Circulares**

```python
#  registry_setup.py importa registry_support.py
#  registry_support.py importa registry_setup.py
#  Ambos ejecutan código al importar
```

### **4. Múltiples Instancias**

```python
#  Cada import crea nuevas instancias
#  No hay garantía de singleton real
```

## ✅ **Solución Propuesta**

### **Implementar Singleton con `__new__` en la clase Registry**

```python
# En registry.py
class Registry[T]:
    _instances = {}
    _lock = threading.RLock()

    def __new__(cls, base_class: type[T], name: str) -> Registry[T]:
        key = (base_class, name)
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(self, base_class: type[T], name: str) -> None:
        if not hasattr(self, '_initialized'):
            self._base_class = base_class
            self._name = name
            self._components = {}
            self._tags = {}
            self._lock = threading.RLock()
            self._initialized = True
```

### **Simplificar registry_setup.py**

```python
# En registry_setup.py
from .registry import Registry
from torch import nn

# Crear instancias directamente - el singleton se encarga de la unicidad
encoder_registry = Registry(nn.Module, "Encoder")
bottleneck_registry = Registry(nn.Module, "Bottleneck")
decoder_registry = Registry(nn.Module, "Decoder")
architecture_registry = Registry(nn.Module, "Architecture")

# Eliminar las funciones get_*_registry() - ya no son necesarias
```

## 📋 **Plan de Implementación**

1. **Modificar la clase Registry** para implementar singleton con `__new__`
2. **Simplificar registry_setup.py** eliminando las funciones get_*_registry()
3. **Eliminar auto-registración** problemática
4. **Implementar lazy loading** para componentes especializados
5. **Agregar tests** para verificar comportamiento singleton

---

**Conclusión**: La implementación actual del patrón singleton es defectuosa y causa múltiples
instancias del registro. La solución recomendada es implementar un singleton robusto con `__new__`
en la clase Registry, que proporciona thread-safety, lazy initialization y manejo correcto de
imports circulares.
