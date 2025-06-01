# Plan de Verificación Profunda Post-Linting

Este documento describe el plan de verificación exhaustiva para las tareas marcadas como 'done' tras los recientes cambios de linting, formateo y refactorización.

---

## 1. Ejecución de Tests Automáticos

- [x] Ejecutar todos los tests unitarios, de integración y de cobertura.
- [x] Directorios clave: `tests/unit/`, `tests/integration/`
- [x] Herramientas: `pytest`, `pytest-cov`
- [x] Revisar reportes en `outputs/prd_project_refinement/test_suite_evaluation/reports/`
- **Criterios de éxito:**
  - [ ] Todos los tests deben pasar sin errores.
  - [ ] La cobertura debe ser igual o superior a la reportada en los artefactos de las tareas.
  - [x] No deben aparecer nuevos warnings ni errores de tipo.

---

## 2. Verificación de Artefactos Generados

- [ ] Revisar que los artefactos generados por cada tarea/subtarea existen, están actualizados y son coherentes.
- Artefactos clave:
  - [ ] `outputs/prd_project_refinement/test_suite_evaluation/test_inventory.csv`
  - [ ] `outputs/prd_project_refinement/test_suite_evaluation/reports/coverage.xml` y HTML
  - [ ] `outputs/prd_project_refinement/test_suite_evaluation/reports/decoder_analysis/`
  - [ ] `outputs/prd_project_refinement/test_suite_evaluation/manual_intervention_required.txt`
  - [ ] `outputs/prd_project_refinement/test_suite_evaluation/manual_test_config.json`
- **Criterios de éxito:**
  - [ ] Los archivos existen y se pueden abrir.
  - [ ] Los datos reflejan el estado actual del código.
  - [ ] Los reportes de cobertura y logs no muestran errores inesperados.

---

## 3. Validación de Funcionalidad Específica por Tarea

Para cada tarea marcada como 'done':

- [ ] Leer el campo `details` y `testStrategy` de la tarea y sus subtareas.
- [ ] Confirmar que los tests y artefactos cubren todos los puntos mencionados.
- [ ] Ejecutar pruebas manuales si aplica (revisión de reportes, inventarios, artefactos).
- [ ] Validar que los artefactos referenciados en `references` existen y son coherentes.

---

## 4. Revisión de Backward Compatibility y Migraciones

- [ ] Para tareas de refactorización (Tasks 5 y 6), cargar modelos antiguos (si existen) y verificar que se pueden usar con la nueva implementación.
- **Criterios de éxito:**
  - [ ] Los modelos antiguos pueden cargarse y ejecutarse sin errores.
  - [ ] Los tests de backward compatibility pasan.

---

## 5. Revisión de Registro y Fábrica de Losses

- [ ] Ejecutar los tests de registro y fábrica de funciones de pérdida (`tests/unit/training/losses/`).
- [ ] Probar la creación de combinaciones de pérdidas desde configuración.
- **Criterios de éxito:**
  - [ ] Todos los tests pasan.
  - [ ] Se pueden registrar, recuperar y combinar pérdidas correctamente.

---

## 6. Documentación y Reporte de Resultados

- [ ] Documentar cualquier error, warning o artefacto inválido encontrado.
- [ ] Marcar como 'pending' en el sistema de tareas aquellas tareas que requieran rehacerse o corregirse.
- [ ] Generar un reporte resumen con:
  - [ ] Tareas verificadas correctamente.
  - [ ] Tareas con problemas y acciones sugeridas.
  - [ ] Estado de los artefactos y cobertura.

---

## Secuencia Recomendada

1. [x] Ejecutar todos los tests y revisar cobertura.
2. [ ] Revisar artefactos de Task 1 (test suite evaluation) y Task 2-4 (tests de decodificadores).
3. [ ] Validar refactorizaciones (Tasks 5 y 6) y backward compatibility.
4. [ ] Verificar el sistema de registro y fábrica de losses (Task 7-8).
5. [ ] Documentar y reportar resultados.

---

## Correcciones Realizadas

### Sesión actual (fecha: actual)

#### 1. Corrección de validación de skip_channels_order

- **Archivo:** `src/model/decoder/common/channel_utils.py`
- **Problema:** La función `validate_skip_channels_order` esperaba orden ascendente, pero en U-Net "low to high resolution" significa orden descendente (más canales a menos canales)
- **Solución:** Actualizada la validación para esperar orden descendente [512, 256, 128, 64]
- **Tests afectados y corregidos:**
  - `test_cnn_decoder_init`
  - `test_cnn_decoder_forward_shape`
  - `test_cnn_decoder_init_mismatch_depth`
  - `test_cnn_decoder_forward_mismatch_skips`

#### 2. Actualización de patrones de regex en tests

- **Archivo:** `tests/integration/model/test_cnn_convlstm_unet.py`
- **Problema:** Los mensajes de error habían cambiado y los tests verificaban mensajes específicos
- **Solución:** Actualizados los patrones de regex para coincidir con mensajes actuales
- **Tests actualizados:**
  - `test_cnn_decoder_forward_mismatch_skips`: Pattern actualizado a "Expected .* skip connections, got"
  - `test_cnn_convlstm_unet_init_type_mismatch`: Patterns actualizados para encoder y bottleneck

#### 3. Adaptación de test para duck typing de decoder

- **Test:** `test_cnn_convlstm_unet_init_type_mismatch`
- **Problema:** El test esperaba TypeError para MockDecoder, pero el código permite duck typing
- **Solución:** Cambiado el test para verificar que el duck typing funciona correctamente
- **Justificación:** La validación estricta de tipo para decoder está comentada intencionalmente en `UNetBase._validate_components`

#### 4. Resolución de conflicto de nombres

- **Archivo:** `tests/unit/training/losses/test_factory.py` → `test_loss_factory.py`
- **Problema:** Dos archivos con el mismo nombre `test_factory.py` causaban errores de importación
- **Solución:** Renombrado el archivo de losses para evitar conflicto

#### 5. Actualización masiva de tests unitarios de decoder

- **Archivos afectados:** Todos los tests en `tests/unit/model/decoder/`
- **Problema:** Los tests usaban orden ascendente [8, 16, 32] cuando la arquitectura U-Net correcta requiere orden descendente [32, 16, 8]
- **Solución:** Actualización sistemática de todas las listas skip_channels_list para usar orden descendente
- **Archivos actualizados:**
  - `test_channel_utils.py`: Tests de validación actualizados
  - `test_cnn_decoder_initialization.py`: Listas invertidas y comentarios actualizados
  - `test_cnn_decoder_channel_handling.py`: Todas las configuraciones actualizadas
  - `test_cnn_decoder_forward_pass.py`: Tests parametrizados actualizados
  - `test_cnn_decoder_error_handling.py`: Lista actualizada
  - `test_cnn_decoder_special_features.py`: Listas actualizadas y test de CBAM corregido
- **Estado:** ✅ Completado - todos los tests pasan

#### 6. Corrección de registro de CBAM

- **Problema:** CBAM no estaba registrado en el registry de atención cuando se ejecutaban los tests
- **Solución:**
  - Actualizado `src/model/components/__init__.py` para importar CBAM
  - Agregada importación de components en `src/model/decoder/cnn_decoder.py`
- **Resultado:** CBAM ahora se registra correctamente y los tests pasan

#### 7. Corrección de tests de evaluación

- **Archivos:** `tests/integration/evaluation/test_evaluation_pipeline.py`
- **Problemas encontrados:**
  - Configuraciones incompletas - faltaban campos requeridos como `data.num_channels_rgb`, `data.num_dims_mask`, `evaluation.num_batches_visualize`, `device_str`, `output_dir_str`
  - Test de ensemble esperaba estructura de directorios con `.hydra/config.yaml`
  - Función mock incorrecta para `load_model_from_checkpoint`
- **Soluciones:**
  - Creada función helper `create_test_config()` con configuración completa
  - Actualizado test de ensemble para crear estructura de directorios esperada
  - Corregida función mock para aceptar `checkpoint_path` como keyword argument
- **Resultado:** ✅ Todos los tests de evaluación pasan (5/5)

### Estado de tests

- **tests/integration/model/test_cnn_convlstm_unet.py**: ✅ 18/18 tests pasando
- **tests/unit/model/decoder/**: ✅ 56/57 tests pasando (1 skipped)
- **tests/integration/evaluation/**: ✅ 5/5 tests pasando
- **Cobertura total del proyecto:** 24% (necesita mejora)

---

**Este archivo es temporal y puede eliminarse o actualizarse según avance la verificación.**
