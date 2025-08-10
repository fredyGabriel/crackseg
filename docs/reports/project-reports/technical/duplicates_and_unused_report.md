<!-- markdownlint-disable-file -->
# Duplicates and unused modules report — 2025-08-10 04:41:38 UTC

Heuristic scan; please review before deletion/refactor.

## Summary

Duplicate groups: 3
Unused modules under src/: 93

## Potential duplicate groups (size+hash)

- Group 1:
  - `docs/reports/analysis-reports/architecture/deps_report.txt`
  - `docs/reports/analysis-reports/architecture/dup_unused_report.txt`
  - `docs/reports/analysis-reports/architecture/language_compliance_report.txt`
  - `gui/__init__.py`
  - `gui/services/__init__.py`
  - `infrastructure/deployment/packages/test-crackseg-model/package/app/streamlit_app.py`
- Group 2:
  - `docs/designs/logo.png`
  - `gui/assets/images/logos/primary-logo.png`
- Group 3:
  - `docs/reports/experiment-reports/plots/legacy/training_curves_20250724_081112.png`
  - `docs/reports/experiment-reports/plots/legacy/training_curves_20250724_081136.png`

## Unused Python modules under src/

Path | Module
:-- | :--
`src/crackseg/evaluation/metrics/batch_processor.py` | `crackseg.evaluation.metrics.batch_processor`
`src/crackseg/evaluation/metrics/calculator.py` | `crackseg.evaluation.metrics.calculator`
`src/crackseg/evaluation/visualization/analysis/parameter.py` | `crackseg.evaluation.visualization.analysis.parameter`
`src/crackseg/evaluation/visualization/analysis/prediction.py` | `crackseg.evaluation.visualization.analysis.prediction`
`src/crackseg/evaluation/visualization/experiment/core.py` | `crackseg.evaluation.visualization.experiment.core`
`src/crackseg/evaluation/visualization/experiment/plots.py` | `crackseg.evaluation.visualization.experiment.plots`
`src/crackseg/evaluation/visualization/interactive_plotly/core.py` | `crackseg.evaluation.visualization.interactive_plotly.core`
`src/crackseg/evaluation/visualization/interactive_plotly/export_handlers.py` | `crackseg.evaluation.visualization.interactive_plotly.export_handlers`
`src/crackseg/evaluation/visualization/interactive_plotly/metadata_handlers.py` | `crackseg.evaluation.visualization.interactive_plotly.metadata_handlers`
`src/crackseg/evaluation/visualization/legacy/experiment_viz.py` | `crackseg.evaluation.visualization.legacy.experiment_viz`
`src/crackseg/evaluation/visualization/legacy/learning_rate_analysis.py` | `crackseg.evaluation.visualization.legacy.learning_rate_analysis`
`src/crackseg/evaluation/visualization/legacy/parameter_analysis.py` | `crackseg.evaluation.visualization.legacy.parameter_analysis`
`src/crackseg/evaluation/visualization/legacy/prediction_viz.py` | `crackseg.evaluation.visualization.legacy.prediction_viz`
`src/crackseg/evaluation/visualization/legacy/training_curves.py` | `crackseg.evaluation.visualization.legacy.training_curves`
`src/crackseg/evaluation/visualization/prediction/confidence.py` | `crackseg.evaluation.visualization.prediction.confidence`
`src/crackseg/evaluation/visualization/prediction/grid.py` | `crackseg.evaluation.visualization.prediction.grid`
`src/crackseg/evaluation/visualization/prediction/overlay.py` | `crackseg.evaluation.visualization.prediction.overlay`
`src/crackseg/evaluation/visualization/training/advanced.py` | `crackseg.evaluation.visualization.training.advanced`
`src/crackseg/evaluation/visualization/training/analysis.py` | `crackseg.evaluation.visualization.training.analysis`
`src/crackseg/evaluation/visualization/training/core.py` | `crackseg.evaluation.visualization.training.core`
`src/crackseg/evaluation/visualization/training/curves.py` | `crackseg.evaluation.visualization.training.curves`
`src/crackseg/evaluation/visualization/training/reports.py` | `crackseg.evaluation.visualization.training.reports`
`src/crackseg/model/common/visualization/matplotlib/components.py` | `crackseg.model.common.visualization.matplotlib.components`
`src/crackseg/model/common/visualization/matplotlib/connections.py` | `crackseg.model.common.visualization.matplotlib.connections`
`src/crackseg/model/common/visualization/matplotlib/utils.py` | `crackseg.model.common.visualization.matplotlib.utils`
`src/crackseg/reporting/comparison/analysis/anomalies.py` | `crackseg.reporting.comparison.analysis.anomalies`
`src/crackseg/reporting/comparison/analysis/ranking.py` | `crackseg.reporting.comparison.analysis.ranking`
`src/crackseg/reporting/comparison/analysis/statistical.py` | `crackseg.reporting.comparison.analysis.statistical`
`src/crackseg/reporting/comparison/analysis/trends.py` | `crackseg.reporting.comparison.analysis.trends`
`src/crackseg/reporting/comparison/core/engine.py` | `crackseg.reporting.comparison.core.engine`
`src/crackseg/reporting/comparison/utils/artifact_integration.py` | `crackseg.reporting.comparison.utils.artifact_integration`
`src/crackseg/reporting/comparison/utils/metrics.py` | `crackseg.reporting.comparison.utils.metrics`
`src/crackseg/reporting/comparison/utils/recommendations.py` | `crackseg.reporting.comparison.utils.recommendations`
`src/crackseg/reporting/comparison/utils/table_utils.py` | `crackseg.reporting.comparison.utils.table_utils`
`src/crackseg/reporting/performance/analyzer.py` | `crackseg.reporting.performance.analyzer`
`src/crackseg/reporting/performance/anomaly_detector.py` | `crackseg.reporting.performance.anomaly_detector`
`src/crackseg/reporting/performance/metric_evaluator.py` | `crackseg.reporting.performance.metric_evaluator`
`src/crackseg/reporting/performance/recommendation_engine.py` | `crackseg.reporting.performance.recommendation_engine`
`src/crackseg/reporting/performance/training_analyzer.py` | `crackseg.reporting.performance.training_analyzer`
`src/crackseg/reporting/recommendations/analyzers/hyperparameters.py` | `crackseg.reporting.recommendations.analyzers.hyperparameters`
`src/crackseg/reporting/recommendations/analyzers/performance.py` | `crackseg.reporting.recommendations.analyzers.performance`
`src/crackseg/reporting/recommendations/analyzers/training_patterns.py` | `crackseg.reporting.recommendations.analyzers.training_patterns`
`src/crackseg/reporting/recommendations/identifiers/architecture.py` | `crackseg.reporting.recommendations.identifiers.architecture`
`src/crackseg/reporting/recommendations/identifiers/opportunities.py` | `crackseg.reporting.recommendations.identifiers.opportunities`
`src/crackseg/training/components/initializer.py` | `crackseg.training.components.initializer`
`src/crackseg/training/components/setup.py` | `crackseg.training.components.setup`
`src/crackseg/training/components/training_loop.py` | `crackseg.training.components.training_loop`
`src/crackseg/training/components/validation_loop.py` | `crackseg.training.components.validation_loop`
`src/crackseg/utils/deployment/config/environment/config.py` | `crackseg.utils.deployment.config.environment.config`
`src/crackseg/utils/deployment/config/environment/core.py` | `crackseg.utils.deployment.config.environment.core`
`src/crackseg/utils/deployment/config/environment/generators.py` | `crackseg.utils.deployment.config.environment.generators`
`src/crackseg/utils/deployment/config/environment/presets.py` | `crackseg.utils.deployment.config.environment.presets`
`src/crackseg/utils/deployment/config/environment/validators.py` | `crackseg.utils.deployment.config.environment.validators`
`src/crackseg/utils/deployment/core/manager.py` | `crackseg.utils.deployment.core.manager`
`src/crackseg/utils/deployment/core/orchestrator.py` | `crackseg.utils.deployment.core.orchestrator`
`src/crackseg/utils/deployment/core/types.py` | `crackseg.utils.deployment.core.types`
`src/crackseg/utils/deployment/monitoring/config.py` | `crackseg.utils.deployment.monitoring.config`
`src/crackseg/utils/deployment/monitoring/core.py` | `crackseg.utils.deployment.monitoring.core`
`src/crackseg/utils/deployment/monitoring/health.py` | `crackseg.utils.deployment.monitoring.health`
`src/crackseg/utils/deployment/monitoring/metrics.py` | `crackseg.utils.deployment.monitoring.metrics`
`src/crackseg/utils/deployment/monitoring/performance.py` | `crackseg.utils.deployment.monitoring.performance`
`src/crackseg/utils/deployment/monitoring/resource.py` | `crackseg.utils.deployment.monitoring.resource`
`src/crackseg/utils/deployment/packaging/config.py` | `crackseg.utils.deployment.packaging.config`
`src/crackseg/utils/deployment/packaging/containerization.py` | `crackseg.utils.deployment.packaging.containerization`
`src/crackseg/utils/deployment/packaging/core.py` | `crackseg.utils.deployment.packaging.core`
`src/crackseg/utils/deployment/packaging/dependencies.py` | `crackseg.utils.deployment.packaging.dependencies`
`src/crackseg/utils/deployment/packaging/docker_compose.py` | `crackseg.utils.deployment.packaging.docker_compose`
`src/crackseg/utils/deployment/packaging/file_generators.py` | `crackseg.utils.deployment.packaging.file_generators`
`src/crackseg/utils/deployment/packaging/helm.py` | `crackseg.utils.deployment.packaging.helm`
`src/crackseg/utils/deployment/packaging/kubernetes.py` | `crackseg.utils.deployment.packaging.kubernetes`
`src/crackseg/utils/deployment/packaging/manifests.py` | `crackseg.utils.deployment.packaging.manifests`
`src/crackseg/utils/deployment/packaging/metrics.py` | `crackseg.utils.deployment.packaging.metrics`
`src/crackseg/utils/deployment/packaging/security.py` | `crackseg.utils.deployment.packaging.security`
`src/crackseg/utils/deployment/utils/multi_target.py` | `crackseg.utils.deployment.utils.multi_target`
`src/crackseg/utils/deployment/utils/production.py` | `crackseg.utils.deployment.utils.production`
`src/crackseg/utils/deployment/validation/pipeline/compatibility.py` | `crackseg.utils.deployment.validation.pipeline.compatibility`
`src/crackseg/utils/deployment/validation/pipeline/config.py` | `crackseg.utils.deployment.validation.pipeline.config`
`src/crackseg/utils/deployment/validation/pipeline/core.py` | `crackseg.utils.deployment.validation.pipeline.core`
`src/crackseg/utils/deployment/validation/pipeline/functional.py` | `crackseg.utils.deployment.validation.pipeline.functional`
`src/crackseg/utils/deployment/validation/pipeline/performance.py` | `crackseg.utils.deployment.validation.pipeline.performance`
`src/crackseg/utils/deployment/validation/pipeline/reporting.py` | `crackseg.utils.deployment.validation.pipeline.reporting`
`src/crackseg/utils/deployment/validation/pipeline/security.py` | `crackseg.utils.deployment.validation.pipeline.security`
`src/crackseg/utils/monitoring/alerts/checker.py` | `crackseg.utils.monitoring.alerts.checker`
`src/crackseg/utils/monitoring/alerts/system.py` | `crackseg.utils.monitoring.alerts.system`
`src/crackseg/utils/monitoring/alerts/types.py` | `crackseg.utils.monitoring.alerts.types`
`src/crackseg/utils/monitoring/resources/config.py` | `crackseg.utils.monitoring.resources.config`
`src/crackseg/utils/monitoring/resources/monitor.py` | `crackseg.utils.monitoring.resources.monitor`
`src/crackseg/utils/monitoring/resources/snapshot.py` | `crackseg.utils.monitoring.resources.snapshot`
`src/training_pipeline/checkpoint_manager.py` | `training_pipeline.checkpoint_manager`
`src/training_pipeline/data_loading.py` | `training_pipeline.data_loading`
`src/training_pipeline/environment_setup.py` | `training_pipeline.environment_setup`
`src/training_pipeline/model_creation.py` | `training_pipeline.model_creation`
`src/training_pipeline/training_setup.py` | `training_pipeline.training_setup`

## Recommended next steps

- For duplicates: consolidate into a single canonical location; update references
- For unused modules: verify via grep/tests; delete or integrate as needed
- Add guard checks into CI to prevent re-introduction of duplicates
