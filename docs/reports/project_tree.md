# Project Directory Structure (excluding .gitignore)

```txt
└── crackseg/
    ├── ${TASK_MASTER_PROJECT_ROOT}/
    ├── artifacts/
    ├── configs/
    │   ├── __pycache__/
    │   ├── archive/
    │   │   ├── config.yaml.backup
    │   │   └── experiment_quick_test.yaml.backup
    │   ├── data/
    │   ├── evaluation/
    │   │   └── default.yaml
    │   ├── experiments/
    │   │   ├── swinv2_hybrid/
    │   │   │   ├── swinv2_320x320_py_crackdb.yaml
    │   │   │   ├── swinv2_360x360_corrected.yaml
    │   │   │   └── swinv2_360x360_standalone.yaml
    │   │   └── README.md
    │   ├── linting/
    │   │   └── config.yaml
    │   ├── model/
    │   │   ├── architecture/
    │   │   ├── architectures/
    │   │   │   ├── cnn_convlstm_unet.yaml
    │   │   │   ├── README.md
    │   │   │   ├── swinv2_hybrid.yaml
    │   │   │   ├── unet_aspp.yaml
    │   │   │   ├── unet_cnn.yaml
    │   │   │   ├── unet_swin.yaml
    │   │   │   ├── unet_swin_base.yaml
    │   │   │   └── unet_swin_transfer.yaml
    │   │   ├── bottleneck/
    │   │   │   ├── __init__.py
    │   │   │   ├── aspp_bottleneck.yaml
    │   │   │   ├── convlstm_bottleneck.yaml
    │   │   │   ├── default_bottleneck.yaml
    │   │   │   ├── mock_bottleneck.yaml
    │   │   │   └── README.md
    │   │   ├── decoder/
    │   │   │   ├── default_decoder.yaml
    │   │   │   └── mock_decoder.yaml
    │   │   ├── encoder/
    │   │   │   ├── default_encoder.yaml
    │   │   │   ├── mock_encoder.yaml
    │   │   │   └── swin_transformer_encoder.yaml
    │   │   ├── default.yaml
    │   │   └── README.md
    │   ├── testing/
    │   │   └── performance_thresholds.yaml
    │   ├── training/
    │   │   ├── logging/
    │   │   │   ├── checkpoints.yaml
    │   │   │   └── logging_base.yaml
    │   │   ├── loss/
    │   │   │   ├── bce.yaml
    │   │   │   ├── bce_dice.yaml
    │   │   │   ├── combined.yaml
    │   │   │   ├── dice.yaml
    │   │   │   ├── focal.yaml
    │   │   │   ├── focal_dice.yaml
    │   │   │   └── smooth_l1.yaml
    │   │   ├── lr_scheduler/
    │   │   │   ├── cosine.yaml
    │   │   │   ├── reduce_on_plateau.yaml
    │   │   │   └── step_lr.yaml
    │   │   ├── metric/
    │   │   │   ├── f1.yaml
    │   │   │   ├── iou.yaml
    │   │   │   ├── precision.yaml
    │   │   │   └── recall.yaml
    │   │   ├── optimizer/
    │   │   ├── default.yaml
    │   │   ├── README.md
    │   │   └── trainer.yaml
    │   ├── __init__.py
    │   ├── base.yaml
    │   ├── basic_verification.yaml
    │   ├── CONFIG_UPDATE_REPORT.md
    │   ├── README.md
    │   └── simple_test.yaml
    ├── data/
    ├── docs/
    │   ├── analysis/
    │   ├── api/
    │   │   ├── gui_components.md
    │   │   ├── gui_pages.md
    │   │   ├── gui_services.md
    │   │   ├── utilities.md
    │   │   └── visualization_api.md
    │   ├── designs/
    │   │   ├── logo.png
    │   │   └── loss_registry_design.md
    │   ├── experiments/
    │   │   └── py_crackdb_swinv2_experiment.md
    │   ├── guides/
    │   │   ├── developer-guides/
    │   │   │   ├── architecture/
    │   │   │   │   └── legacy/
    │   │   │   │       ├── architectural_decisions.md
    │   │   │   │       ├── README.md
    │   │   │   │       └── TECHNICAL_ARCHITECTURE.md
    │   │   │   ├── development/
    │   │   │   │   └── legacy/
    │   │   │   │       ├── CONTRIBUTING.md
    │   │   │   │       ├── gui_development_guidelines.md
    │   │   │   │       ├── README.md
    │   │   │   │       └── SYSTEM_DEPENDENCIES.md
    │   │   │   ├── quality/
    │   │   │   │   └── legacy/
    │   │   │   │       ├── comprehensive_integration_test_reporting_guide.md
    │   │   │   │       ├── gui_testing_best_practices.md
    │   │   │   │       ├── gui_testing_implementation_checklist.md
    │   │   │   │       ├── quality_gates_guide.md
    │   │   │   │       ├── README.md
    │   │   │   │       └── test_maintenance_procedures.md
    │   │   │   └── README.md
    │   │   ├── operational-guides/
    │   │   │   ├── cicd/
    │   │   │   │   └── legacy/
    │   │   │   │       ├── ci_cd_integration_guide.md
    │   │   │   │       ├── ci_cd_testing_integration.md
    │   │   │   │       └── README.md
    │   │   │   ├── deployment/
    │   │   │   │   ├── legacy/
    │   │   │   │   │   ├── DEPLOYMENT_PIPELINE_ARCHITECTURE.md
    │   │   │   │   │   ├── deployment_system_configuration_guide.md
    │   │   │   │   │   ├── deployment_system_troubleshooting_guide.md
    │   │   │   │   │   └── deployment_system_user_guide.md
    │   │   │   │   └── deployment_guide.md
    │   │   │   ├── monitoring/
    │   │   │   │   ├── legacy/
    │   │   │   │   │   ├── continuous_coverage_monitoring_guide.md
    │   │   │   │   │   └── README.md
    │   │   │   │   └── monitoring_guide.md
    │   │   │   ├── workflows/
    │   │   │   │   ├── legacy/
    │   │   │   │   │   ├── CLEAN_INSTALLATION.md
    │   │   │   │   │   ├── README.md
    │   │   │   │   │   └── WORKFLOW_TRAINING.md
    │   │   │   │   └── training_workflow_guide.md
    │   │   │   ├── README.md
    │   │   │   └── successful_experiments_guide.md
    │   │   ├── reporting-visualization/
    │   │   │   ├── reporting/
    │   │   │   │   └── legacy/
    │   │   │   │       ├── experiment_reporter_architecture.md
    │   │   │   │       └── experiment_reporter_usage.md
    │   │   │   ├── visualization/
    │   │   │   │   └── legacy/
    │   │   │   │       ├── visualization_customization_guide.md
    │   │   │   │       └── visualization_usage_examples.md
    │   │   │   └── README.md
    │   │   ├── technical-specs/
    │   │   │   ├── experiments/
    │   │   │   │   └── legacy/
    │   │   │   │       └── README_swinv2_hybrid.md
    │   │   │   ├── specifications/
    │   │   │   │   └── legacy/
    │   │   │   │       ├── checkpoint_format_specification.md
    │   │   │   │       ├── configuration_storage_specification.md
    │   │   │   │       ├── performance_benchmarking_system.md
    │   │   │   │       ├── README.md
    │   │   │   │       └── traceability_data_model_specification.md
    │   │   │   └── README.md
    │   │   ├── user-guides/
    │   │   │   ├── getting-started/
    │   │   │   ├── troubleshooting/
    │   │   │   │   └── legacy/
    │   │   │   │       ├── README.md
    │   │   │   │       └── TROUBLESHOOTING.md
    │   │   │   ├── usage/
    │   │   │   │   ├── legacy/
    │   │   │   │   │   ├── experiment_tracker/
    │   │   │   │   │   ├── focal_dice_loss_usage.md
    │   │   │   │   │   ├── loss_registry_usage.md
    │   │   │   │   │   ├── README.md
    │   │   │   │   │   └── USAGE.md
    │   │   │   │   ├── deployment_orchestration_api.md
    │   │   │   │   ├── experiment_tracker_usage.md
    │   │   │   │   ├── health_monitoring_guide.md
    │   │   │   │   ├── multi_target_deployment_guide.md
    │   │   │   │   └── prediction_analysis_guide.md
    │   │   │   └── README.md
    │   │   ├── experiment_data_saving_guide.md
    │   │   ├── generalized_experiment_organization.md
    │   │   ├── README.md
    │   │   ├── REORGANIZATION_COMPLETION_REPORT.md
    │   │   └── trainer_refactoring.md
    │   ├── plans/
    │   │   ├── articulo_cientifico_swinv2_cnn_aspp_unet.md
    │   │   ├── artifact_system_development_plan.md
    │   │   └── refactoring_plan_large_files.md
    │   ├── reports/
    │   │   ├── analysis-reports/
    │   │   │   ├── architecture/
    │   │   │   ├── code-quality/
    │   │   │   │   └── legacy/
    │   │   │   │       ├── basedpyright_analysis_report.md
    │   │   │   │       ├── consolidation-implementation-summary.md
    │   │   │   │       ├── duplication-mapping.md
    │   │   │   │       ├── final-rule-cleanup-summary.md
    │   │   │   │       ├── pytorch_cuda_compatibility_issue.md
    │   │   │   │       ├── rule-consolidation-report.md
    │   │   │   │       ├── rule-system-analysis.md
    │   │   │   │       └── tensorboard_component_refactoring_summary.md
    │   │   │   ├── performance/
    │   │   │   └── README.md
    │   │   ├── experiment-reports/
    │   │   │   ├── comparisons/
    │   │   │   ├── plots/
    │   │   │   │   └── legacy/
    │   │   │   │       ├── tutorial_02_plots/
    │   │   │   │       ├── experiment_comparison_20250724_081112.csv
    │   │   │   │       ├── experiment_comparison_20250724_081136.csv
    │   │   │   │       ├── performance_radar_20250724_081112.png
    │   │   │   │       ├── performance_radar_20250724_081136.png
    │   │   │   │       ├── swinv2_hybrid_summary_20250724_081510.csv
    │   │   │   │       ├── swinv2_hybrid_training_curves_20250724_081510.png
    │   │   │   │       ├── training_curves_20250724_081112.png
    │   │   │   │       └── training_curves_20250724_081136.png
    │   │   │   ├── results/
    │   │   │   └── README.md
    │   │   ├── model-reports/
    │   │   │   ├── analysis/
    │   │   │   │   └── legacy/
    │   │   │   │       ├── model_expected_structure.json
    │   │   │   │       ├── model_imports_catalog.json
    │   │   │   │       ├── model_inventory.json
    │   │   │   │       ├── model_pyfiles.json
    │   │   │   │       └── model_structure_diff.json
    │   │   │   ├── architecture/
    │   │   │   ├── performance/
    │   │   │   └── README.md
    │   │   ├── project-reports/
    │   │   │   ├── academic/
    │   │   │   ├── documentation/
    │   │   │   │   ├── legacy/
    │   │   │   │   │   ├── tasks/
    │   │   │   │   │   ├── crackseg_paper.md
    │   │   │   │   │   ├── crackseg_paper_es.md
    │   │   │   │   │   ├── documentation_checklist.md
    │   │   │   │   │   ├── plan_verificacion_post_linting.md
    │   │   │   │   │   └── technical_report.md
    │   │   │   │   ├── documentation_catalog.json
    │   │   │   │   ├── documentation_catalog_summary.md
    │   │   │   │   └── README.md
    │   │   │   ├── technical/
    │   │   │   │   └── deployment_system_documentation_summary.md
    │   │   │   └── README.md
    │   │   ├── templates/
    │   │   │   ├── examples/
    │   │   │   ├── scripts/
    │   │   │   │   └── legacy/
    │   │   │   │       ├── example_prd.txt
    │   │   │   │       ├── hydra_examples.txt
    │   │   │   │       └── README.md
    │   │   │   └── README.md
    │   │   ├── testing-reports/
    │   │   │   ├── analysis/
    │   │   │   │   └── legacy/
    │   │   │   │       ├── automated_test_execution_report.md
    │   │   │   │       ├── gui_corrections_inventory.md
    │   │   │   │       ├── gui_test_coverage_analysis.md
    │   │   │   │       ├── next_testing_priorities.md
    │   │   │   │       ├── test_coverage_improvement_plan.md
    │   │   │   │       ├── test_fixes_validation_report.md
    │   │   │   │       └── test_inventory.txt
    │   │   │   ├── coverage/
    │   │   │   │   └── legacy/
    │   │   │   │       ├── coverage_gaps_analysis.md
    │   │   │   │       ├── coverage_validation_report.md
    │   │   │   │       ├── test_coverage_analysis_report.md
    │   │   │   │       └── test_coverage_comparison_report.md
    │   │   │   ├── execution/
    │   │   │   └── README.md
    │   │   ├── file_inventory.md
    │   │   ├── project_tree.md
    │   │   ├── README.md
    │   │   └── REORGANIZATION_COMPLETION_REPORT.md
    │   ├── stylesheets/
    │   │   └── extra.css
    │   ├── testing/
    │   │   ├── artifact_testing_plan.md
    │   │   ├── test_patterns_and_best_practices.md
    │   │   └── visualization_testing_guide.md
    │   ├── tools/
    │   │   └── task-master-guide.md
    │   ├── tutorials/
    │   │   ├── cli/
    │   │   │   ├── 01_basic_training_cli.md
    │   │   │   ├── 02_custom_experiment_cli.md
    │   │   │   ├── 03_extending_project_cli.md
    │   │   │   ├── CURRENT_EXPERIMENT_EXECUTION_GUIDE.md
    │   │   │   └── TUTORIALS_UPDATE_REPORT.md
    │   │   ├── gui/
    │   │   │   ├── 01_basic_training.md
    │   │   │   ├── 02_custom_experiment.md
    │   │   │   └── 03_extending_project.md
    │   │   └── README.md
    │   └── index.md
    ├── gui/
    │   ├── __pycache__/
    │   ├── assets/
    │   │   ├── __pycache__/
    │   │   ├── css/
    │   │   │   ├── components/
    │   │   │   │   ├── navigation.css
    │   │   │   │   └── README.md
    │   │   │   ├── global/
    │   │   │   │   ├── base.css
    │   │   │   │   └── README.md
    │   │   │   └── themes/
    │   │   │       └── README.md
    │   │   ├── fonts/
    │   │   │   └── primary/
    │   │   │       └── README.md
    │   │   ├── images/
    │   │   │   ├── backgrounds/
    │   │   │   │   └── README.md
    │   │   │   ├── icons/
    │   │   │   │   └── README.md
    │   │   │   ├── logos/
    │   │   │   │   ├── primary-logo.png
    │   │   │   │   └── README.md
    │   │   │   └── samples/
    │   │   │       └── README.md
    │   │   ├── js/
    │   │   │   └── components/
    │   │   │       └── README.md
    │   │   ├── manifest/
    │   │   │   ├── asset_registry.json
    │   │   │   └── optimization_config.json
    │   │   ├── init_assets.py
    │   │   ├── manager.py
    │   │   ├── README.md
    │   │   └── structure.md
    │   ├── components/
    │   │   ├── __pycache__/
    │   │   ├── core/
    │   │   │   ├── __pycache__/
    │   │   │   ├── loading/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── optimized.py
    │   │   │   │   └── standard.py
    │   │   │   ├── navigation/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── router.py
    │   │   │   │   └── sidebar.py
    │   │   │   ├── progress/
    │   │   │   │   ├── optimized.py
    │   │   │   │   └── standard.py
    │   │   │   └── __init__.py
    │   │   ├── data/
    │   │   ├── deprecated/
    │   │   │   └── file_browser_obsolete.py
    │   │   ├── ml/
    │   │   │   ├── config/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── editor.py
    │   │   │   │   ├── editor_core.py
    │   │   │   │   ├── file_browser_integration.py
    │   │   │   │   └── validation_panel.py
    │   │   │   ├── device/
    │   │   │   │   ├── detector.py
    │   │   │   │   ├── info.py
    │   │   │   │   ├── selector.py
    │   │   │   │   └── ui.py
    │   │   │   ├── tensorboard/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── recovery/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   └── recovery_strategies.py
    │   │   │   │   ├── rendering/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── diagnostics/
    │   │   │   │   │   ├── status_cards/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── advanced_status_renderer.py
    │   │   │   │   │   ├── control_renderer.py
    │   │   │   │   │   ├── error_renderer.py
    │   │   │   │   │   ├── iframe_renderer.py
    │   │   │   │   │   ├── startup_renderer.py
    │   │   │   │   │   └── status_renderer.py
    │   │   │   │   ├── state/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── progress_tracker.py
    │   │   │   │   │   └── session_manager.py
    │   │   │   │   ├── utils/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── formatters.py
    │   │   │   │   │   └── validators.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── legacy.py
    │   │   │   │   └── main.py
    │   │   │   └── __init__.py
    │   │   ├── ui/
    │   │   │   ├── dialogs/
    │   │   │   │   ├── confirmation.py
    │   │   │   │   ├── renderer.py
    │   │   │   │   └── utils.py
    │   │   │   ├── error/
    │   │   │   │   ├── auto_save.py
    │   │   │   │   ├── console.py
    │   │   │   │   └── log_viewer.py
    │   │   │   ├── theme/
    │   │   │   │   ├── header.py
    │   │   │   │   ├── logo.py
    │   │   │   │   └── main.py
    │   │   │   └── __init__.py
    │   │   ├── __init__.py
    │   │   └── README.md
    │   ├── docs/
    │   │   ├── error_messaging_system.md
    │   │   ├── file_upload_guide.md
    │   │   └── tensorboard_integration_summary.md
    │   ├── pages/
    │   │   ├── __pycache__/
    │   │   ├── core/
    │   │   │   ├── home/
    │   │   │   │   └── main.py
    │   │   │   ├── navigation/
    │   │   │   └── __init__.py
    │   │   ├── data/
    │   │   ├── deprecated/
    │   │   ├── ml/
    │   │   │   ├── architecture/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── config_section.py
    │   │   │   │   ├── info_section.py
    │   │   │   │   ├── main.py
    │   │   │   │   ├── model_section.py
    │   │   │   │   ├── utils.py
    │   │   │   │   └── visualization_section.py
    │   │   │   ├── config/
    │   │   │   │   ├── advanced.py
    │   │   │   │   └── basic.py
    │   │   │   ├── training/
    │   │   │   │   ├── legacy.py
    │   │   │   │   └── main.py
    │   │   │   └── __init__.py
    │   │   ├── __init__.py
    │   │   └── README.md
    │   ├── services/
    │   │   ├── __pycache__/
    │   │   ├── __init__.py
    │   │   ├── gallery_export_service.py
    │   │   ├── gallery_scanner_service.py
    │   │   └── gpu_monitor.py
    │   ├── styles/
    │   │   └── main.css
    │   ├── utils/
    │   │   ├── __pycache__/
    │   │   ├── config/
    │   │   │   ├── __pycache__/
    │   │   │   ├── error_reporting/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── core_reporter.py
    │   │   │   │   ├── formatters.py
    │   │   │   │   ├── report_models.py
    │   │   │   │   └── utils.py
    │   │   │   ├── io/
    │   │   │   ├── parsing/
    │   │   │   ├── schema/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── constraint_validator.py
    │   │   │   │   ├── core_validator.py
    │   │   │   │   ├── type_validator.py
    │   │   │   │   └── utils.py
    │   │   │   ├── validation/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── error_categorizer.py
    │   │   │   │   └── yaml_engine.py
    │   │   │   ├── __init__.py
    │   │   │   ├── cache.py
    │   │   │   ├── config_loader.py
    │   │   │   ├── error_reporter.py
    │   │   │   ├── exceptions.py
    │   │   │   ├── formatters.py
    │   │   │   ├── io.py
    │   │   │   ├── parsing_engine.py
    │   │   │   ├── schema_validator.py
    │   │   │   └── templates.py
    │   │   ├── core/
    │   │   │   ├── __pycache__/
    │   │   │   ├── config/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── gui.py
    │   │   │   │   └── io.py
    │   │   │   ├── session/
    │   │   │   │   ├── auto_save.py
    │   │   │   │   ├── state.py
    │   │   │   │   └── sync.py
    │   │   │   ├── validation/
    │   │   │   │   └── error_state.py
    │   │   │   └── __init__.py
    │   │   ├── data/
    │   │   ├── deprecated/
    │   │   │   ├── demo_tensorboard_obsolete.py
    │   │   │   ├── override_examples_obsolete.py
    │   │   │   └── streaming_examples_obsolete.py
    │   │   ├── ml/
    │   │   │   ├── architecture/
    │   │   │   │   └── viewer.py
    │   │   │   ├── tensorboard/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── core.py
    │   │   │   │   ├── demo_refactored.py
    │   │   │   │   ├── lifecycle_manager.py
    │   │   │   │   ├── manager.py
    │   │   │   │   ├── port_management.py
    │   │   │   │   └── process_manager.py
    │   │   │   ├── training/
    │   │   │   │   └── state.py
    │   │   │   └── __init__.py
    │   │   ├── process/
    │   │   │   ├── manager/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── cleanup/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── abort_system.py
    │   │   │   │   │   └── process_cleanup.py
    │   │   │   │   ├── core/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── core.py
    │   │   │   │   │   ├── error_handling.py
    │   │   │   │   │   ├── manager_backup_original.py
    │   │   │   │   │   ├── process_manager.py
    │   │   │   │   │   └── states.py
    │   │   │   │   ├── logging/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── log_integration.py
    │   │   │   │   │   └── log_streamer.py
    │   │   │   │   ├── monitoring/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── monitoring.py
    │   │   │   │   │   └── process_monitor.py
    │   │   │   │   ├── overrides/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── override_handler.py
    │   │   │   │   │   └── override_parser.py
    │   │   │   │   ├── streaming/
    │   │   │   │   ├── threading/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── abort_api.py
    │   │   │   │   ├── main.py
    │   │   │   │   ├── manager_backup.py
    │   │   │   │   ├── orchestrator.py
    │   │   │   │   ├── session_api.py
    │   │   │   │   ├── status_integration.py
    │   │   │   │   ├── status_updates.py
    │   │   │   │   ├── streaming_api.py
    │   │   │   │   ├── ui_integration.py
    │   │   │   │   └── ui_status_helpers.py
    │   │   │   ├── streaming/
    │   │   │   │   ├── sources/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── file_watcher.py
    │   │   │   │   │   └── stdout_reader.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── core.py
    │   │   │   │   └── exceptions.py
    │   │   │   ├── threading/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── cancellation.py
    │   │   │   │   ├── coordinator.py
    │   │   │   │   ├── progress_tracking.py
    │   │   │   │   ├── task_results.py
    │   │   │   │   ├── task_status.py
    │   │   │   │   ├── ui_responsive_backup.py
    │   │   │   │   └── ui_wrapper.py
    │   │   │   └── __init__.py
    │   │   ├── ui/
    │   │   │   ├── dialogs/
    │   │   │   │   └── save.py
    │   │   │   ├── styling/
    │   │   │   │   └── css.py
    │   │   │   ├── theme/
    │   │   │   │   ├── manager.py
    │   │   │   │   └── optimizer.py
    │   │   │   └── __init__.py
    │   │   ├── __init__.py
    │   │   └── README.md
    │   ├── utils(results/
    │   ├── __init__.py
    │   ├── app.py
    │   ├── debug_page_rendering.py
    │   └── README.md
    ├── infrastructure/
    │   ├── deployment/
    │   │   ├── config/
    │   │   ├── docker/
    │   │   ├── kubernetes/
    │   │   ├── packages/
    │   │   │   ├── test/
    │   │   │   ├── test-crackseg-model/
    │   │   │   │   └── package/
    │   │   │   │       ├── app/
    │   │   │   │       ├── config/
    │   │   │   │       ├── docs/
    │   │   │   │       ├── scripts/
    │   │   │   │       └── tests/
    │   │   │   ├── test-package/
    │   │   │   │   └── package/
    │   │   │   │       ├── app/
    │   │   │   │       ├── config/
    │   │   │   │       ├── docs/
    │   │   │   │       ├── helm/
    │   │   │   │       ├── k8s/
    │   │   │   │       ├── scripts/
    │   │   │   │       ├── tests/
    │   │   │   │       ├── docker-compose.yml
    │   │   │   │       ├── Dockerfile
    │   │   │   │       └── requirements.txt
    │   │   │   └── README.md
    │   │   ├── scripts/
    │   │   └── README.md
    │   ├── monitoring/
    │   │   ├── config/
    │   │   ├── dashboards/
    │   │   └── scripts/
    │   ├── shared/
    │   │   ├── config/
    │   │   │   ├── env-test.yml
    │   │   │   ├── env.local.template
    │   │   │   ├── env.production.template
    │   │   │   ├── env.staging.template
    │   │   │   ├── env.test.template
    │   │   │   ├── env_config.py
    │   │   │   ├── env_manager.py
    │   │   │   └── env_utils.py
    │   │   ├── docs/
    │   │   │   ├── README-LOCAL-DEV.md
    │   │   │   ├── README.environment-management.md
    │   │   │   └── README.network-setup.md
    │   │   └── scripts/
    │   │       ├── ci-setup.sh
    │   │       ├── docker-stack-manager.sh
    │   │       ├── health-check-manager.sh
    │   │       ├── manage-grid.sh
    │   │       ├── network-manager.sh
    │   │       ├── run-e2e-tests.sh
    │   │       ├── setup-env.sh
    │   │       ├── setup-local-dev.sh
    │   │       ├── start-test-env.sh
    │   │       └── system-monitor.sh
    │   ├── testing/
    │   │   ├── config/
    │   │   │   ├── browser-capabilities.json
    │   │   │   ├── grid-config.json
    │   │   │   ├── mobile-browser-config.json
    │   │   │   ├── pytest.ini
    │   │   │   └── test-runner.config
    │   │   ├── docker/
    │   │   │   ├── docker-compose.test.yml
    │   │   │   ├── docker-entrypoint.sh
    │   │   │   ├── Dockerfile.streamlit
    │   │   │   └── Dockerfile.test-runner
    │   │   ├── docs/
    │   │   │   ├── docker-compose.README.md
    │   │   │   ├── README-ARCHITECTURE.md
    │   │   │   ├── README-DOCKER-TESTING.md
    │   │   │   ├── README-TROUBLESHOOTING.md
    │   │   │   ├── README-USAGE.md
    │   │   │   ├── README.artifact-management.md
    │   │   │   ├── README.cross-browser-testing.md
    │   │   │   ├── README.md
    │   │   │   ├── REORGANIZATION_PLAN.md
    │   │   │   └── selenium-grid-guide.md
    │   │   ├── health_check/
    │   │   │   ├── health_check/
    │   │   │   │   ├── analytics/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── dashboard_generator.py
    │   │   │   │   │   ├── metrics_collector.py
    │   │   │   │   │   └── recommendation_engine.py
    │   │   │   │   ├── checkers/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── dependency_validator.py
    │   │   │   │   │   ├── docker_checker.py
    │   │   │   │   │   └── endpoint_checker.py
    │   │   │   │   ├── cli/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   └── commands.py
    │   │   │   │   ├── models/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── config.py
    │   │   │   │   │   ├── enums.py
    │   │   │   │   │   └── results.py
    │   │   │   │   ├── orchestration/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── health_orchestrator.py
    │   │   │   │   │   ├── monitoring.py
    │   │   │   │   │   └── service_registry.py
    │   │   │   │   ├── persistence/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   └── report_saver.py
    │   │   │   │   └── __init__.py
    │   │   │   └── health_check_system.py
    │   │   └── scripts/
    │   │       ├── artifact-manager.sh
    │   │       ├── browser-manager.sh
    │   │       ├── e2e-test-orchestrator.sh
    │   │       └── run-test-runner.sh
    │   ├── INFRASTRUCTURE_REORGANIZATION_SUMMARY.md
    │   └── README.md
    ├── scripts/
    │   ├── __pycache__/
    │   ├── archive/
    │   │   ├── limpieza_scripts_summary.md
    │   │   └── README.md
    │   ├── data_processing/
    │   │   ├── format_conversion/
    │   │   │   ├── convert_crackseg_dataset.py
    │   │   │   ├── README_segmentation_to_detection.md
    │   │   │   └── segmentation_to_detection.py
    │   │   ├── image_processing/
    │   │   │   ├── __pycache__/
    │   │   │   ├── crop_crack_images.py
    │   │   │   ├── crop_crack_images_configurable.py
    │   │   │   ├── crop_py_crackdb_images.py
    │   │   │   ├── process_cfd_dataset.py
    │   │   │   ├── process_py_crackdb_example.py
    │   │   │   ├── README_crop_crack_images.md
    │   │   │   ├── README_crop_crack_images_configurable.md
    │   │   │   ├── README_py_crackdb_cropping.md
    │   │   │   └── test_py_crackdb_cropping.py
    │   │   ├── mask_verification/
    │   │   │   ├── demo_verification.py
    │   │   │   ├── example_verification.py
    │   │   │   ├── README_verification.md
    │   │   │   ├── run_verification.py
    │   │   │   ├── segmentation_mask_verifier.py
    │   │   │   └── VERIFICATION_SYSTEM_SUMMARY.md
    │   │   ├── CORRECTION_SUMMARY.md
    │   │   └── README.md
    │   ├── debug/
    │   │   ├── __init__.py
    │   │   ├── artifact_diagnostics.py
    │   │   ├── artifact_fixer.py
    │   │   ├── checkpoint_validator.py
    │   │   ├── main.py
    │   │   ├── mass_git_restore.py
    │   │   ├── syntax_scanner.py
    │   │   └── utils.py
    │   ├── deployment/
    │   │   ├── examples/
    │   │   │   ├── artifact_selection_example.py
    │   │   │   ├── deployment_example.py
    │   │   │   ├── orchestration_example.py
    │   │   │   └── packaging_example.py
    │   │   ├── __init__.py
    │   │   └── README.md
    │   ├── docs/
    │   │   └── reports/
    │   ├── examples/
    │   │   ├── advanced_orchestration_demo.py
    │   │   ├── advanced_prediction_viz_demo.py
    │   │   ├── advanced_training_viz_demo.py
    │   │   ├── deployment_orchestration_example.py
    │   │   ├── experiment_saver_example.py
    │   │   ├── factory_registry_integration.py
    │   │   ├── health_monitoring_demo.py
    │   │   ├── interactive_plotly_demo.py
    │   │   ├── multi_target_deployment_demo.py
    │   │   ├── prediction_analysis_demo.py
    │   │   ├── production_readiness_validation_example.py
    │   │   ├── template_system_demo.py
    │   │   ├── tensorboard_port_management_demo.py
    │   │   ├── validation_pipeline_demo.py
    │   │   └── validation_reporting_demo.py
    │   ├── experiments/
    │   │   ├── analysis/
    │   │   │   └── swinv2_hybrid/
    │   │   │       └── analysis/
    │   │   │           ├── __init__.py
    │   │   │           └── analyze_experiment.py
    │   │   ├── benchmarking/
    │   │   │   ├── automated_comparison.py
    │   │   │   └── benchmark_aspp.py
    │   │   ├── debugging/
    │   │   │   └── debug_swin_params.py
    │   │   ├── demos/
    │   │   │   ├── example_generalized_experiment.py
    │   │   │   ├── hybrid_registry_demo.py
    │   │   │   └── registry_demo.py
    │   │   ├── e2e/
    │   │   │   ├── modules/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── checkpointing.py
    │   │   │   │   ├── config.py
    │   │   │   │   ├── data.py
    │   │   │   │   ├── dataclasses.py
    │   │   │   │   ├── evaluation.py
    │   │   │   │   ├── setup.py
    │   │   │   │   ├── training.py
    │   │   │   │   └── utils.py
    │   │   │   ├── __init__.py
    │   │   │   ├── README.md
    │   │   │   └── test_pipeline_e2e.py
    │   │   ├── tutorials/
    │   │   │   └── tutorial_02/
    │   │   │       ├── tutorial_02_batch.ps1
    │   │   │       ├── tutorial_02_compare.py
    │   │   │       └── tutorial_02_visualize.py
    │   │   ├── README.md
    │   │   └── REORGANIZATION_SUMMARY.md
    │   ├── maintenance/
    │   │   ├── debugging/
    │   │   ├── performance/
    │   │   ├── __init__.py
    │   │   └── cleanup_hydra_folders.py
    │   ├── monitoring/
    │   │   └── continuous_coverage.py
    │   ├── performance/
    │   │   ├── __pycache__/
    │   │   ├── __init__.py
    │   │   ├── base_executor.py
    │   │   ├── baseline_updater.py
    │   │   ├── cleanup_validator.py
    │   │   ├── health_checker.py
    │   │   ├── maintenance_manager.py
    │   │   └── utils.py
    │   ├── prediction/
    │   │   ├── __init__.py
    │   │   ├── predict_image.py
    │   │   └── README.md
    │   ├── reports/
    │   │   ├── autofix_backups/
    │   │   ├── compare_model_structure.py
    │   │   ├── model_imports_autofix.py
    │   │   ├── model_imports_catalog.py
    │   │   ├── model_imports_cycles.py
    │   │   └── model_imports_validation.py
    │   ├── utils/
    │   │   ├── analysis/
    │   │   │   ├── __init__.py
    │   │   │   ├── inventory_training_imports.py
    │   │   │   └── README.md
    │   │   ├── documentation/
    │   │   │   ├── __init__.py
    │   │   │   ├── catalog_documentation.py
    │   │   │   ├── generate_project_tree.py
    │   │   │   ├── organize_reports.py
    │   │   │   └── README.md
    │   │   ├── maintenance/
    │   │   │   ├── __init__.py
    │   │   │   ├── audit_rules_checklist.py
    │   │   │   ├── check_updates.py
    │   │   │   ├── clean_workspace.py
    │   │   │   ├── README.md
    │   │   │   ├── validate-rule-references.py
    │   │   │   └── verify_setup.py
    │   │   ├── model_tools/
    │   │   │   ├── __init__.py
    │   │   │   ├── example_override.py
    │   │   │   ├── model_summary.py
    │   │   │   ├── README.md
    │   │   │   └── unet_diagram.py
    │   │   ├── test_suite_refinement/
    │   │   │   ├── add_reproducibility_score.py
    │   │   │   ├── categorize_tests_status.py
    │   │   │   ├── generate_executive_report.py
    │   │   │   ├── generate_test_inventory.py
    │   │   │   ├── report_environment_issues.py
    │   │   │   ├── report_manual_intervention.py
    │   │   │   ├── report_slow_tests.py
    │   │   │   ├── run_coverage_report.py
    │   │   │   ├── tag_test_priority.py
    │   │   │   └── update_test_inventory_status.py
    │   │   ├── generate_missing_plots.py
    │   │   ├── generate_py_crackdb_plots.py
    │   │   ├── generate_sensitivity_specificity_plot.py
    │   │   ├── generate_user_manual.py
    │   │   ├── README.md
    │   │   └── REORGANIZATION_SUMMARY.md
    │   ├── __init__.py
    │   ├── README.md
    │   └── SCRIPTS_UPDATE_REPORT.md
    ├── src/
    │   ├── __pycache__/
    │   ├── crackseg/
    │   │   ├── __pycache__/
    │   │   ├── artifacts/
    │   │   ├── data/
    │   │   ├── evaluation/
    │   │   │   ├── __pycache__/
    │   │   │   ├── cli/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── components.py
    │   │   │   │   ├── config.py
    │   │   │   │   ├── environment.py
    │   │   │   │   ├── prediction_cli.py
    │   │   │   │   └── runner.py
    │   │   │   ├── core/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── analyzer.py
    │   │   │   │   ├── image_processor.py
    │   │   │   │   └── model_loader.py
    │   │   │   ├── ensemble/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── ensemble.py
    │   │   │   ├── metrics/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── batch_processor.py
    │   │   │   │   └── calculator.py
    │   │   │   ├── utils/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── core.py
    │   │   │   │   ├── data.py
    │   │   │   │   ├── loading.py
    │   │   │   │   ├── results.py
    │   │   │   │   └── setup.py
    │   │   │   ├── visualization/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── analysis/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── parameter.py
    │   │   │   │   │   └── prediction.py
    │   │   │   │   ├── experiment/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── core.py
    │   │   │   │   │   └── plots.py
    │   │   │   │   ├── interactive_plotly/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── core.py
    │   │   │   │   │   ├── export_handlers.py
    │   │   │   │   │   └── metadata_handlers.py
    │   │   │   │   ├── legacy/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── experiment_viz.py
    │   │   │   │   │   ├── learning_rate_analysis.py
    │   │   │   │   │   ├── parameter_analysis.py
    │   │   │   │   │   ├── prediction_viz.py
    │   │   │   │   │   └── training_curves.py
    │   │   │   │   ├── prediction/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── confidence.py
    │   │   │   │   │   ├── grid.py
    │   │   │   │   │   └── overlay.py
    │   │   │   │   ├── templates/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── base_template.py
    │   │   │   │   │   ├── prediction_template.py
    │   │   │   │   │   └── training_template.py
    │   │   │   │   ├── training/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── advanced.py
    │   │   │   │   │   ├── analysis.py
    │   │   │   │   │   ├── core.py
    │   │   │   │   │   ├── curves.py
    │   │   │   │   │   └── reports.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── advanced_prediction_viz.py
    │   │   │   │   ├── architecture.md
    │   │   │   │   └── VISUALIZATION_FINAL_REPORT.md
    │   │   │   ├── __init__.py
    │   │   │   ├── __main__.py
    │   │   │   ├── EVALUATION_MODULE_SUMMARY.md
    │   │   │   └── README.md
    │   │   ├── integration/
    │   │   ├── model/
    │   │   │   ├── __pycache__/
    │   │   │   ├── architectures/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── cnn_convlstm_unet.py
    │   │   │   │   ├── registry.py
    │   │   │   │   ├── simple_unet.py
    │   │   │   │   ├── swinv2_cnn_aspp_unet.py
    │   │   │   │   └── unet.py
    │   │   │   ├── base/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── abstract.py
    │   │   │   │   └── abstract.py.backup
    │   │   │   ├── bottleneck/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── cnn_bottleneck.py
    │   │   │   ├── common/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── visualization/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── matplotlib/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── graphviz_renderer.py
    │   │   │   │   │   ├── main.py
    │   │   │   │   │   └── matplotlib_renderer.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── spatial_utils.py
    │   │   │   │   ├── utils.py
    │   │   │   │   └── utils.py.backup
    │   │   │   ├── components/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── aspp.py
    │   │   │   │   ├── attention_decorator.py
    │   │   │   │   ├── cbam.py
    │   │   │   │   ├── convlstm.py
    │   │   │   │   └── registry_support.py
    │   │   │   ├── config/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── core.py
    │   │   │   │   ├── factory.py
    │   │   │   │   ├── instantiation.py
    │   │   │   │   ├── schemas.py
    │   │   │   │   └── validation.py
    │   │   │   ├── core/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── unet.py
    │   │   │   ├── decoder/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── common/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   └── channel_utils.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── cnn_decoder.py
    │   │   │   ├── encoder/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── swin/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── config.py
    │   │   │   │   │   ├── core.py
    │   │   │   │   │   ├── initialization.py
    │   │   │   │   │   ├── preprocessing.py
    │   │   │   │   │   └── transfer_learning.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── cnn_encoder.py
    │   │   │   │   ├── feature_info_utils.py
    │   │   │   │   ├── swin_transformer_encoder.py
    │   │   │   │   └── swin_v2_adapter.py
    │   │   │   ├── factory/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── config.py
    │   │   │   │   ├── config_schema.py
    │   │   │   │   ├── factory.py
    │   │   │   │   ├── factory_utils.py
    │   │   │   │   ├── hybrid_registry.py
    │   │   │   │   ├── registry.py
    │   │   │   │   └── registry_setup.py
    │   │   │   ├── __init__.py
    │   │   │   └── README.md
    │   │   ├── reporting/
    │   │   │   ├── __pycache__/
    │   │   │   ├── comparison/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── engine.py
    │   │   │   ├── figures/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── publication_figure_generator.py
    │   │   │   ├── performance/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── analyzer.py
    │   │   │   │   ├── anomaly_detector.py
    │   │   │   │   ├── metric_evaluator.py
    │   │   │   │   ├── recommendation_engine.py
    │   │   │   │   └── training_analyzer.py
    │   │   │   ├── recommendations/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── analyzers/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── hyperparameters.py
    │   │   │   │   │   ├── performance.py
    │   │   │   │   │   └── training_patterns.py
    │   │   │   │   ├── identifiers/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── architecture.py
    │   │   │   │   │   └── opportunities.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── engine.py
    │   │   │   │   └── thresholds.py
    │   │   │   ├── templates/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── html_templates.py
    │   │   │   │   ├── latex_templates.py
    │   │   │   │   ├── markdown_templates.py
    │   │   │   │   └── template_manager.py
    │   │   │   ├── __init__.py
    │   │   │   ├── config.py
    │   │   │   ├── core.py
    │   │   │   ├── data_loader.py
    │   │   │   └── interfaces.py
    │   │   ├── training/
    │   │   │   ├── __pycache__/
    │   │   │   ├── components/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── initializer.py
    │   │   │   │   ├── setup.py
    │   │   │   │   ├── training_loop.py
    │   │   │   │   └── validation_loop.py
    │   │   │   ├── losses/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── combinators/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── base_combinator.py
    │   │   │   │   │   ├── enhanced_product.py
    │   │   │   │   │   ├── enhanced_weighted_sum.py
    │   │   │   │   │   ├── product.py
    │   │   │   │   │   └── weighted_sum.py
    │   │   │   │   ├── factory/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── config_parser.py
    │   │   │   │   │   ├── config_validator.py
    │   │   │   │   │   └── recursive_factory.py
    │   │   │   │   ├── interfaces/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   └── loss_interface.py
    │   │   │   │   ├── registry/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── clean_registry.py
    │   │   │   │   │   ├── enhanced_registry.py
    │   │   │   │   │   └── setup_losses.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── base_loss.py
    │   │   │   │   ├── bce_dice_loss.py
    │   │   │   │   ├── bce_loss.py
    │   │   │   │   ├── combined_loss.py
    │   │   │   │   ├── dice_loss.py
    │   │   │   │   ├── focal_dice_loss.py
    │   │   │   │   ├── focal_loss.py
    │   │   │   │   ├── loss_registry_setup.py
    │   │   │   │   ├── recursive_factory.py
    │   │   │   │   └── smooth_l1_loss.py
    │   │   │   ├── optimizers/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── custom_adam.py
    │   │   │   │   └── registry.py
    │   │   │   ├── __init__.py
    │   │   │   ├── batch_processing.py
    │   │   │   ├── config_validation.py
    │   │   │   ├── factory.py
    │   │   │   ├── metrics.py
    │   │   │   ├── README.md
    │   │   │   ├── trainer.py
    │   │   │   └── trainer.py.backup
    │   │   ├── utils/
    │   │   │   ├── __pycache__/
    │   │   │   ├── artifact_manager/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── core.py
    │   │   │   │   ├── metadata.py
    │   │   │   │   ├── storage.py
    │   │   │   │   ├── validation.py
    │   │   │   │   └── versioning.py
    │   │   │   ├── checkpointing/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── config.py
    │   │   │   │   ├── helpers.py
    │   │   │   │   ├── legacy.py
    │   │   │   │   ├── load.py
    │   │   │   │   ├── save.py
    │   │   │   │   ├── setup.py
    │   │   │   │   └── validation.py
    │   │   │   ├── config/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── env.py
    │   │   │   │   ├── init.py
    │   │   │   │   ├── override.py
    │   │   │   │   ├── schema.py
    │   │   │   │   ├── standardized_storage.py
    │   │   │   │   └── validation.py
    │   │   │   ├── core/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── device.py
    │   │   │   │   ├── exceptions.py
    │   │   │   │   ├── paths.py
    │   │   │   │   └── seeds.py
    │   │   │   ├── deployment/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── artifacts/
    │   │   │   │   ├── config/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── environment/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── deployment.py
    │   │   │   │   │   └── handlers.py
    │   │   │   │   ├── core/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── manager.py
    │   │   │   │   │   ├── orchestrator.py
    │   │   │   │   │   └── types.py
    │   │   │   │   ├── monitoring/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── config.py
    │   │   │   │   │   ├── core.py
    │   │   │   │   │   ├── health.py
    │   │   │   │   │   ├── metrics.py
    │   │   │   │   │   ├── performance.py
    │   │   │   │   │   └── resource.py
    │   │   │   │   ├── packaging/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── config.py
    │   │   │   │   │   ├── containerization.py
    │   │   │   │   │   ├── core.py
    │   │   │   │   │   ├── dependencies.py
    │   │   │   │   │   ├── docker_compose.py
    │   │   │   │   │   ├── file_generators.py
    │   │   │   │   │   ├── helm.py
    │   │   │   │   │   ├── kubernetes.py
    │   │   │   │   │   ├── manifests.py
    │   │   │   │   │   ├── metrics.py
    │   │   │   │   │   └── security.py
    │   │   │   │   ├── templates/
    │   │   │   │   │   └── validation_report.md.j2
    │   │   │   │   ├── utils/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── multi_target.py
    │   │   │   │   │   └── production.py
    │   │   │   │   ├── validation/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── pipeline/
    │   │   │   │   │   ├── reporting/
    │   │   │   │   │   └── __init__.py
    │   │   │   │   └── __init__.py
    │   │   │   ├── experiment/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── tracker/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── tracker_artifacts.py
    │   │   │   │   │   ├── tracker_config.py
    │   │   │   │   │   ├── tracker_git.py
    │   │   │   │   │   └── tracker_lifecycle.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── experiment.py
    │   │   │   │   ├── manager.py
    │   │   │   │   ├── metadata.py
    │   │   │   │   └── tracker.py
    │   │   │   ├── factory/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── cache.py
    │   │   │   │   └── factory.py
    │   │   │   ├── integrity/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── artifact_verifier.py
    │   │   │   │   ├── checkpoint_verifier.py
    │   │   │   │   ├── config_verifier.py
    │   │   │   │   ├── core.py
    │   │   │   │   └── experiment_verifier.py
    │   │   │   ├── logging/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── base.py
    │   │   │   │   ├── experiment.py
    │   │   │   │   ├── metrics_manager.py
    │   │   │   │   ├── setup.py
    │   │   │   │   └── training.py
    │   │   │   ├── monitoring/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── alerts/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── checker.py
    │   │   │   │   │   ├── system.py
    │   │   │   │   │   └── types.py
    │   │   │   │   ├── callbacks/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── base.py
    │   │   │   │   │   ├── gpu.py
    │   │   │   │   │   └── system.py
    │   │   │   │   ├── coverage/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── alerts.py
    │   │   │   │   │   ├── analysis.py
    │   │   │   │   │   ├── config.py
    │   │   │   │   │   ├── core.py
    │   │   │   │   │   ├── reporting.py
    │   │   │   │   │   └── trends.py
    │   │   │   │   ├── resources/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── config.py
    │   │   │   │   │   ├── monitor.py
    │   │   │   │   │   └── snapshot.py
    │   │   │   │   ├── retention/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   └── policies.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── exceptions.py
    │   │   │   │   └── manager.py
    │   │   │   ├── traceability/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── access_control.py
    │   │   │   │   ├── compliance.py
    │   │   │   │   ├── entities.py
    │   │   │   │   ├── enums.py
    │   │   │   │   ├── integration_manager.py
    │   │   │   │   ├── lineage_manager.py
    │   │   │   │   ├── metadata_manager.py
    │   │   │   │   ├── models.py
    │   │   │   │   ├── queries.py
    │   │   │   │   ├── query_interface.py
    │   │   │   │   └── storage.py
    │   │   │   ├── training/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── amp_utils.py
    │   │   │   │   ├── early_stopping.py
    │   │   │   │   ├── early_stopping_setup.py
    │   │   │   │   └── scheduler_helper.py
    │   │   │   ├── visualization/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── plots.py
    │   │   │   ├── __init__.py
    │   │   │   ├── artifact_manager.py
    │   │   │   ├── component_cache.py
    │   │   │   ├── experiment_saver.py
    │   │   │   ├── README.md
    │   │   │   └── UTILS_COMPREHENSIVE_REORGANIZATION_SUMMARY.md
    │   │   ├── __init__.py
    │   │   ├── __main__.py
    │   │   ├── FILE_REORGANIZATION_REPORT.md
    │   │   └── README.md
    │   ├── crackseg.egg-info/
    │   ├── training_pipeline/
    │   │   ├── __pycache__/
    │   │   ├── __init__.py
    │   │   ├── checkpoint_manager.py
    │   │   ├── data_loading.py
    │   │   ├── environment_setup.py
    │   │   ├── model_creation.py
    │   │   └── training_setup.py
    │   └── main.py
    ├── tests/
    │   ├── __pycache__/
    │   ├── docker/
    │   ├── e2e/
    │   │   ├── __pycache__/
    │   │   ├── capture/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── screenshot.py
    │   │   │   ├── storage.py
    │   │   │   ├── video.py
    │   │   │   └── visual_regression.py
    │   │   ├── cleanup/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── audit_trail.py
    │   │   │   ├── ci_integration.py
    │   │   │   ├── cleanup_manager.py
    │   │   │   ├── environment_readiness.py
    │   │   │   ├── post_cleanup_validator.py
    │   │   │   ├── resource_cleanup.py
    │   │   │   ├── validation_reporter.py
    │   │   │   └── validation_system.py
    │   │   ├── config/
    │   │   │   ├── __pycache__/
    │   │   │   ├── viewport_config/
    │   │   │   │   ├── devices/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── core.py
    │   │   │   │   │   ├── defaults.py
    │   │   │   │   │   └── factories.py
    │   │   │   │   ├── matrix/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── core.py
    │   │   │   │   │   └── presets.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── core.py
    │   │   │   ├── __init__.py
    │   │   │   ├── browser_capabilities.py
    │   │   │   ├── browser_config_manager.py
    │   │   │   ├── browser_matrix_config.py
    │   │   │   ├── cross_browser_test.py
    │   │   │   ├── execution_strategies.py
    │   │   │   ├── parallel_execution_config.py
    │   │   │   ├── parallel_performance_integration.py
    │   │   │   ├── performance_thresholds.py
    │   │   │   ├── pytest_markers.py
    │   │   │   ├── resource_manager.py
    │   │   │   ├── test_parallel_framework_validation.py
    │   │   │   └── threshold_validator.py
    │   │   ├── data/
    │   │   ├── drivers/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── config.py
    │   │   │   ├── driver_factory.py
    │   │   │   ├── driver_manager.py
    │   │   │   └── exceptions.py
    │   │   ├── helpers/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── api_integration.py
    │   │   │   ├── performance_monitoring.py
    │   │   │   ├── setup_teardown.py
    │   │   │   └── test_coordination.py
    │   │   ├── maintenance/
    │   │   │   ├── __init__.py
    │   │   │   ├── config.py
    │   │   │   ├── core.py
    │   │   │   ├── health_monitor.py
    │   │   │   └── models.py
    │   │   ├── mixins/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── capture_mixin.py
    │   │   │   ├── logging_mixin.py
    │   │   │   ├── performance_mixin.py
    │   │   │   ├── retry_mixin.py
    │   │   │   └── streamlit_mixin.py
    │   │   ├── pages/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── advanced_config_page.py
    │   │   │   ├── architecture_page.py
    │   │   │   ├── base_page.py
    │   │   │   ├── config_page.py
    │   │   │   ├── locators.py
    │   │   │   ├── results_page.py
    │   │   │   └── train_page.py
    │   │   ├── performance/
    │   │   │   ├── __pycache__/
    │   │   │   ├── reporting/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── analysis.py
    │   │   │   │   ├── comparison_charts.py
    │   │   │   │   ├── config.py
    │   │   │   │   ├── core.py
    │   │   │   │   ├── data_processor.py
    │   │   │   │   ├── factory_functions.py
    │   │   │   │   ├── formats.py
    │   │   │   │   ├── historical_manager.py
    │   │   │   │   ├── html_formatter.py
    │   │   │   │   ├── insights_generator.py
    │   │   │   │   ├── json_formatter.py
    │   │   │   │   ├── metric_extractor.py
    │   │   │   │   ├── pdf_formatter.py
    │   │   │   │   ├── regression_analyzer.py
    │   │   │   │   ├── summary_charts.py
    │   │   │   │   ├── templates.py
    │   │   │   │   ├── trend_analyzer.py
    │   │   │   │   ├── trend_charts.py
    │   │   │   │   └── visualizations.py
    │   │   │   ├── __init__.py
    │   │   │   ├── benchmark_runner.py
    │   │   │   ├── benchmark_suite.py
    │   │   │   ├── ci_integration.py
    │   │   │   ├── endurance_test.py
    │   │   │   ├── load_test.py
    │   │   │   ├── metrics_collector.py
    │   │   │   ├── regression_alerting_system.py
    │   │   │   └── stress_test.py
    │   │   ├── reporting/
    │   │   │   ├── analysis/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── failure_analyzer.py
    │   │   │   │   └── trend_analyzer.py
    │   │   │   ├── __init__.py
    │   │   │   ├── config.py
    │   │   │   ├── core.py
    │   │   │   ├── exporters.py
    │   │   │   ├── generator.py
    │   │   │   └── models.py
    │   │   ├── session/
    │   │   │   ├── __init__.py
    │   │   │   ├── cookie_manager.py
    │   │   │   ├── mixins.py
    │   │   │   ├── state_manager.py
    │   │   │   ├── storage_manager.py
    │   │   │   └── streamlit_session.py
    │   │   ├── test_data/
    │   │   │   └── invalid_configs/
    │   │   │       ├── conflicting_dependencies.yaml
    │   │   │       ├── invalid_syntax.yaml
    │   │   │       └── missing_required_fields.yaml
    │   │   ├── tests/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── test_capture_demonstration.py
    │   │   │   ├── test_cross_browser_compatibility.py
    │   │   │   ├── test_edge_cases.py
    │   │   │   ├── test_edge_cases_boundary.py
    │   │   │   ├── test_edge_cases_concurrent.py
    │   │   │   ├── test_edge_cases_corruption.py
    │   │   │   ├── test_edge_cases_interactions.py
    │   │   │   ├── test_edge_cases_performance.py
    │   │   │   ├── test_edge_cases_resources.py
    │   │   │   ├── test_error_scenarios.py
    │   │   │   ├── test_happy_path.py
    │   │   │   └── test_performance_integration.py
    │   │   ├── utils/
    │   │   │   ├── __pycache__/
    │   │   │   ├── responsive/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── layout.py
    │   │   │   │   ├── testing.py
    │   │   │   │   ├── touch.py
    │   │   │   │   └── viewport.py
    │   │   │   ├── __init__.py
    │   │   │   ├── browser_validation.py
    │   │   │   ├── config.py
    │   │   │   ├── data.py
    │   │   │   ├── debugging.py
    │   │   │   ├── element.py
    │   │   │   ├── file.py
    │   │   │   ├── README_TestEnvironmentManager.md
    │   │   │   ├── streamlit.py
    │   │   │   ├── string.py
    │   │   │   ├── test_environment_fixtures.py
    │   │   │   ├── test_environment_manager.py
    │   │   │   └── time.py
    │   │   ├── waits/
    │   │   │   ├── __init__.py
    │   │   │   ├── conditions.py
    │   │   │   └── strategies.py
    │   │   ├── __init__.py
    │   │   ├── base_test.py
    │   │   ├── conftest.py
    │   │   ├── README.md
    │   │   ├── test_cleanup_integration.py
    │   │   ├── test_driver_integration.py
    │   │   ├── test_environment_setup_demo.py
    │   │   ├── test_fixture_usage_example.py
    │   │   ├── test_streamlit_basic.py
    │   │   └── test_workflow_regression_4_4.py
    │   ├── examples/
    │   │   ├── __pycache__/
    │   │   ├── enhanced_gui_testing_demo.py
    │   │   └── visual_regression_demo.py
    │   ├── fixtures/
    │   │   └── mocks/
    │   │       ├── experiment_manager/
    │   │       └── README.md
    │   ├── gui/
    │   │   ├── __pycache__/
    │   │   ├── test_auto_save.py
    │   │   ├── test_confirmation_dialog.py
    │   │   ├── test_device_selector.py
    │   │   ├── test_error_state.py
    │   │   ├── test_loading_spinner.py
    │   │   ├── test_performance_optimization.py
    │   │   ├── test_progress_bar.py
    │   │   └── test_results_gallery_component.py
    │   ├── integration/
    │   │   ├── __pycache__/
    │   │   ├── config/
    │   │   │   ├── __pycache__/
    │   │   │   └── test_hydra_config.py
    │   │   ├── data/
    │   │   ├── end_to_end/
    │   │   ├── evaluation/
    │   │   │   ├── __pycache__/
    │   │   │   └── test_evaluation_pipeline.py
    │   │   ├── gui/
    │   │   │   ├── __pycache__/
    │   │   │   ├── automation/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── reporting/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── analysis/
    │   │   │   │   │   ├── data_collectors/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── analysis_engine.py
    │   │   │   │   │   ├── content_generators.py
    │   │   │   │   │   ├── csv_export.py
    │   │   │   │   │   ├── data_aggregation.py
    │   │   │   │   │   ├── export_manager.py
    │   │   │   │   │   ├── html_export.py
    │   │   │   │   │   ├── integration_test_reporting.py
    │   │   │   │   │   ├── json_export.py
    │   │   │   │   │   ├── metrics_compiler.py
    │   │   │   │   │   ├── regression_detection.py
    │   │   │   │   │   ├── stakeholder_reporting.py
    │   │   │   │   │   ├── trend_analysis.py
    │   │   │   │   │   ├── trend_analyzers.py
    │   │   │   │   │   ├── trend_predictions.py
    │   │   │   │   │   └── validation_utils.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── automation_orchestrator.py
    │   │   │   │   ├── automation_protocols.py
    │   │   │   │   ├── ci_integration.py
    │   │   │   │   ├── performance_benchmarking.py
    │   │   │   │   ├── resource_cleanup_monitoring.py
    │   │   │   │   ├── resource_cleanup_protocols.py
    │   │   │   │   ├── resource_cleanup_validation.py
    │   │   │   │   ├── run_performance_benchmarking_tests.py
    │   │   │   │   ├── test_automation_execution.py
    │   │   │   │   ├── test_data_automation.py
    │   │   │   │   ├── test_performance_benchmarking.py
    │   │   │   │   ├── test_resource_cleanup_validation.py
    │   │   │   │   └── workflow_automation.py
    │   │   │   ├── concurrent_tests/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── test_multi_user_operations.py
    │   │   │   │   ├── test_resource_contention.py
    │   │   │   │   └── test_system_stability.py
    │   │   │   ├── helpers/
    │   │   │   ├── workflow_components/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── concurrent/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── base.py
    │   │   │   │   │   ├── data_integrity.py
    │   │   │   │   │   ├── multi_user.py
    │   │   │   │   │   ├── resource_contention.py
    │   │   │   │   │   ├── stability.py
    │   │   │   │   │   └── synchronization.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── component_interaction_workflow.py
    │   │   │   │   ├── config_error_component.py
    │   │   │   │   ├── config_workflow.py
    │   │   │   │   ├── error_scenario_mixin.py
    │   │   │   │   ├── session_state_mixin.py
    │   │   │   │   ├── training_error_component.py
    │   │   │   │   └── training_workflow.py
    │   │   │   ├── __init__.py
    │   │   │   ├── test_advanced_workflows.py
    │   │   │   ├── test_base.py
    │   │   │   ├── test_basic_workflows.py
    │   │   │   ├── test_component_interactions.py
    │   │   │   ├── test_concurrent_operations.py
    │   │   │   ├── test_config_editor_component.py
    │   │   │   ├── test_config_io.py
    │   │   │   ├── test_error_scenarios.py
    │   │   │   ├── test_file_browser_component.py
    │   │   │   ├── test_session_state_simple.py
    │   │   │   ├── test_session_state_verification.py
    │   │   │   ├── test_specialized_config.py
    │   │   │   ├── test_specialized_parsing.py
    │   │   │   ├── test_specialized_run_manager.py
    │   │   │   ├── test_specialized_streaming.py
    │   │   │   ├── test_specialized_tensorboard.py
    │   │   │   ├── test_specialized_threading.py
    │   │   │   ├── test_workflow_performance.py
    │   │   │   ├── test_workflow_scenarios.py
    │   │   │   ├── test_yaml_validation.py
    │   │   │   └── workflow_scenarios.py
    │   │   ├── model/
    │   │   │   ├── __pycache__/
    │   │   │   ├── factory/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   └── test_registry_integration.py
    │   │   │   ├── conftest.py
    │   │   │   ├── test_cbam_integration.py
    │   │   │   ├── test_cnn_convlstm_unet.py
    │   │   │   ├── test_config_validation.py
    │   │   │   ├── test_factory_config.py
    │   │   │   ├── test_factory_instantiation_flow.py
    │   │   │   ├── test_integration.py
    │   │   │   ├── test_model_factory.py
    │   │   │   ├── test_swin_integration.py
    │   │   │   ├── test_swin_transfer_learning.py
    │   │   │   ├── test_swin_unet_integration.py
    │   │   │   └── test_unet_aspp_integration.py
    │   │   ├── monitoring/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   └── test_monitoring_integration.py
    │   │   ├── reporting/
    │   │   │   ├── __pycache__/
    │   │   │   ├── analysis/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── test_data_aggregation.py
    │   │   │   │   └── test_trend_analysis.py
    │   │   │   ├── __init__.py
    │   │   │   ├── test_automated_comparison.py
    │   │   │   ├── test_end_to_end_reporting.py
    │   │   │   └── test_sample_report_generation.py
    │   │   ├── training/
    │   │   │   ├── __pycache__/
    │   │   │   ├── test_artifacts_performance_regression.py
    │   │   │   ├── test_config_parser_validation.py
    │   │   │   ├── test_enhanced_combinators_validation.py
    │   │   │   ├── test_enhanced_registry_validation.py
    │   │   │   ├── test_loss_factory_integration.py
    │   │   │   ├── test_standardized_config_integration.py
    │   │   │   ├── test_trainer_integration.py
    │   │   │   ├── test_training_artifacts_integration.py
    │   │   │   └── test_training_loop.py
    │   │   ├── utils/
    │   │   │   ├── __pycache__/
    │   │   │   ├── conftest.py
    │   │   │   ├── test_orchestration.py
    │   │   │   ├── test_packaging_system.py
    │   │   │   ├── test_traceability_access.py
    │   │   │   ├── test_traceability_advanced_workflows.py
    │   │   │   ├── test_traceability_bulk_operations.py
    │   │   │   ├── test_traceability_operations.py
    │   │   │   └── test_traceability_workflows.py
    │   │   ├── visualization/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── test_plotly_integration.py
    │   │   │   └── test_training_visualization.py
    │   │   └── test_backward_compatibility.py
    │   ├── tools/
    │   │   ├── analysis/
    │   │   │   ├── __pycache__/
    │   │   │   ├── comprehensive_failure_analysis.py
    │   │   │   ├── failure_data.json
    │   │   │   ├── pytest_executor.py
    │   │   │   ├── pytest_output_parser.py
    │   │   │   ├── report_generator.py
    │   │   │   ├── test_failure_analysis.py
    │   │   │   ├── test_failure_categorization.py
    │   │   │   └── test_priority_matrix_creator.py
    │   │   ├── benchmark/
    │   │   │   └── benchmark_tests.py
    │   │   ├── coverage/
    │   │   │   ├── check_test_files.py
    │   │   │   └── coverage_check.sh
    │   │   ├── execution/
    │   │   │   ├── run_tests_phased.py
    │   │   │   └── simple_install_check.sh
    │   │   ├── quality/
    │   │   │   └── validate_test_quality.py
    │   │   ├── testing/
    │   │   │   ├── __pycache__/
    │   │   │   └── test_config_system.py
    │   │   ├── utilities/
    │   │   │   └── temp_storage.py
    │   │   └── README.md
    │   ├── tutorials/
    │   │   ├── README.md
    │   │   └── tutorial_03_verification.py
    │   ├── unit/
    │   │   ├── __pycache__/
    │   │   ├── data/
    │   │   ├── deployment/
    │   │   │   ├── __pycache__/
    │   │   │   ├── test_health_monitoring.py
    │   │   │   ├── test_multi_target.py
    │   │   │   ├── test_orchestration.py
    │   │   │   └── test_production_readiness_validator.py
    │   │   ├── docker/
    │   │   │   ├── __pycache__/
    │   │   │   ├── test_env_manager.py
    │   │   │   └── test_health_check_system.py
    │   │   ├── e2e/
    │   │   │   ├── capture/
    │   │   │   ├── cleanup/
    │   │   │   ├── config/
    │   │   │   ├── performance/
    │   │   │   │   └── reporting/
    │   │   │   └── waits/
    │   │   ├── evaluation/
    │   │   │   ├── __pycache__/
    │   │   │   ├── test_advanced_training_viz.py
    │   │   │   ├── test_core.py
    │   │   │   ├── test_data.py
    │   │   │   ├── test_ensemble.py
    │   │   │   ├── test_evaluate.py
    │   │   │   ├── test_evaluation_main.py
    │   │   │   ├── test_loading.py
    │   │   │   └── test_results.py
    │   │   ├── gui/
    │   │   │   ├── __pycache__/
    │   │   │   ├── components/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── conftest.py
    │   │   │   │   ├── test_component_base.py
    │   │   │   │   ├── test_file_browser_component.py
    │   │   │   │   ├── test_file_upload_component.py
    │   │   │   │   ├── test_logo_component.py
    │   │   │   │   ├── test_page_router.py
    │   │   │   │   ├── test_results_display.py
    │   │   │   │   ├── test_sidebar_component.py
    │   │   │   │   └── test_theme_component.py
    │   │   │   ├── pages/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── test_advanced_config_page.py
    │   │   │   │   ├── test_config_page.py
    │   │   │   │   ├── test_home_page.py
    │   │   │   │   ├── test_pages_smoke.py
    │   │   │   │   └── test_train_page.py
    │   │   │   ├── utils/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── config/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── validation/
    │   │   │   │   │   ├── test_cache.py
    │   │   │   │   │   ├── test_formatters.py
    │   │   │   │   │   ├── test_io.py
    │   │   │   │   │   └── test_templates.py
    │   │   │   │   ├── test_export_manager.py
    │   │   │   │   ├── test_gui_config.py
    │   │   │   │   ├── test_performance_optimizer.py
    │   │   │   │   └── test_session_state.py
    │   │   │   ├── __init__.py
    │   │   │   ├── test_critical_coverage_paths.py
    │   │   │   ├── test_edge_cases.py
    │   │   │   ├── test_enhanced_abort.py
    │   │   │   ├── test_error_console.py
    │   │   │   ├── test_error_console_simple.py
    │   │   │   ├── test_essential_coverage.py
    │   │   │   ├── test_file_upload.py
    │   │   │   ├── test_session_state_updates.py
    │   │   │   ├── test_tensorboard_coverage.py
    │   │   │   └── test_threading_integration.py
    │   │   ├── integration/
    │   │   │   └── gui/
    │   │   │       └── automation/
    │   │   │           └── reporting/
    │   │   ├── model/
    │   │   │   ├── __pycache__/
    │   │   │   ├── config/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   └── test_instantiation.py
    │   │   │   ├── decoder/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── test_channel_utils.py
    │   │   │   │   ├── test_cnn_decoder_channel_handling.py
    │   │   │   │   ├── test_cnn_decoder_error_handling.py
    │   │   │   │   ├── test_cnn_decoder_forward_pass.py
    │   │   │   │   ├── test_cnn_decoder_initialization.py
    │   │   │   │   └── test_cnn_decoder_special_features.py
    │   │   │   ├── factory/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   └── test_registry.py
    │   │   │   ├── architectures
    │   │   │   ├── conftest.py
    │   │   │   ├── test_aspp.py
    │   │   │   ├── test_base.py
    │   │   │   ├── test_bottleneckblock.py
    │   │   │   ├── test_cbam.py
    │   │   │   ├── test_cbam_config.py
    │   │   │   ├── test_cnn_encoder.py
    │   │   │   ├── test_convlstm.py
    │   │   │   ├── test_decoderblock.py
    │   │   │   ├── test_encoderblock.py
    │   │   │   ├── test_exports.py
    │   │   │   ├── test_factory_utils.py
    │   │   │   ├── test_feature_info_utils.py
    │   │   │   ├── test_hybrid_registry.py
    │   │   │   ├── test_import_compat.py
    │   │   │   ├── test_registry.py
    │   │   │   ├── test_swin_basic.py
    │   │   │   ├── test_swin_encoder.py
    │   │   │   ├── test_swin_transfer_learning_script.py
    │   │   │   ├── test_swin_transformer_encoder.py
    │   │   │   ├── test_swin_unet.py
    │   │   │   ├── test_thread_safety.py
    │   │   │   ├── test_unet.py
    │   │   │   └── test_utils.py
    │   │   ├── reporting/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── test_configurable_templates.py
    │   │   │   ├── test_publication_figures.py
    │   │   │   └── test_recommendation_engine.py
    │   │   ├── training/
    │   │   │   ├── __pycache__/
    │   │   │   ├── losses/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── test_clean_factory.py
    │   │   │   │   ├── test_config_parser.py
    │   │   │   │   ├── test_enhanced_combinators.py
    │   │   │   │   ├── test_focal_dice_loss.py
    │   │   │   │   ├── test_isolated_clean_factory.py
    │   │   │   │   ├── test_loss_factory.py
    │   │   │   │   ├── test_loss_registry.py
    │   │   │   │   ├── test_recursive_factory.py
    │   │   │   │   ├── test_recursive_factory_basic.py
    │   │   │   │   ├── test_recursive_factory_combinations.py
    │   │   │   │   ├── test_recursive_factory_config.py
    │   │   │   │   ├── test_recursive_factory_errors.py
    │   │   │   │   ├── test_recursive_factory_performance.py
    │   │   │   │   └── test_recursive_factory_regression.py
    │   │   │   ├── test_losses.py
    │   │   │   ├── test_lr_scheduler_factory.py
    │   │   │   ├── test_metrics.py
    │   │   │   ├── test_reproducibility.py
    │   │   │   ├── test_trainer.py
    │   │   │   ├── test_trainer_initialization.py
    │   │   │   └── test_trainer_training.py
    │   │   ├── utils/
    │   │   │   ├── __pycache__/
    │   │   │   ├── artifacts/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.cpython-312.pyc
    │   │   │   │   │   ├── test_artifact_manager.cpython-312-pytest-8.4.1.pyc
    │   │   │   │   │   ├── test_artifact_versioner.cpython-312-pytest-8.4.1.pyc
    │   │   │   │   │   └── test_checkpointing.cpython-312-pytest-8.4.1.pyc
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── test_artifact_manager.py
    │   │   │   │   ├── test_artifact_versioner.py
    │   │   │   │   └── test_checkpointing.py
    │   │   │   ├── config/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── test_env.py
    │   │   │   │   ├── test_override.py
    │   │   │   │   ├── test_schema.py
    │   │   │   │   ├── test_standardized_storage.py
    │   │   │   │   └── test_validation.py
    │   │   │   ├── data/
    │   │   │   ├── experiment/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── test_experiment_metadata.py
    │   │   │   │   ├── test_experiment_tracker.py
    │   │   │   │   ├── test_experiment_tracker_artifacts.py
    │   │   │   │   └── test_experiment_tracker_lifecycle.py
    │   │   │   ├── integrity/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── test_artifact_verifier.py
    │   │   │   │   ├── test_checkpoint_verifier.py
    │   │   │   │   ├── test_config_verifier.py
    │   │   │   │   ├── test_experiment_verifier.py
    │   │   │   │   └── test_integrity_core.py
    │   │   │   ├── logging/
    │   │   │   │   └── __init__.py
    │   │   │   ├── monitoring/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── test_callbacks.py
    │   │   │   │   ├── test_monitoring_manager.py
    │   │   │   │   └── test_retention.py
    │   │   │   ├── monitoring_logging/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── test_logging.py
    │   │   │   │   └── test_metrics_manager.py
    │   │   │   ├── traceability/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── test_integration_manager.py
    │   │   │   │   ├── test_integration_manager_bulk.py
    │   │   │   │   ├── test_integration_manager_compliance.py
    │   │   │   │   ├── test_integration_manager_search.py
    │   │   │   │   ├── test_lineage_manager.py
    │   │   │   │   ├── test_query_interface.py
    │   │   │   │   └── test_storage.py
    │   │   │   └── training/
    │   │   │       ├── __pycache__/
    │   │   │       ├── __init__.py
    │   │   │       └── test_early_stopping.py
    │   │   ├── __init__.py
    │   │   ├── test_main_data.py
    │   │   ├── test_main_environment.py
    │   │   ├── test_main_integration.py
    │   │   ├── test_main_model.py
    │   │   └── test_main_training.py
    │   ├── utils/
    │   │   ├── __pycache__/
    │   │   ├── unified_testing/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── __init__.py.backup
    │   │   │   ├── core.py
    │   │   │   ├── core.py.backup
    │   │   │   ├── helpers.py
    │   │   │   ├── helpers.py.backup
    │   │   │   ├── mocking.py
    │   │   │   ├── mocking.py.backup
    │   │   │   ├── performance.py
    │   │   │   ├── performance.py.backup
    │   │   │   └── visual.py
    │   │   ├── __init__.py
    │   │   ├── performance_optimizer.py
    │   │   ├── pytest_performance_plugin.py
    │   │   ├── test_benchmark.py
    │   │   └── visual_regression_benchmarks.py
    │   ├── __init__.py
    │   ├── conftest.py
    │   ├── README.md
    │   └── requirements-testing.txt
    ├── analisis_metricas_cfd.json
    ├── CHANGELOG.md
    ├── codecov.yml
    ├── environment.yml
    ├── mkdocs.yml
    ├── pyproject.toml
    ├── pyrightconfig.json
    ├── README.md
    ├── requirements.txt
    ├── run.py
    └── TODOs.md
```
