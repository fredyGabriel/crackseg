# Project Directory Structure (excluding .gitignore)

```txt
└── crackseg/
    ├── artifacts/
    ├── configs/
    │   ├── data/
    │   │   ├── dataloader/
    │   │   │   └── default.yaml
    │   │   ├── transform/
    │   │   │   └── augmentations.yaml
    │   │   ├── default.yaml
    │   │   └── README.md
    │   ├── evaluation/
    │   │   └── default.yaml
    │   ├── linting/
    │   │   └── config.yaml
    │   ├── model/
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
    │   │   │   └── focal.yaml
    │   │   ├── lr_scheduler/
    │   │   │   ├── cosine.yaml
    │   │   │   ├── reduce_on_plateau.yaml
    │   │   │   └── step_lr.yaml
    │   │   ├── metric/
    │   │   │   ├── f1.yaml
    │   │   │   ├── iou.yaml
    │   │   │   ├── precision.yaml
    │   │   │   └── recall.yaml
    │   │   ├── default.yaml
    │   │   ├── README.md
    │   │   └── trainer.yaml
    │   ├── __init__.py
    │   ├── base.yaml
    │   ├── basic_verification.yaml
    │   ├── config.yaml
    │   └── README.md
    ├── data/
    │   ├── test/
    │   │   ├── images/
    │   │   │   ├── 101.jpg
    │   │   │   ├── 102.jpg
    │   │   │   ├── 104.jpg
    │   │   │   ├── 109.jpg
    │   │   │   ├── 110.jpg
    │   │   │   ├── 114.jpg
    │   │   │   ├── 123.jpg
    │   │   │   ├── 124.jpg
    │   │   │   ├── 125.jpg
    │   │   │   ├── 127.jpg
    │   │   │   ├── 30.jpg
    │   │   │   ├── 44.jpg
    │   │   │   ├── 45.jpg
    │   │   │   ├── 5.jpg
    │   │   │   ├── 6.jpg
    │   │   │   ├── 67.jpg
    │   │   │   ├── 85.jpg
    │   │   │   ├── 88.jpg
    │   │   │   ├── 93.jpg
    │   │   │   └── 99.jpg
    │   │   └── masks/
    │   │       ├── 101.png
    │   │       ├── 102.png
    │   │       ├── 104.png
    │   │       ├── 109.png
    │   │       ├── 110.png
    │   │       ├── 114.png
    │   │       ├── 123.png
    │   │       ├── 124.png
    │   │       ├── 125.png
    │   │       ├── 127.png
    │   │       ├── 30.png
    │   │       ├── 44.png
    │   │       ├── 45.png
    │   │       ├── 5.png
    │   │       ├── 6.png
    │   │       ├── 67.png
    │   │       ├── 85.png
    │   │       ├── 88.png
    │   │       ├── 93.png
    │   │       └── 99.png
    │   ├── train/
    │   │   ├── images/
    │   │   │   ├── 10.jpg
    │   │   │   ├── 100.jpg
    │   │   │   ├── 103.jpg
    │   │   │   ├── 105.jpg
    │   │   │   ├── 106.jpg
    │   │   │   ├── 107.jpg
    │   │   │   ├── 108.jpg
    │   │   │   ├── 11.jpg
    │   │   │   ├── 111.jpg
    │   │   │   ├── 112.jpg
    │   │   │   ├── 113.jpg
    │   │   │   ├── 115.jpg
    │   │   │   ├── 116.jpg
    │   │   │   ├── 118.jpg
    │   │   │   ├── 119.jpg
    │   │   │   ├── 12.jpg
    │   │   │   ├── 120.jpg
    │   │   │   ├── 122.jpg
    │   │   │   ├── 126.jpg
    │   │   │   ├── 128.jpg
    │   │   │   ├── 129.jpg
    │   │   │   ├── 13.jpg
    │   │   │   ├── 14.jpg
    │   │   │   ├── 15.jpg
    │   │   │   ├── 16.jpg
    │   │   │   ├── 17.jpg
    │   │   │   ├── 18.jpg
    │   │   │   ├── 19.jpg
    │   │   │   ├── 2.jpg
    │   │   │   ├── 21.jpg
    │   │   │   ├── 22.jpg
    │   │   │   ├── 23.jpg
    │   │   │   ├── 24.jpg
    │   │   │   ├── 25.jpg
    │   │   │   ├── 26.jpg
    │   │   │   ├── 27.jpg
    │   │   │   ├── 28.jpg
    │   │   │   ├── 29.jpg
    │   │   │   ├── 31.jpg
    │   │   │   ├── 32.jpg
    │   │   │   ├── 34.jpg
    │   │   │   ├── 35.jpg
    │   │   │   ├── 36.jpg
    │   │   │   ├── 37.jpg
    │   │   │   ├── 38.jpg
    │   │   │   ├── 39.jpg
    │   │   │   ├── 40.jpg
    │   │   │   ├── 41.jpg
    │   │   │   ├── 42.jpg
    │   │   │   ├── 43.jpg
    │   │   │   ├── 47.jpg
    │   │   │   ├── 48.jpg
    │   │   │   ├── 49.jpg
    │   │   │   ├── 50.jpg
    │   │   │   ├── 52.jpg
    │   │   │   ├── 53.jpg
    │   │   │   ├── 54.jpg
    │   │   │   ├── 55.jpg
    │   │   │   ├── 56.jpg
    │   │   │   ├── 58.jpg
    │   │   │   ├── 59.jpg
    │   │   │   ├── 60.jpg
    │   │   │   ├── 61.jpg
    │   │   │   ├── 63.jpg
    │   │   │   ├── 65.jpg
    │   │   │   ├── 66.jpg
    │   │   │   ├── 68.jpg
    │   │   │   ├── 69.jpg
    │   │   │   ├── 7.jpg
    │   │   │   ├── 70.jpg
    │   │   │   ├── 71.jpg
    │   │   │   ├── 72.jpg
    │   │   │   ├── 73.jpg
    │   │   │   ├── 75.jpg
    │   │   │   ├── 76.jpg
    │   │   │   ├── 77.jpg
    │   │   │   ├── 78.jpg
    │   │   │   ├── 79.jpg
    │   │   │   ├── 8.jpg
    │   │   │   ├── 80.jpg
    │   │   │   ├── 81.jpg
    │   │   │   ├── 82.jpg
    │   │   │   ├── 83.jpg
    │   │   │   ├── 9.jpg
    │   │   │   ├── 90.jpg
    │   │   │   ├── 91.jpg
    │   │   │   ├── 92.jpg
    │   │   │   ├── 94.jpg
    │   │   │   ├── 96.jpg
    │   │   │   ├── 97.jpg
    │   │   │   └── 98.jpg
    │   │   └── masks/
    │   │       ├── 10.png
    │   │       ├── 100.png
    │   │       ├── 103.png
    │   │       ├── 105.png
    │   │       ├── 106.png
    │   │       ├── 107.png
    │   │       ├── 108.png
    │   │       ├── 11.png
    │   │       ├── 111.png
    │   │       ├── 112.png
    │   │       ├── 113.png
    │   │       ├── 115.png
    │   │       ├── 116.png
    │   │       ├── 118.png
    │   │       ├── 119.png
    │   │       ├── 12.png
    │   │       ├── 120.png
    │   │       ├── 122.png
    │   │       ├── 126.png
    │   │       ├── 128.png
    │   │       ├── 129.png
    │   │       ├── 13.png
    │   │       ├── 14.png
    │   │       ├── 15.png
    │   │       ├── 16.png
    │   │       ├── 17.png
    │   │       ├── 18.png
    │   │       ├── 19.png
    │   │       ├── 2.png
    │   │       ├── 21.png
    │   │       ├── 22.png
    │   │       ├── 23.png
    │   │       ├── 24.png
    │   │       ├── 25.png
    │   │       ├── 26.png
    │   │       ├── 27.png
    │   │       ├── 28.png
    │   │       ├── 29.png
    │   │       ├── 31.png
    │   │       ├── 32.png
    │   │       ├── 34.png
    │   │       ├── 35.png
    │   │       ├── 36.png
    │   │       ├── 37.png
    │   │       ├── 38.png
    │   │       ├── 39.png
    │   │       ├── 40.png
    │   │       ├── 41.png
    │   │       ├── 42.png
    │   │       ├── 43.png
    │   │       ├── 47.png
    │   │       ├── 48.png
    │   │       ├── 49.png
    │   │       ├── 50.png
    │   │       ├── 52.png
    │   │       ├── 53.png
    │   │       ├── 54.png
    │   │       ├── 55.png
    │   │       ├── 56.png
    │   │       ├── 58.png
    │   │       ├── 59.png
    │   │       ├── 60.png
    │   │       ├── 61.png
    │   │       ├── 63.png
    │   │       ├── 65.png
    │   │       ├── 66.png
    │   │       ├── 68.png
    │   │       ├── 69.png
    │   │       ├── 7.png
    │   │       ├── 70.png
    │   │       ├── 71.png
    │   │       ├── 72.png
    │   │       ├── 73.png
    │   │       ├── 75.png
    │   │       ├── 76.png
    │   │       ├── 77.png
    │   │       ├── 78.png
    │   │       ├── 79.png
    │   │       ├── 8.png
    │   │       ├── 80.png
    │   │       ├── 81.png
    │   │       ├── 82.png
    │   │       ├── 83.png
    │   │       ├── 9.png
    │   │       ├── 90.png
    │   │       ├── 91.png
    │   │       ├── 92.png
    │   │       ├── 94.png
    │   │       ├── 96.png
    │   │       ├── 97.png
    │   │       └── 98.png
    │   ├── val/
    │   │   ├── images/
    │   │   │   ├── 1.jpg
    │   │   │   ├── 117.jpg
    │   │   │   ├── 121.jpg
    │   │   │   ├── 130.jpg
    │   │   │   ├── 20.jpg
    │   │   │   ├── 3.jpg
    │   │   │   ├── 33.jpg
    │   │   │   ├── 4.jpg
    │   │   │   ├── 46.jpg
    │   │   │   ├── 51.jpg
    │   │   │   ├── 57.jpg
    │   │   │   ├── 62.jpg
    │   │   │   ├── 64.jpg
    │   │   │   ├── 74.jpg
    │   │   │   ├── 84.jpg
    │   │   │   ├── 86.jpg
    │   │   │   ├── 87.jpg
    │   │   │   ├── 89.jpg
    │   │   │   └── 95.jpg
    │   │   └── masks/
    │   │       ├── 1.png
    │   │       ├── 117.png
    │   │       ├── 121.png
    │   │       ├── 130.png
    │   │       ├── 20.png
    │   │       ├── 3.png
    │   │       ├── 33.png
    │   │       ├── 4.png
    │   │       ├── 46.png
    │   │       ├── 51.png
    │   │       ├── 57.png
    │   │       ├── 62.png
    │   │       ├── 64.png
    │   │       ├── 74.png
    │   │       ├── 84.png
    │   │       ├── 86.png
    │   │       ├── 87.png
    │   │       ├── 89.png
    │   │       └── 95.png
    │   ├── dummy_mask.png
    │   ├── examples
    │   └── README.md
    ├── docker/
    │   ├── __pycache__/
    │   ├── health_check/
    │   │   ├── analytics/
    │   │   │   ├── __init__.py
    │   │   │   ├── dashboard_generator.py
    │   │   │   ├── metrics_collector.py
    │   │   │   └── recommendation_engine.py
    │   │   ├── checkers/
    │   │   │   ├── __init__.py
    │   │   │   ├── dependency_validator.py
    │   │   │   ├── docker_checker.py
    │   │   │   └── endpoint_checker.py
    │   │   ├── cli/
    │   │   │   ├── __init__.py
    │   │   │   └── commands.py
    │   │   ├── models/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── config.py
    │   │   │   ├── enums.py
    │   │   │   └── results.py
    │   │   ├── orchestration/
    │   │   │   ├── __init__.py
    │   │   │   ├── health_orchestrator.py
    │   │   │   ├── monitoring.py
    │   │   │   └── service_registry.py
    │   │   ├── persistence/
    │   │   │   ├── __init__.py
    │   │   │   └── report_saver.py
    │   │   └── __init__.py
    │   ├── scripts/
    │   │   ├── artifact-manager.sh
    │   │   ├── browser-manager.sh
    │   │   ├── ci-setup.sh
    │   │   ├── docker-stack-manager.sh
    │   │   ├── e2e-test-orchestrator.sh
    │   │   ├── health-check-manager.sh
    │   │   ├── manage-grid.sh
    │   │   ├── network-manager.sh
    │   │   ├── run-e2e-tests.sh
    │   │   ├── run-test-runner.sh
    │   │   ├── setup-env.sh
    │   │   ├── setup-local-dev.sh
    │   │   ├── start-test-env.sh
    │   │   └── system-monitor.sh
    │   ├── browser-capabilities.json
    │   ├── docker-compose.README.md
    │   ├── docker-compose.test.yml
    │   ├── docker-entrypoint.sh
    │   ├── Dockerfile.streamlit
    │   ├── Dockerfile.test-runner
    │   ├── env-test.yml
    │   ├── env.local.template
    │   ├── env.production.template
    │   ├── env.staging.template
    │   ├── env.test.template
    │   ├── env_manager.py
    │   ├── grid-config.json
    │   ├── health_check_system.py
    │   ├── health_check_system.py
    │   ├── mobile-browser-config.json
    │   ├── pytest.ini
    │   ├── README-ARCHITECTURE.md
    │   ├── README-DOCKER-TESTING.md
    │   ├── README-LOCAL-DEV.md
    │   ├── README-TROUBLESHOOTING.md
    │   ├── README-USAGE.md
    │   ├── README.artifact-management.md
    │   ├── README.cross-browser-testing.md
    │   ├── README.environment-management.md
    │   ├── README.md
    │   ├── README.network-setup.md
    │   ├── selenium-grid-guide.md
    │   ├── setup-local-dev.sh
    │   └── test-runner.config
    ├── docs/
    │   ├── api/
    │   │   ├── gui_components.md
    │   │   ├── gui_pages.md
    │   │   ├── gui_services.md
    │   │   └── utilities.md
    │   ├── designs/
    │   │   ├── logo.png
    │   │   └── loss_registry_design.md
    │   ├── guides/
    │   │   ├── architectural_decisions.md
    │   │   ├── checkpoint_format_specification.md
    │   │   ├── ci_cd_integration_guide.md
    │   │   ├── ci_cd_stakeholder_training.md
    │   │   ├── ci_cd_testing_integration.md
    │   │   ├── CLEAN_INSTALLATION.md
    │   │   ├── comprehensive_integration_test_reporting_guide.md
    │   │   ├── configuration_storage_specification.md
    │   │   ├── continuous_coverage_monitoring_guide.md
    │   │   ├── CONTRIBUTING.md
    │   │   ├── DEVELOPMENT.md
    │   │   ├── gui_development_guidelines.md
    │   │   ├── gui_testing_best_practices.md
    │   │   ├── gui_testing_implementation_checklist.md
    │   │   ├── INSTALL.md
    │   │   ├── loss_registry_usage.md
    │   │   ├── migration_summary_graphviz_to_matplotlib.md
    │   │   ├── performance_benchmarking_system.md
    │   │   ├── quality_gates_guide.md
    │   │   ├── SYSTEM_DEPENDENCIES.md
    │   │   ├── TECHNICAL_ARCHITECTURE.md
    │   │   ├── TEST_EXECUTION_PLAN.md
    │   │   ├── test_maintenance_procedures.md
    │   │   ├── TROUBLESHOOTING.md
    │   │   ├── USAGE.md
    │   │   └── WORKFLOW_TRAINING.md
    │   ├── reports/
    │   │   ├── analysis/
    │   │   │   ├── consolidation-implementation-summary.md
    │   │   │   ├── duplication-mapping.md
    │   │   │   ├── final-rule-cleanup-summary.md
    │   │   │   ├── rule-consolidation-report.md
    │   │   │   └── rule-system-analysis.md
    │   │   ├── archive/
    │   │   ├── coverage/
    │   │   │   ├── coverage_gaps_analysis.md
    │   │   │   ├── coverage_validation_report.md
    │   │   │   ├── test_coverage_analysis_report.md
    │   │   │   └── test_coverage_comparison_report.md
    │   │   ├── models/
    │   │   │   ├── model_expected_structure.json
    │   │   │   ├── model_imports_catalog.json
    │   │   │   ├── model_inventory.json
    │   │   │   ├── model_pyfiles.json
    │   │   │   └── model_structure_diff.json
    │   │   ├── project/
    │   │   │   └── plan_verificacion_post_linting.md
    │   │   ├── scripts/
    │   │   │   ├── example_prd.txt
    │   │   │   ├── hydra_examples.txt
    │   │   │   └── README.md
    │   │   ├── tasks/
    │   │   ├── testing/
    │   │   │   ├── next_testing_priorities.md
    │   │   │   ├── test_coverage_improvement_plan.md
    │   │   │   └── test_inventory.txt
    │   │   ├── automated_test_execution_report.md
    │   │   ├── basedpyright_analysis_report.md
    │   │   ├── crackseg_paper.md
    │   │   ├── crackseg_paper_es.md
    │   │   ├── documentation_checklist.md
    │   │   ├── gui_corrections_inventory.md
    │   │   ├── gui_test_coverage_analysis.md
    │   │   ├── project_tree.md
    │   │   ├── pytorch_cuda_compatibility_issue.md
    │   │   ├── README.md
    │   │   ├── task_4_final_integration_report.md
    │   │   ├── technical_report.md
    │   │   ├── tensorboard_component_refactoring_summary.md
    │   │   └── test_fixes_validation_report.md
    │   ├── stylesheets/
    │   │   └── extra.css
    │   ├── testing/
    │   │   ├── artifact_testing_plan.md
    │   │   └── test_patterns_and_best_practices.md
    │   ├── tools/
    │   │   └── task-master-guide.md
    │   ├── tutorials/
    │   │   ├── 01_basic_training.md
    │   │   ├── 02_custom_experiment.md
    │   │   └── 03_extending_project.md
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
    │   │   ├── config_editor/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── editor_core.py
    │   │   │   ├── file_browser_integration.py
    │   │   │   └── validation_panel.py
    │   │   ├── gallery/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── actions.py
    │   │   │   ├── event_handlers.py
    │   │   │   ├── renderer.py
    │   │   │   └── state_manager.py
    │   │   ├── tensorboard/
    │   │   │   ├── __pycache__/
    │   │   │   ├── recovery/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── recovery_strategies.py
    │   │   │   ├── rendering/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── diagnostics/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── action_controls.py
    │   │   │   │   │   └── diagnostic_panel.py
    │   │   │   │   ├── status_cards/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── base_card.py
    │   │   │   │   │   ├── health_card.py
    │   │   │   │   │   ├── network_card.py
    │   │   │   │   │   ├── process_card.py
    │   │   │   │   │   └── resource_card.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── advanced_status_renderer.py
    │   │   │   │   ├── control_renderer.py
    │   │   │   │   ├── error_renderer.py
    │   │   │   │   ├── iframe_renderer.py
    │   │   │   │   ├── startup_renderer.py
    │   │   │   │   └── status_renderer.py
    │   │   │   ├── state/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── progress_tracker.py
    │   │   │   │   └── session_manager.py
    │   │   │   ├── utils/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── formatters.py
    │   │   │   │   └── validators.py
    │   │   │   ├── __init__.py
    │   │   │   └── component.py
    │   │   ├── __init__.py
    │   │   ├── auto_save_manager.py
    │   │   ├── config_editor_component.py
    │   │   ├── confirmation_dialog.py
    │   │   ├── confirmation_renderer.py
    │   │   ├── confirmation_utils.py
    │   │   ├── device_detector.py
    │   │   ├── device_info.py
    │   │   ├── device_selector.py
    │   │   ├── device_selector_backup.py
    │   │   ├── device_selector_new.py
    │   │   ├── device_selector_ui.py
    │   │   ├── error_console.py
    │   │   ├── file_browser.py
    │   │   ├── file_browser_component.py
    │   │   ├── file_upload_component.py
    │   │   ├── header_component.py
    │   │   ├── loading_spinner.py
    │   │   ├── loading_spinner_optimized.py
    │   │   ├── log_viewer.py
    │   │   ├── logo_component.py
    │   │   ├── metrics_viewer.py
    │   │   ├── page_router.py
    │   │   ├── progress_bar.py
    │   │   ├── progress_bar_optimized.py
    │   │   ├── results_display.py
    │   │   ├── results_gallery_component.py
    │   │   ├── sidebar_component.py
    │   │   ├── tensorboard_component.py
    │   │   └── theme_component.py
    │   ├── docs/
    │   │   ├── error_messaging_system.md
    │   │   ├── file_upload_guide.md
    │   │   └── tensorboard_integration_summary.md
    │   ├── pages/
    │   │   ├── __pycache__/
    │   │   ├── architecture/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── config_section.py
    │   │   │   ├── info_section.py
    │   │   │   ├── model_section.py
    │   │   │   ├── utils.py
    │   │   │   └── visualization_section.py
    │   │   ├── results/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── comparison_section.py
    │   │   │   ├── config_section.py
    │   │   │   ├── gallery_section.py
    │   │   │   ├── metrics_section.py
    │   │   │   ├── setup_section.py
    │   │   │   ├── tensorboard_section.py
    │   │   │   └── utils.py
    │   │   ├── __init__.py
    │   │   ├── advanced_config_page.py
    │   │   ├── architecture_page.py
    │   │   ├── config_page.py
    │   │   ├── home_page.py
    │   │   ├── page_train.py
    │   │   ├── results_page.py
    │   │   ├── results_page_new.py
    │   │   └── train_page.py
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
    │   │   ├── parsing/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── exceptions.py
    │   │   │   └── override_parser.py
    │   │   ├── process/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── abort_system.py
    │   │   │   ├── core.py
    │   │   │   ├── error_handling.py
    │   │   │   ├── log_integration.py
    │   │   │   ├── manager_backup.py
    │   │   │   ├── monitoring.py
    │   │   │   ├── override_parser.py
    │   │   │   └── states.py
    │   │   ├── reports/
    │   │   │   └── models.py
    │   │   ├── results/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── advanced_validation.py
    │   │   │   ├── cache.py
    │   │   │   ├── core.py
    │   │   │   ├── demo.py
    │   │   │   ├── demo_advanced_validation.py
    │   │   │   ├── demo_reactive.py
    │   │   │   ├── demo_streamlit_integration.py
    │   │   │   ├── events.py
    │   │   │   ├── results_validator.py
    │   │   │   ├── scanner.py
    │   │   │   └── validation.py
    │   │   ├── results_scanning/
    │   │   │   └── __init__.py
    │   │   ├── run_manager/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── abort_api.py
    │   │   │   ├── orchestrator.py
    │   │   │   ├── session_api.py
    │   │   │   ├── status_integration.py
    │   │   │   ├── status_updates.py
    │   │   │   ├── streaming_api.py
    │   │   │   ├── ui_integration.py
    │   │   │   └── ui_status_helpers.py
    │   │   ├── streaming/
    │   │   │   ├── __pycache__/
    │   │   │   ├── sources/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── file_watcher.py
    │   │   │   │   └── stdout_reader.py
    │   │   │   ├── __init__.py
    │   │   │   ├── core.py
    │   │   │   └── exceptions.py
    │   │   ├── tensorboard/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── core.py
    │   │   │   ├── demo_refactored.py
    │   │   │   ├── lifecycle_manager.py
    │   │   │   ├── manager.py
    │   │   │   ├── port_management.py
    │   │   │   └── process_manager.py
    │   │   ├── threading/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── cancellation.py
    │   │   │   ├── coordinator.py
    │   │   │   ├── progress_tracking.py
    │   │   │   ├── task_results.py
    │   │   │   ├── task_status.py
    │   │   │   ├── ui_responsive_backup.py
    │   │   │   └── ui_wrapper.py
    │   │   ├── __init__.py
    │   │   ├── architecture_viewer.py
    │   │   ├── auto_save.py
    │   │   ├── config_io.py
    │   │   ├── data_stats.py
    │   │   ├── demo_tensorboard.py
    │   │   ├── error_state.py
    │   │   ├── export_manager.py
    │   │   ├── gui_config.py
    │   │   ├── log_parser.py
    │   │   ├── override_examples.py
    │   │   ├── performance_optimizer.py
    │   │   ├── process_manager.py
    │   │   ├── save_dialog.py
    │   │   ├── session_state.py
    │   │   ├── session_sync.py
    │   │   ├── streaming_examples.py
    │   │   ├── styling.py
    │   │   ├── tb_manager.py
    │   │   ├── theme.py
    │   │   └── training_state.py
    │   ├── __init__.py
    │   ├── app.py
    │   ├── debug_page_rendering.py
    │   └── README.md
    ├── logs/
    ├── scripts/
    │   ├── __pycache__/
    │   ├── archive/
    │   │   ├── limpieza_scripts_summary.md
    │   │   └── README.md
    │   ├── debug/
    │   │   ├── __init__.py
    │   │   ├── artifact_diagnostics.py
    │   │   ├── artifact_fixer.py
    │   │   ├── checkpoint_validator.py
    │   │   ├── main.py
    │   │   └── utils.py
    │   ├── examples/
    │   │   ├── factory_registry_integration.py
    │   │   └── tensorboard_port_management_demo.py
    │   ├── experiments/
    │   │   ├── benchmark_aspp.py
    │   │   ├── debug_swin_params.py
    │   │   ├── hybrid_registry_demo.py
    │   │   ├── registry_demo.py
    │   │   ├── test_pipeline_e2e.py
    │   │   ├── test_swin_encoder.py
    │   │   ├── test_swin_transfer_learning_script.py
    │   │   └── test_swin_unet.py
    │   ├── monitoring/
    │   │   └── continuous_coverage.py
    │   ├── performance/
    │   │   ├── __init__.py
    │   │   ├── base_executor.py
    │   │   ├── baseline_updater.py
    │   │   ├── cleanup_validator.py
    │   │   ├── health_checker.py
    │   │   ├── maintenance_manager.py
    │   │   └── utils.py
    │   ├── reports/
    │   │   ├── autofix_backups/
    │   │   ├── compare_model_structure.py
    │   │   ├── model_imports_autofix.py
    │   │   ├── model_imports_catalog.py
    │   │   ├── model_imports_cycles.py
    │   │   └── model_imports_validation.py
    │   ├── utils/
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
    │   │   ├── audit_rules_checklist.py
    │   │   ├── check_updates.py
    │   │   ├── clean_workspace.py
    │   │   ├── example_override.py
    │   │   ├── generate_project_tree.py
    │   │   ├── inventory_training_imports.py
    │   │   ├── model_summary.py
    │   │   ├── organize_reports.py
    │   │   ├── organize_reports_plan.md
    │   │   ├── reorganize_legacy_folders_plan.md
    │   │   ├── unet_diagram.py
    │   │   ├── validate-rule-references.py
    │   │   └── verify_setup.py
    │   ├── __init__.py
    │   ├── benchmark_tests.py
    │   ├── check_test_files.py
    │   ├── coverage_check.sh
    │   ├── debug_artifacts.py
    │   ├── performance_maintenance.py
    │   ├── README.md
    │   ├── run_tests_phased.py
    │   ├── simple_install_check.sh
    │   └── validate_test_quality.py
    ├── src/
    │   ├── crackseg/
    │   │   ├── __pycache__/
    │   │   ├── data/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── dataloader.py
    │   │   │   ├── dataset.py
    │   │   │   ├── distributed.py
    │   │   │   ├── factory.py
    │   │   │   ├── memory.py
    │   │   │   ├── README.md
    │   │   │   ├── sampler.py
    │   │   │   ├── splitting.py
    │   │   │   ├── transforms.py
    │   │   │   └── validation.py
    │   │   ├── evaluation/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── __main__.py
    │   │   │   ├── core.py
    │   │   │   ├── data.py
    │   │   │   ├── ensemble.py
    │   │   │   ├── loading.py
    │   │   │   ├── README.md
    │   │   │   ├── results.py
    │   │   │   └── setup.py
    │   │   ├── model/
    │   │   │   ├── __pycache__/
    │   │   │   ├── architectures/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── cnn_convlstm_unet.py
    │   │   │   │   └── swinv2_cnn_aspp_unet.py
    │   │   │   ├── base/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── abstract.py
    │   │   │   ├── bottleneck/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── cnn_bottleneck.py
    │   │   │   ├── common/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── visualization/
    │   │   │   │   │   ├── matplotlib/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── graphviz_renderer.py
    │   │   │   │   │   ├── main.py
    │   │   │   │   │   └── matplotlib_renderer.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── spatial_utils.py
    │   │   │   │   └── utils.py
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
    │   │   ├── outputs/
    │   │   │   ├── experiments/
    │   │   │   │   ├── 20250602-041102-basic_verification/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   ├── error_log.txt
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   ├── 20250602-041159-basic_verification/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   ├── error_log.txt
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   ├── 20250602-041325-basic_verification/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   ├── error_log.txt
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   ├── 20250602-041432-basic_verification/
    │   │   │   │   │   ├── checkpoints/
    │   │   │   │   │   ├── configurations/
    │   │   │   │   │   ├── logs/
    │   │   │   │   │   ├── metrics/
    │   │   │   │   │   ├── results/
    │   │   │   │   │   ├── config.json
    │   │   │   │   │   └── experiment_info.json
    │   │   │   │   └── 20250603-010344-basic_verification/
    │   │   │   │       ├── checkpoints/
    │   │   │   │       ├── configurations/
    │   │   │   │       ├── logs/
    │   │   │   │       ├── metrics/
    │   │   │   │       ├── results/
    │   │   │   │       ├── config.json
    │   │   │   │       └── experiment_info.json
    │   │   │   ├── shared/
    │   │   │   └── experiment_registry.json
    │   │   ├── training/
    │   │   │   ├── __pycache__/
    │   │   │   ├── losses/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── combinators/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── base_combinator.py
    │   │   │   │   │   ├── enhanced_product.py
    │   │   │   │   │   ├── enhanced_weighted_sum.py
    │   │   │   │   │   ├── product.py
    │   │   │   │   │   └── weighted_sum.py
    │   │   │   │   ├── factory/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── config_parser.py
    │   │   │   │   │   ├── config_validator.py
    │   │   │   │   │   └── recursive_factory.py
    │   │   │   │   ├── interfaces/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   └── loss_interface.py
    │   │   │   │   ├── registry/
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
    │   │   │   │   ├── focal_loss.py
    │   │   │   │   ├── loss_registry_setup.py
    │   │   │   │   └── recursive_factory.py
    │   │   │   ├── __init__.py
    │   │   │   ├── batch_processing.py
    │   │   │   ├── config_validation.py
    │   │   │   ├── factory.py
    │   │   │   ├── metrics.py
    │   │   │   ├── README.md
    │   │   │   └── trainer.py
    │   │   ├── utils/
    │   │   │   ├── __pycache__/
    │   │   │   ├── checkpointing/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── core.py
    │   │   │   │   ├── helpers.py
    │   │   │   │   └── setup.py
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
    │   │   │   ├── experiment/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── experiment.py
    │   │   │   │   └── manager.py
    │   │   │   ├── factory/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── cache.py
    │   │   │   │   └── factory.py
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
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── alert_types.py
    │   │   │   │   ├── alerting_system.py
    │   │   │   │   ├── callbacks.py
    │   │   │   │   ├── coverage_monitor.py
    │   │   │   │   ├── exceptions.py
    │   │   │   │   ├── gpu_callbacks.py
    │   │   │   │   ├── manager.py
    │   │   │   │   ├── resource_monitor.py
    │   │   │   │   ├── resource_snapshot.py
    │   │   │   │   ├── retention.py
    │   │   │   │   ├── system_callbacks.py
    │   │   │   │   ├── threshold_checker.py
    │   │   │   │   └── threshold_config.py
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
    │   │   │   ├── component_cache.py
    │   │   │   ├── exceptions.py
    │   │   │   ├── README.md
    │   │   │   └── torchvision_compat.py
    │   │   ├── __init__.py
    │   │   ├── __main__.py
    │   │   ├── dataclasses.py
    │   │   ├── evaluate.py
    │   │   └── README.md
    │   ├── crackseg.egg-info/
    │   └── main.py
    ├── tests/
    │   ├── __pycache__/
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
    │   │   │   ├── factories/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── base.py
    │   │   │   │   ├── config_factory.py
    │   │   │   │   ├── coordinator.py
    │   │   │   │   ├── image_factory.py
    │   │   │   │   └── model_factory.py
    │   │   │   ├── provisioning/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── core.py
    │   │   │   │   ├── database.py
    │   │   │   │   └── suites.py
    │   │   │   ├── __init__.py
    │   │   │   └── isolation.py
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
    │   │   ├── enhanced_gui_testing_demo.py
    │   │   └── visual_regression_demo.py
    │   ├── fixtures/
    │   │   └── mocks/
    │   │       ├── experiment_manager/
    │   │       │   ├── mock.experiment_manager.experiment_manager.experiment_dir/
    │   │       │   │   ├── 2254896334992/
    │   │       │   │   ├── 2254896338784/
    │   │       │   │   ├── 2254920579888/
    │   │       │   │   ├── 2254920693232/
    │   │       │   │   ├── 2254921007024/
    │   │       │   │   ├── 2254921628848/
    │   │       │   │   ├── 2254921775968/
    │   │       │   │   ├── 2254921841792/
    │   │       │   │   ├── 2254921913536/
    │   │       │   │   ├── 2254922611312/
    │   │       │   │   ├── 2256368976960/
    │   │       │   │   ├── 2399501198368/
    │   │       │   │   ├── 2399501899136/
    │   │       │   │   ├── 2399501909216/
    │   │       │   │   ├── 2399502072480/
    │   │       │   │   ├── 2399502175984/
    │   │       │   │   ├── 2399502348320/
    │   │       │   │   ├── 2399502532384/
    │   │       │   │   ├── 2399503130960/
    │   │       │   │   ├── 2399503481312/
    │   │       │   │   ├── 2399503869728/
    │   │       │   │   ├── 2789862072848/
    │   │       │   │   ├── 2789862077936/
    │   │       │   │   ├── 2789862548272/
    │   │       │   │   ├── 2789864181296/
    │   │       │   │   ├── 2789864185664/
    │   │       │   │   ├── 2789864403744/
    │   │       │   │   ├── 2789864459584/
    │   │       │   │   ├── 2789864648544/
    │   │       │   │   ├── 2789864714032/
    │   │       │   │   └── 2789865142272/
    │   │       │   └── mock.experiment_manager.experiment_manager.experiment_dir.__truediv__()/
    │   │       │       ├── 2254896338256/
    │   │       │       ├── 2254896346752/
    │   │       │       ├── 2254920643024/
    │   │       │       ├── 2254920935152/
    │   │       │       ├── 2254921165824/
    │   │       │       ├── 2254921294208/
    │   │       │       ├── 2254921768912/
    │   │       │       ├── 2254922315728/
    │   │       │       ├── 2254922618864/
    │   │       │       ├── 2256368910512/
    │   │       │       └── 2256371459392/
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
    │   │   │   ├── __pycache__/
    │   │   │   ├── test_data_loading_pipeline.py
    │   │   │   └── test_data_pipeline.py
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
    │   │   └── test_backward_compatibility.py
    │   ├── tools/
    │   │   ├── analysis/
    │   │   │   ├── __pycache__/
    │   │   │   ├── comprehensive_failure_analysis.py
    │   │   │   ├── test_failure_analysis.py
    │   │   │   ├── test_failure_categorization.py
    │   │   │   └── test_priority_matrix_creator.py
    │   │   ├── testing/
    │   │   │   ├── __pycache__/
    │   │   │   └── test_config_system.py
    │   │   └── utilities/
    │   │       └── temp_storage.py
    │   ├── unit/
    │   │   ├── data/
    │   │   │   ├── test_dataloader.py
    │   │   │   ├── test_dataset_pipeline.py
    │   │   │   ├── test_distributed.py
    │   │   │   ├── test_factory.py
    │   │   │   ├── test_memory.py
    │   │   │   └── test_sampler.py
    │   │   ├── docker/
    │   │   │   ├── __pycache__/
    │   │   │   ├── test_env_manager.py
    │   │   │   └── test_health_check_system.py
    │   │   ├── evaluation/
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
    │   │   ├── model/
    │   │   │   ├── config/
    │   │   │   │   └── test_instantiation.py
    │   │   │   ├── decoder/
    │   │   │   │   ├── test_channel_utils.py
    │   │   │   │   ├── test_cnn_decoder_channel_handling.py
    │   │   │   │   ├── test_cnn_decoder_error_handling.py
    │   │   │   │   ├── test_cnn_decoder_forward_pass.py
    │   │   │   │   ├── test_cnn_decoder_initialization.py
    │   │   │   │   └── test_cnn_decoder_special_features.py
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
    │   │   │   ├── test_swin_transformer_encoder.py
    │   │   │   ├── test_thread_safety.py
    │   │   │   ├── test_unet.py
    │   │   │   └── test_utils.py
    │   │   ├── training/
    │   │   │   ├── losses/
    │   │   │   │   ├── test_clean_factory.py
    │   │   │   │   ├── test_config_parser.py
    │   │   │   │   ├── test_enhanced_combinators.py
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
    │   │   │   ├── config/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   └── test_standardized_storage.py
    │   │   │   ├── logging/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   └── test_metrics_manager.py
    │   │   │   ├── monitoring/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── test_callbacks.py
    │   │   │   │   ├── test_monitoring_manager.py
    │   │   │   │   └── test_retention.py
    │   │   │   ├── test_checkpointing.py
    │   │   │   ├── test_dataset.py
    │   │   │   ├── test_early_stopping.py
    │   │   │   ├── test_env.py
    │   │   │   ├── test_logging.py
    │   │   │   ├── test_override.py
    │   │   │   ├── test_schema.py
    │   │   │   ├── test_splitting.py
    │   │   │   └── test_validation.py
    │   │   ├── test_main_data.py
    │   │   ├── test_main_environment.py
    │   │   ├── test_main_integration.py
    │   │   ├── test_main_model.py
    │   │   └── test_main_training.py
    │   ├── utils/
    │   │   ├── unified_testing/
    │   │   │   ├── __init__.py
    │   │   │   ├── core.py
    │   │   │   ├── helpers.py
    │   │   │   ├── mocking.py
    │   │   │   ├── performance.py
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
    ├── CHANGELOG.md
    ├── codecov.yml
    ├── environment.yml
    ├── mkdocs.yml
    ├── pyproject.toml
    ├── pyrightconfig.json
    ├── README.md
    ├── requirements.txt
    └── run.py
```
