# Project Directory Structure (excluding .gitignore)

```txt
└── crackseg/
    ├── archive/
    │   ├── legacy_docs/
    │   │   ├── development-guide.mdc
    │   │   ├── project-structure.mdc
    │   │   ├── README.md
    │   │   └── structural-guide.mdc
    │   └── README.md
    ├── configs/
    │   ├── __pycache__/
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
    │   │   │   ├── unet_mock.yaml
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
    ├── docs/
    │   ├── designs/
    │   │   ├── logo.png
    │   │   └── loss_registry_design.md
    │   ├── guides/
    │   │   ├── checkpoint_format_specification.md
    │   │   ├── CLEAN_INSTALLATION.md
    │   │   ├── configuration_storage_specification.md
    │   │   ├── CONTRIBUTING.md
    │   │   ├── loss_registry_usage.md
    │   │   ├── SYSTEM_DEPENDENCIES.md
    │   │   └── WORKFLOW_TRAINING.md
    │   ├── reports/
    │   │   ├── analysis/
    │   │   │   ├── consolidation-implementation-summary.md
    │   │   │   ├── duplication-mapping.md
    │   │   │   ├── final-rule-cleanup-summary.md
    │   │   │   ├── rule-consolidation-report.md
    │   │   │   └── rule-system-analysis.md
    │   │   ├── archive/
    │   │   │   ├── stats_report_20250514_220750.txt
    │   │   │   └── stats_report_20250516_034210.txt
    │   │   ├── coverage/
    │   │   │   ├── coverage_gaps_analysis.md
    │   │   │   ├── coverage_validation_report.md
    │   │   │   ├── test_coverage_analysis_report.md
    │   │   │   └── test_coverage_comparison_report.md
    │   │   ├── models/
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
    │   │   ├── documentation_checklist.md
    │   │   ├── legacy_folders_reorganization_summary.md
    │   │   ├── organization_summary.md
    │   │   ├── project_tree.md
    │   │   ├── README.md
    │   │   └── reorganization_summary.md
    │   ├── testing/
    │   │   ├── artifact_testing_plan.md
    │   │   └── test_patterns_and_best_practices.md
    │   └── tools/
    ├── htmlcov/
    ├── outputs/
    ├── scripts/
    │   ├── __pycache__/
    │   ├── examples/
    │   │   └── factory_registry_integration.py
    │   ├── experiments/
    │   ├── gui/
    │   │   ├── assets/
    │   │   │   ├── __pycache__/
    │   │   │   ├── css/
    │   │   │   │   ├── components/
    │   │   │   │   │   ├── navigation.css
    │   │   │   │   │   └── README.md
    │   │   │   │   ├── global/
    │   │   │   │   │   ├── base.css
    │   │   │   │   │   └── README.md
    │   │   │   │   └── themes/
    │   │   │   │       └── README.md
    │   │   │   ├── fonts/
    │   │   │   │   └── primary/
    │   │   │   │       └── README.md
    │   │   │   ├── images/
    │   │   │   │   ├── backgrounds/
    │   │   │   │   │   └── README.md
    │   │   │   │   ├── icons/
    │   │   │   │   │   └── README.md
    │   │   │   │   ├── logos/
    │   │   │   │   │   ├── primary-logo.png
    │   │   │   │   │   └── README.md
    │   │   │   │   └── samples/
    │   │   │   │       └── README.md
    │   │   │   ├── js/
    │   │   │   │   └── components/
    │   │   │   │       └── README.md
    │   │   │   ├── manifest/
    │   │   │   │   ├── asset_registry.json
    │   │   │   │   └── optimization_config.json
    │   │   │   ├── init_assets.py
    │   │   │   ├── manager.py
    │   │   │   ├── README.md
    │   │   │   └── structure.md
    │   │   ├── components/
    │   │   │   ├── __pycache__/
    │   │   │   ├── config_editor/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── editor_core.py
    │   │   │   │   ├── file_browser_integration.py
    │   │   │   │   └── validation_panel.py
    │   │   │   ├── __init__.py
    │   │   │   ├── config_editor_component.py
    │   │   │   ├── file_browser_component.py
    │   │   │   ├── logo_component.py
    │   │   │   ├── page_router.py
    │   │   │   ├── sidebar_component.py
    │   │   │   └── theme_component.py
    │   │   ├── pages/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── advanced_config_page.py
    │   │   │   ├── architecture_page.py
    │   │   │   ├── config_page.py
    │   │   │   ├── results_page.py
    │   │   │   └── train_page.py
    │   │   ├── utils/
    │   │   │   ├── __pycache__/
    │   │   │   ├── config/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── validation/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   └── yaml_engine.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── cache.py
    │   │   │   │   ├── exceptions.py
    │   │   │   │   ├── formatters.py
    │   │   │   │   ├── io.py
    │   │   │   │   └── templates.py
    │   │   │   ├── __init__.py
    │   │   │   ├── config_io.py
    │   │   │   ├── gui_config.py
    │   │   │   ├── save_dialog.py
    │   │   │   ├── session_state.py
    │   │   │   └── theme.py
    │   │   ├── app.py
    │   │   ├── app_legacy.py
    │   │   └── README_REFACTORING.md
    │   ├── outputs/
    │   ├── reports/
    │   │   ├── autofix_backups/
    │   │   ├── compare_model_structure.py
    │   │   ├── model_imports_autofix.py
    │   │   ├── model_imports_catalog.py
    │   │   ├── model_imports_cycles.py
    │   │   ├── model_imports_validation.py
    │   │   └── model_pyfiles_inventory.py
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
    │   │   ├── update_test_imports.py
    │   │   ├── validate-rule-references.py
    │   │   └── verify_setup.py
    │   ├── __init__.py
    │   ├── check_test_files.py
    │   ├── debug_artifacts.py
    │   ├── model_inventory.py
    │   ├── README.md
    │   ├── test_clean_installation.py
    │   ├── validate_coverage.py
    │   ├── validate_test_quality.py
    │   ├── verify_python_compatibility.py
    │   └── verify_system_dependencies.py
    ├── src/
    │   ├── __pycache__/
    │   ├── data/
    │   │   ├── __pycache__/
    │   │   ├── __init__.py
    │   │   ├── dataloader.py
    │   │   ├── dataset.py
    │   │   ├── distributed.py
    │   │   ├── factory.py
    │   │   ├── memory.py
    │   │   ├── README.md
    │   │   ├── sampler.py
    │   │   ├── splitting.py
    │   │   ├── transforms.py
    │   │   └── validation.py
    │   ├── evaluation/
    │   │   ├── __pycache__/
    │   │   ├── __init__.py
    │   │   ├── __main__.py
    │   │   ├── core.py
    │   │   ├── data.py
    │   │   ├── ensemble.py
    │   │   ├── loading.py
    │   │   ├── README.md
    │   │   ├── results.py
    │   │   └── setup.py
    │   ├── integration/
    │   │   └── model/
    │   │       └── test_integration.py
    │   ├── model/
    │   │   ├── __pycache__/
    │   │   ├── architectures/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── cnn_convlstm_unet.py
    │   │   │   ├── swinv2_cnn_aspp_unet.py
    │   │   │   └── unet.py
    │   │   ├── base/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   └── abstract.py
    │   │   ├── bottleneck/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   └── cnn_bottleneck.py
    │   │   ├── common/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   └── utils.py
    │   │   ├── components/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── aspp.py
    │   │   │   ├── attention_decorator.py
    │   │   │   ├── cbam.py
    │   │   │   ├── convlstm.py
    │   │   │   └── registry_support.py
    │   │   ├── config/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── core.py
    │   │   │   ├── factory.py
    │   │   │   ├── instantiation.py
    │   │   │   ├── schemas.py
    │   │   │   └── validation.py
    │   │   ├── core/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   └── unet.py
    │   │   ├── decoder/
    │   │   │   ├── __pycache__/
    │   │   │   ├── common/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   └── channel_utils.py
    │   │   │   ├── __init__.py
    │   │   │   └── cnn_decoder.py
    │   │   ├── encoder/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── cnn_encoder.py
    │   │   │   ├── feature_info_utils.py
    │   │   │   ├── swin_transformer_encoder.py
    │   │   │   └── swin_v2_adapter.py
    │   │   ├── factory/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── config.py
    │   │   │   ├── config_schema.py
    │   │   │   ├── factory.py
    │   │   │   ├── factory_utils.py
    │   │   │   ├── hybrid_registry.py
    │   │   │   ├── registry.py
    │   │   │   └── registry_setup.py
    │   │   ├── __init__.py
    │   │   └── README.md
    │   ├── outputs/
    │   ├── training/
    │   │   ├── __pycache__/
    │   │   ├── losses/
    │   │   │   ├── __pycache__/
    │   │   │   ├── combinators/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── base_combinator.py
    │   │   │   │   ├── enhanced_product.py
    │   │   │   │   ├── enhanced_weighted_sum.py
    │   │   │   │   ├── product.py
    │   │   │   │   └── weighted_sum.py
    │   │   │   ├── factory/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── config_parser.py
    │   │   │   │   ├── config_validator.py
    │   │   │   │   └── recursive_factory.py
    │   │   │   ├── interfaces/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── loss_interface.py
    │   │   │   ├── registry/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── clean_registry.py
    │   │   │   │   ├── enhanced_registry.py
    │   │   │   │   └── setup_losses.py
    │   │   │   ├── __init__.py
    │   │   │   ├── base_loss.py
    │   │   │   ├── bce_dice_loss.py
    │   │   │   ├── bce_loss.py
    │   │   │   ├── combined_loss.py
    │   │   │   ├── dice_loss.py
    │   │   │   ├── focal_loss.py
    │   │   │   ├── loss_registry_setup.py
    │   │   │   └── recursive_factory.py
    │   │   ├── __init__.py
    │   │   ├── batch_processing.py
    │   │   ├── config_validation.py
    │   │   ├── factory.py
    │   │   ├── metrics.py
    │   │   ├── README.md
    │   │   └── trainer.py
    │   ├── utils/
    │   │   ├── __pycache__/
    │   │   ├── checkpointing/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── core.py
    │   │   │   ├── helpers.py
    │   │   │   └── setup.py
    │   │   ├── config/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── env.py
    │   │   │   ├── init.py
    │   │   │   ├── override.py
    │   │   │   ├── schema.py
    │   │   │   ├── standardized_storage.py
    │   │   │   └── validation.py
    │   │   ├── core/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── device.py
    │   │   │   ├── exceptions.py
    │   │   │   ├── paths.py
    │   │   │   └── seeds.py
    │   │   ├── experiment/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── experiment.py
    │   │   │   └── manager.py
    │   │   ├── factory/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── cache.py
    │   │   │   └── factory.py
    │   │   ├── logging/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── base.py
    │   │   │   ├── experiment.py
    │   │   │   ├── metrics_manager.py
    │   │   │   ├── setup.py
    │   │   │   └── training.py
    │   │   ├── training/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── amp_utils.py
    │   │   │   ├── early_stopping.py
    │   │   │   ├── early_stopping_setup.py
    │   │   │   └── scheduler_helper.py
    │   │   ├── visualization/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   └── plots.py
    │   │   ├── __init__.py
    │   │   ├── component_cache.py
    │   │   ├── exceptions.py
    │   │   └── README.md
    │   ├── __init__.py
    │   ├── __main__.py
    │   ├── evaluate.py
    │   ├── main.py
    │   └── README.md
    ├── tasks/
    ├── tests/
    │   ├── __pycache__/
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
    │   │   │   ├── test_config_editor_component.py
    │   │   │   ├── test_config_io.py
    │   │   │   ├── test_file_browser_component.py
    │   │   │   └── test_yaml_validation.py
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
    │   ├── unit/
    │   │   ├── __pycache__/
    │   │   ├── data/
    │   │   │   ├── __pycache__/
    │   │   │   ├── test_dataloader.py
    │   │   │   ├── test_dataset_pipeline.py
    │   │   │   ├── test_distributed.py
    │   │   │   ├── test_factory.py
    │   │   │   ├── test_memory.py
    │   │   │   └── test_sampler.py
    │   │   ├── evaluation/
    │   │   │   ├── __pycache__/
    │   │   │   ├── test_core.py
    │   │   │   ├── test_data.py
    │   │   │   ├── test_ensemble.py
    │   │   │   ├── test_evaluate.py
    │   │   │   ├── test_evaluation_main.py
    │   │   │   ├── test_loading.py
    │   │   │   └── test_results.py
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
    │   │   │   ├── __pycache__/
    │   │   │   ├── losses/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── test_clean_factory.py
    │   │   │   │   ├── test_config_parser.py
    │   │   │   │   ├── test_enhanced_combinators.py
    │   │   │   │   ├── test_isolated_clean_factory.py
    │   │   │   │   ├── test_loss_factory.py
    │   │   │   │   ├── test_loss_registry.py
    │   │   │   │   └── test_recursive_factory.py
    │   │   │   ├── test_losses.py
    │   │   │   ├── test_lr_scheduler_factory.py
    │   │   │   ├── test_metrics.py
    │   │   │   ├── test_reproducibility.py
    │   │   │   └── test_trainer.py
    │   │   ├── utils/
    │   │   │   ├── __pycache__/
    │   │   │   ├── config/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   └── test_standardized_storage.py
    │   │   │   ├── logging/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   └── test_metrics_manager.py
    │   │   │   ├── test_checkpointing.py
    │   │   │   ├── test_dataset.py
    │   │   │   ├── test_early_stopping.py
    │   │   │   ├── test_env.py
    │   │   │   ├── test_logging.py
    │   │   │   ├── test_override.py
    │   │   │   ├── test_schema.py
    │   │   │   ├── test_splitting.py
    │   │   │   ├── test_transforms.py
    │   │   │   └── test_validation.py
    │   │   └── test_main.py
    │   ├── __init__.py
    │   ├── conftest.py
    │   └── README.md
    ├── CHANGELOG.md
    ├── environment.yml
    ├── pyproject.toml
    ├── pyrightconfig.json
    ├── README.md
    ├── requirements.txt
    └── run.py
```
