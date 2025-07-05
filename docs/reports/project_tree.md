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
    ├── archived-artifacts/
    ├── checkpoints/
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
    ├── crackseg.egg-info/
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
    │   ├── api/
    │   │   ├── gui_components.md
    │   │   ├── gui_services.md
    │   │   └── utilities.md
    │   ├── designs/
    │   │   ├── logo.png
    │   │   └── loss_registry_design.md
    │   ├── guides/
    │   │   ├── checkpoint_format_specification.md
    │   │   ├── ci_cd_integration_guide.md
    │   │   ├── CLEAN_INSTALLATION.md
    │   │   ├── comprehensive_integration_test_reporting_guide.md
    │   │   ├── configuration_storage_specification.md
    │   │   ├── CONTRIBUTING.md
    │   │   ├── DEVELOPMENT.md
    │   │   ├── INSTALL.md
    │   │   ├── loss_registry_usage.md
    │   │   ├── SYSTEM_DEPENDENCIES.md
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
    │   │   ├── reorganization_summary.md
    │   │   └── tensorboard_component_refactoring_summary.md
    │   ├── stylesheets/
    │   │   └── extra.css
    │   ├── testing/
    │   │   ├── artifact_testing_plan.md
    │   │   └── test_patterns_and_best_practices.md
    │   ├── tools/
    │   └── index.md
    ├── generated_configs/
    ├── htmlcov/
    ├── outputs/
    ├── results/
    ├── scripts/
    │   ├── __pycache__/
    │   ├── examples/
    │   │   ├── factory_registry_integration.py
    │   │   └── tensorboard_port_management_demo.py
    │   ├── experiments/
    │   ├── gui/
    │   │   ├── __pycache__/
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
    │   │   │   ├── gallery/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── actions.py
    │   │   │   │   ├── event_handlers.py
    │   │   │   │   ├── renderer.py
    │   │   │   │   └── state_manager.py
    │   │   │   ├── tensorboard/
    │   │   │   ├── __init__.py
    │   │   │   ├── auto_save_manager.py
    │   │   │   ├── config_editor_component.py
    │   │   │   ├── confirmation_dialog.py
    │   │   │   ├── confirmation_renderer.py
    │   │   │   ├── confirmation_utils.py
    │   │   │   ├── device_selector.py
    │   │   │   ├── error_console.py
    │   │   │   ├── file_browser_component.py
    │   │   │   ├── file_upload_component.py
    │   │   │   ├── loading_spinner.py
    │   │   │   ├── loading_spinner_optimized.py
    │   │   │   ├── logo_component.py
    │   │   │   ├── page_router.py
    │   │   │   ├── progress_bar.py
    │   │   │   ├── progress_bar_optimized.py
    │   │   │   ├── results_display.py
    │   │   │   ├── results_gallery_component.py
    │   │   │   ├── sidebar_component.py
    │   │   │   ├── tensorboard_component.py
    │   │   │   └── theme_component.py
    │   │   ├── docs/
    │   │   │   ├── error_messaging_system.md
    │   │   │   ├── file_upload_guide.md
    │   │   │   └── tensorboard_integration_summary.md
    │   │   ├── pages/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── advanced_config_page.py
    │   │   │   ├── architecture_page.py
    │   │   │   ├── config_page.py
    │   │   │   ├── page_train.py
    │   │   │   ├── results_page.py
    │   │   │   └── train_page.py
    │   │   ├── services/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── gallery_export_service.py
    │   │   │   └── gallery_scanner_service.py
    │   │   ├── utils/
    │   │   │   ├── __pycache__/
    │   │   │   ├── config/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── validation/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── error_categorizer.py
    │   │   │   │   │   └── yaml_engine.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── cache.py
    │   │   │   │   ├── exceptions.py
    │   │   │   │   ├── formatters.py
    │   │   │   │   ├── io.py
    │   │   │   │   └── templates.py
    │   │   │   ├── parsing/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── exceptions.py
    │   │   │   │   └── override_parser.py
    │   │   │   ├── process/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── abort_system.py
    │   │   │   │   ├── core.py
    │   │   │   │   ├── error_handling.py
    │   │   │   │   ├── log_integration.py
    │   │   │   │   ├── manager_backup.py
    │   │   │   │   ├── monitoring.py
    │   │   │   │   ├── override_parser.py
    │   │   │   │   └── states.py
    │   │   │   ├── reports/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   └── models.py
    │   │   │   ├── results/
    │   │   │   ├── results_scanning/
    │   │   │   │   └── __init__.py
    │   │   │   ├── run_manager/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── abort_api.py
    │   │   │   │   ├── orchestrator.py
    │   │   │   │   ├── session_api.py
    │   │   │   │   ├── status_integration.py
    │   │   │   │   ├── status_updates.py
    │   │   │   │   ├── streaming_api.py
    │   │   │   │   ├── ui_integration.py
    │   │   │   │   └── ui_status_helpers.py
    │   │   │   ├── streaming/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── sources/
    │   │   │   │   │   ├── __pycache__/
    │   │   │   │   │   ├── __init__.py
    │   │   │   │   │   ├── file_watcher.py
    │   │   │   │   │   └── stdout_reader.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── core.py
    │   │   │   │   └── exceptions.py
    │   │   │   ├── tensorboard/
    │   │   │   ├── threading/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── cancellation.py
    │   │   │   │   ├── coordinator.py
    │   │   │   │   ├── progress_tracking.py
    │   │   │   │   ├── task_results.py
    │   │   │   │   ├── task_status.py
    │   │   │   │   ├── ui_responsive_backup.py
    │   │   │   │   └── ui_wrapper.py
    │   │   │   ├── __init__.py
    │   │   │   ├── architecture_viewer.py
    │   │   │   ├── auto_save.py
    │   │   │   ├── config_io.py
    │   │   │   ├── demo_tensorboard.py
    │   │   │   ├── error_state.py
    │   │   │   ├── export_manager.py
    │   │   │   ├── gui_config.py
    │   │   │   ├── override_examples.py
    │   │   │   ├── performance_optimizer.py
    │   │   │   ├── save_dialog.py
    │   │   │   ├── session_state.py
    │   │   │   ├── session_sync.py
    │   │   │   ├── streaming_examples.py
    │   │   │   ├── tb_manager.py
    │   │   │   └── theme.py
    │   │   ├── __init__.py
    │   │   ├── app.py
    │   │   ├── app_legacy.py
    │   │   ├── README.md
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
    │   │   ├── check_file_sizes.py
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
    ├── selenium-videos/
    ├── site/
    │   ├── api/
    │   │   ├── gui_components/
    │   │   │   └── index.html
    │   │   ├── gui_services/
    │   │   │   └── index.html
    │   │   └── utilities/
    │   │       └── index.html
    │   ├── assets/
    │   │   ├── images/
    │   │   │   └── favicon.png
    │   │   ├── javascripts/
    │   │   │   ├── lunr/
    │   │   │   │   ├── min/
    │   │   │   │   │   ├── lunr.ar.min.js
    │   │   │   │   │   ├── lunr.da.min.js
    │   │   │   │   │   ├── lunr.de.min.js
    │   │   │   │   │   ├── lunr.du.min.js
    │   │   │   │   │   ├── lunr.el.min.js
    │   │   │   │   │   ├── lunr.es.min.js
    │   │   │   │   │   ├── lunr.fi.min.js
    │   │   │   │   │   ├── lunr.fr.min.js
    │   │   │   │   │   ├── lunr.he.min.js
    │   │   │   │   │   ├── lunr.hi.min.js
    │   │   │   │   │   ├── lunr.hu.min.js
    │   │   │   │   │   ├── lunr.hy.min.js
    │   │   │   │   │   ├── lunr.it.min.js
    │   │   │   │   │   ├── lunr.ja.min.js
    │   │   │   │   │   ├── lunr.jp.min.js
    │   │   │   │   │   ├── lunr.kn.min.js
    │   │   │   │   │   ├── lunr.ko.min.js
    │   │   │   │   │   ├── lunr.multi.min.js
    │   │   │   │   │   ├── lunr.nl.min.js
    │   │   │   │   │   ├── lunr.no.min.js
    │   │   │   │   │   ├── lunr.pt.min.js
    │   │   │   │   │   ├── lunr.ro.min.js
    │   │   │   │   │   ├── lunr.ru.min.js
    │   │   │   │   │   ├── lunr.sa.min.js
    │   │   │   │   │   ├── lunr.stemmer.support.min.js
    │   │   │   │   │   ├── lunr.sv.min.js
    │   │   │   │   │   ├── lunr.ta.min.js
    │   │   │   │   │   ├── lunr.te.min.js
    │   │   │   │   │   ├── lunr.th.min.js
    │   │   │   │   │   ├── lunr.tr.min.js
    │   │   │   │   │   ├── lunr.vi.min.js
    │   │   │   │   │   └── lunr.zh.min.js
    │   │   │   │   ├── tinyseg.js
    │   │   │   │   └── wordcut.js
    │   │   │   ├── workers/
    │   │   │   │   ├── search.d50fe291.min.js
    │   │   │   │   └── search.d50fe291.min.js.map
    │   │   │   ├── bundle.13a4f30d.min.js
    │   │   │   └── bundle.13a4f30d.min.js.map
    │   │   ├── stylesheets/
    │   │   │   ├── main.342714a4.min.css
    │   │   │   ├── main.342714a4.min.css.map
    │   │   │   ├── palette.06af60db.min.css
    │   │   │   └── palette.06af60db.min.css.map
    │   │   └── _mkdocstrings.css
    │   ├── CONTRIBUTING/
    │   ├── designs/
    │   │   ├── loss_registry_design/
    │   │   │   └── index.html
    │   │   └── logo.png
    │   ├── DEVELOPMENT/
    │   ├── guides/
    │   │   ├── checkpoint_format_specification/
    │   │   │   └── index.html
    │   │   ├── CLEAN_INSTALLATION/
    │   │   │   └── index.html
    │   │   ├── configuration_storage_specification/
    │   │   │   └── index.html
    │   │   ├── CONTRIBUTING/
    │   │   │   └── index.html
    │   │   ├── DEVELOPMENT/
    │   │   │   └── index.html
    │   │   ├── INSTALL/
    │   │   │   └── index.html
    │   │   ├── loss_registry_usage/
    │   │   │   └── index.html
    │   │   ├── SYSTEM_DEPENDENCIES/
    │   │   │   └── index.html
    │   │   ├── USAGE/
    │   │   │   └── index.html
    │   │   └── WORKFLOW_TRAINING/
    │   │       └── index.html
    │   ├── INSTALL/
    │   ├── reports/
    │   │   ├── analysis/
    │   │   │   ├── consolidation-implementation-summary/
    │   │   │   │   └── index.html
    │   │   │   ├── duplication-mapping/
    │   │   │   │   └── index.html
    │   │   │   ├── final-rule-cleanup-summary/
    │   │   │   │   └── index.html
    │   │   │   ├── rule-consolidation-report/
    │   │   │   │   └── index.html
    │   │   │   └── rule-system-analysis/
    │   │   │       └── index.html
    │   │   ├── archive/
    │   │   │   ├── stats_report_20250514_220750.txt
    │   │   │   └── stats_report_20250516_034210.txt
    │   │   ├── coverage/
    │   │   │   ├── coverage_gaps_analysis/
    │   │   │   │   └── index.html
    │   │   │   ├── coverage_validation_report/
    │   │   │   │   └── index.html
    │   │   │   ├── test_coverage_analysis_report/
    │   │   │   │   └── index.html
    │   │   │   └── test_coverage_comparison_report/
    │   │   │       └── index.html
    │   │   ├── documentation_checklist/
    │   │   │   └── index.html
    │   │   ├── legacy_folders_reorganization_summary/
    │   │   │   └── index.html
    │   │   ├── models/
    │   │   ├── organization_summary/
    │   │   │   └── index.html
    │   │   ├── project/
    │   │   │   └── plan_verificacion_post_linting/
    │   │   │       └── index.html
    │   │   ├── project_tree/
    │   │   │   └── index.html
    │   │   ├── reorganization_summary/
    │   │   │   └── index.html
    │   │   ├── scripts/
    │   │   │   ├── example_prd.txt
    │   │   │   ├── hydra_examples.txt
    │   │   │   └── index.html
    │   │   ├── tasks/
    │   │   ├── tensorboard_component_refactoring_summary/
    │   │   │   └── index.html
    │   │   ├── testing/
    │   │   │   ├── next_testing_priorities/
    │   │   │   │   └── index.html
    │   │   │   ├── test_coverage_improvement_plan/
    │   │   │   │   └── index.html
    │   │   │   └── test_inventory.txt
    │   │   └── index.html
    │   ├── search/
    │   │   └── search_index.json
    │   ├── stylesheets/
    │   │   └── extra.css
    │   ├── testing/
    │   │   ├── artifact_testing_plan/
    │   │   │   └── index.html
    │   │   └── test_patterns_and_best_practices/
    │   │       └── index.html
    │   ├── tools/
    │   │   └── task-master-guide/
    │   │       └── index.html
    │   ├── USAGE/
    │   ├── 404.html
    │   ├── index.html
    │   ├── objects.inv
    │   ├── sitemap.xml
    │   └── sitemap.xml.gz
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
    ├── test-artifacts/
    │   ├── e2e/
    │   └── screenshots/
    ├── test-results/
    ├── tests/
    │   ├── __pycache__/
    │   ├── docker/
    │   │   ├── __pycache__/
    │   │   ├── scripts/
    │   │   │   ├── artifact-manager.sh
    │   │   │   ├── browser-manager.sh
    │   │   │   ├── ci-setup.sh
    │   │   │   ├── docker-stack-manager.sh
    │   │   │   ├── e2e-test-orchestrator.sh
    │   │   │   ├── health-check-manager.sh
    │   │   │   ├── manage-grid.sh
    │   │   │   ├── network-manager.sh
    │   │   │   ├── run-e2e-tests.sh
    │   │   │   ├── run-test-runner.sh
    │   │   │   ├── setup-env.sh
    │   │   │   ├── setup-local-dev.sh
    │   │   │   ├── start-test-env.sh
    │   │   │   └── system-monitor.sh
    │   │   ├── browser-capabilities.json
    │   │   ├── docker-compose.README.md
    │   │   ├── docker-compose.test.yml
    │   │   ├── docker-entrypoint.sh
    │   │   ├── Dockerfile.streamlit
    │   │   ├── Dockerfile.test-runner
    │   │   ├── env-test.yml
    │   │   ├── env.local.template
    │   │   ├── env.production.template
    │   │   ├── env.staging.template
    │   │   ├── env.test.template
    │   │   ├── env_manager.py
    │   │   ├── grid-config.json
    │   │   ├── health_check_system.py
    │   │   ├── mobile-browser-config.json
    │   │   ├── pytest.ini
    │   │   ├── README-ARCHITECTURE.md
    │   │   ├── README-DOCKER-TESTING.md
    │   │   ├── README-LOCAL-DEV.md
    │   │   ├── README-TROUBLESHOOTING.md
    │   │   ├── README-USAGE.md
    │   │   ├── README.artifact-management.md
    │   │   ├── README.cross-browser-testing.md
    │   │   ├── README.environment-management.md
    │   │   ├── README.md
    │   │   ├── README.network-setup.md
    │   │   ├── selenium-grid-guide.md
    │   │   ├── setup-local-dev.sh
    │   │   └── test-runner.config
    │   ├── e2e/
    │   │   ├── __pycache__/
    │   │   ├── capture/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── screenshot.py
    │   │   │   ├── storage.py
    │   │   │   ├── video.py
    │   │   │   └── visual_regression.py
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
    │   │   │   ├── pytest_markers.py
    │   │   │   ├── resource_manager.py
    │   │   │   └── test_parallel_framework_validation.py
    │   │   ├── data/
    │   │   │   ├── __pycache__/
    │   │   │   ├── factories/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── base.py
    │   │   │   │   ├── config_factory.py
    │   │   │   │   ├── coordinator.py
    │   │   │   │   ├── image_factory.py
    │   │   │   │   └── model_factory.py
    │   │   │   ├── provisioning/
    │   │   │   │   ├── __pycache__/
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
    │   │   ├── reporting/
    │   │   │   ├── __pycache__/
    │   │   │   ├── analysis/
    │   │   │   │   ├── __pycache__/
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
    │   │   │   ├── streamlit.py
    │   │   │   ├── string.py
    │   │   │   └── time.py
    │   │   ├── waits/
    │   │   │   ├── __pycache__/
    │   │   │   ├── __init__.py
    │   │   │   ├── conditions.py
    │   │   │   └── strategies.py
    │   │   ├── __init__.py
    │   │   ├── base_test.py
    │   │   ├── conftest.py
    │   │   ├── README.md
    │   │   ├── test_driver_integration.py
    │   │   ├── test_fixture_usage_example.py
    │   │   └── test_streamlit_basic.py
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
    │   │   │   │   └── __pycache__/
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
    │   │   │   ├── test_concurrent_operations.py
    │   │   │   ├── test_config_editor_component.py
    │   │   │   ├── test_config_io.py
    │   │   │   ├── test_error_scenarios.py
    │   │   │   ├── test_file_browser_component.py
    │   │   │   ├── test_session_state_simple.py
    │   │   │   ├── test_session_state_verification.py
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
    │   │   ├── reporting/
    │   │   │   ├── __pycache__/
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
    │   │   ├── docker/
    │   │   │   ├── __pycache__/
    │   │   │   ├── test_artifact_manager.py
    │   │   │   ├── test_env_manager.py
    │   │   │   └── test_health_check_system.py
    │   │   ├── e2e/
    │   │   │   ├── __pycache__/
    │   │   │   ├── capture/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── test_screenshot.py
    │   │   │   │   └── test_storage.py
    │   │   │   ├── config/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── test_browser_capabilities.py
    │   │   │   │   └── test_browser_config_manager.py
    │   │   │   ├── waits/
    │   │   │   │   ├── __pycache__/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── test_conditions.py
    │   │   │   │   └── test_strategies.py
    │   │   │   ├── test_base_test.py
    │   │   │   ├── test_conftest_fixtures.py
    │   │   │   ├── test_helpers.py
    │   │   │   ├── test_performance_integration.py
    │   │   │   └── test_utils_basic.py
    │   │   ├── evaluation/
    │   │   │   ├── __pycache__/
    │   │   │   ├── test_core.py
    │   │   │   ├── test_data.py
    │   │   │   ├── test_ensemble.py
    │   │   │   ├── test_evaluate.py
    │   │   │   ├── test_evaluation_main.py
    │   │   │   ├── test_loading.py
    │   │   │   └── test_results.py
    │   │   ├── gui/
    │   │   │   ├── __pycache__/
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
    │   │   │   │   └── test_session_state.py
    │   │   │   ├── test_enhanced_abort.py
    │   │   │   ├── test_error_console.py
    │   │   │   ├── test_error_console_simple.py
    │   │   │   ├── test_file_upload.py
    │   │   │   ├── test_session_state_updates.py
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
    │   ├── README.md
    │   └── requirements-testing.txt
    ├── CHANGELOG.md
    ├── debug_parsing.py
    ├── environment.yml
    ├── gui_test_results.txt
    ├── gui_unit_test_results.txt
    ├── mkdocs.yml
    ├── pyproject.toml
    ├── pyrightconfig.json
    ├── README.md
    ├── requirements.txt
    ├── run.py
    ├── temp_storage.py
    └── test_results.txt
```
