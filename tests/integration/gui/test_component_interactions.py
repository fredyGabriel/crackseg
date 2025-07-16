"""Component interaction integration tests for CrackSeg GUI.

This module implements integration tests for Subtask 7.4, verifying proper
interaction between GUI components, file operations, and data processing
modules with focus on end-to-end workflows.

Target workflows:
1. Results scanning to triplet display pipeline
2. Image selection UI to ZIP export workflow
3. Download button integration with file system operations
4. Error propagation between components
5. Session state synchronization across components
6. Performance optimization impact on component interactions
"""

from typing import Any

from .test_base import WorkflowTestBase
from .workflow_components import ComponentInteractionWorkflow


class TestComponentInteractions(WorkflowTestBase):
    """Integration tests for component interactions following Task 7.4 spec."""

    def setup_method(self) -> None:
        """Setup component interaction test environment."""
        super().setup_method()
        self.interaction_workflow = ComponentInteractionWorkflow(self)

        # Create test data directories
        self.results_dir = self.temp_path / "results"
        self.exports_dir = self.temp_path / "exports"
        self.results_dir.mkdir(exist_ok=True)
        self.exports_dir.mkdir(exist_ok=True)

        # Create sample triplet files for testing
        self._create_sample_triplet_files()

    def _create_sample_triplet_files(self) -> None:
        """Create sample triplet files (image, mask, prediction) for test."""
        # Create sample dataset structure
        dataset_dir = self.results_dir / "test_dataset"
        dataset_dir.mkdir(exist_ok=True)

        for i in range(3):
            triplet_id = f"crack_{i:03d}"

            # Create placeholder files
            (dataset_dir / f"{triplet_id}.jpg").write_text("sample_image")
            (dataset_dir / f"{triplet_id}_mask.png").write_text("sample_mask")
            (dataset_dir / f"{triplet_id}_pred.png").write_text("sample_pred")

    def test_results_scanning_to_triplet_display_pipeline(self) -> None:
        """Test pipeline from results scanning to triplet display component.

        Workflow: Directory scan → triplet discovery → display preparation
        """
        # Execute the scanning to display workflow
        pipeline_result = (
            self.interaction_workflow.execute_scanning_to_display_workflow(
                scan_directory=self.results_dir,
                validation_level="STANDARD",
                display_config={
                    "items_per_page": 12,
                    "enable_thumbnails": True,
                    "health_checking": True,
                },
            )
        )

        # Verify complete pipeline execution
        assert pipeline_result[
            "success"
        ], f"Pipeline failed: {pipeline_result}"
        assert pipeline_result["scan_completed"]
        assert pipeline_result["triplets_discovered"] > 0
        assert pipeline_result["display_component_initialized"]
        assert pipeline_result["session_state_updated"]

        # Verify triplet health checking
        assert "triplet_health_stats" in pipeline_result
        health_stats = pipeline_result["triplet_health_stats"]
        assert health_stats["total_triplets"] == 3
        assert health_stats["healthy_triplets"] >= 0

    def test_image_selection_ui_to_zip_export_workflow(self) -> None:
        """Test workflow from image selection UI to ZIP export completion.

        Workflow: Selection → validation → export preparation → ZIP creation
        """
        # Create triplet selection scenario
        triplet_selection = {
            "selected_ids": {"crack_001", "crack_002"},
            "include_images": True,
            "include_metadata": True,
            "export_scope": "Selected Items",
        }

        export_path = self.exports_dir / "test_export.zip"

        # Execute selection to export workflow
        export_result = (
            self.interaction_workflow.execute_selection_to_export_workflow(
                selection_config=triplet_selection,
                export_path=export_path,
                background_processing=True,
            )
        )

        # Verify export workflow success
        assert export_result[
            "success"
        ], f"Export workflow failed: {export_result}"
        assert export_result["selection_validated"]
        assert export_result["export_manager_initialized"]
        assert export_result["zip_creation_started"]
        assert export_result["progress_tracking_enabled"]

        # Verify file system integration
        if export_result["export_completed"]:
            assert export_path.exists()
            assert export_path.stat().st_size > 0

    def test_download_button_file_system_integration(self) -> None:
        """Test download button integration with file system operations.

        Workflow: Download trigger → file preparation → stream delivery
        """
        # Create test export file
        test_export = self.exports_dir / "download_test.zip"
        test_export.write_bytes(b"Sample ZIP content for download test")

        download_config = {
            "file_path": test_export,
            "download_name": "crackseg_results.zip",
            "mime_type": "application/zip",
            "cleanup_after": True,
        }

        # Execute download integration workflow
        download_result = (
            self.interaction_workflow.execute_download_integration_workflow(
                download_config
            )
        )

        # Verify download integration
        assert download_result["success"]
        assert download_result["file_validation_passed"]
        assert download_result["stream_preparation_completed"]
        assert download_result["download_button_functional"]

        # Verify file cleanup behavior
        if download_config["cleanup_after"]:
            assert download_result["cleanup_scheduled"]

    def test_error_propagation_between_components(self) -> None:
        """Test error propagation and handling across component boundaries.

        Scenarios: Scan errors → display errors → export errors
        """
        error_scenarios = [
            {
                "component": "scanner",
                "error_type": "DirectoryNotFound",
                "trigger": "invalid_scan_directory",
            },
            {
                "component": "display",
                "error_type": "TripletValidationError",
                "trigger": "corrupted_triplet_files",
            },
            {
                "component": "export",
                "error_type": "ExportPermissionError",
                "trigger": "readonly_export_directory",
            },
        ]

        # Execute error propagation testing
        error_results = (
            self.interaction_workflow.execute_error_propagation_workflow(
                error_scenarios
            )
        )

        # Verify error handling and propagation
        assert error_results["success"]
        assert error_results["all_errors_caught"]
        assert error_results["error_boundaries_respected"]
        assert error_results["user_feedback_provided"]

        # Verify specific error handling
        for scenario in error_scenarios:
            component = scenario["component"]
            assert error_results[f"{component}_error_handled"]
            assert error_results[f"{component}_graceful_degradation"]

    def test_session_state_synchronization_across_components(self) -> None:
        """Test session state synchronization across multiple GUI components.

        Workflow: State updates → cross-component synchronization → consistency
        """
        # Define multi-component state scenario
        state_scenario: dict[str, Any] = {
            "initial_state": {
                "scan_results": [],
                "selected_triplets": set(),
                "export_progress": 0.0,
                "validation_stats": {},
            },
            "state_updates": [
                {
                    "component": "scanner",
                    "key": "scan_results",
                    "value": "mock_triplets",
                },
                {
                    "component": "gallery",
                    "key": "selected_triplets",
                    "value": {"crack_001"},
                },
                {
                    "component": "export",
                    "key": "export_progress",
                    "value": 0.5,
                },
                {
                    "component": "validator",
                    "key": "validation_stats",
                    "value": {"total": 3},
                },
            ],
        }

        # Execute session state synchronization workflow
        sync_result = (
            self.interaction_workflow.execute_session_state_sync_workflow(
                state_scenario
            )
        )

        # Verify synchronization across components
        assert sync_result["success"]
        assert sync_result["state_consistency_maintained"]
        assert sync_result["cross_component_updates_propagated"]
        assert sync_result["no_state_conflicts_detected"]

        # Verify individual component state updates
        state_updates = state_scenario.get("state_updates", [])
        for update_dict in state_updates:
            component = update_dict.get("component", "")
            assert sync_result[f"{component}_state_updated"]

    def test_performance_optimization_component_interactions(self) -> None:
        """Test performance optimization impact on component interactions.

        Scenarios: Caching → lazy loading → async operations
        """
        performance_config = {
            "enable_caching": True,
            "lazy_loading": True,
            "async_scanning": True,
            "batch_processing": True,
            "memory_optimization": True,
        }

        # Execute performance optimization workflow
        workflow = self.interaction_workflow
        perf_result = workflow.execute_performance_optimization_workflow(
            performance_config, large_dataset_simulation=True
        )

        # Verify performance optimizations
        assert perf_result["success"]
        assert perf_result["caching_effective"]
        assert perf_result["lazy_loading_functional"]
        assert perf_result["async_operations_working"]
        assert perf_result["memory_usage_optimized"]

        # Verify performance metrics
        assert (
            perf_result["scan_time"] < 5.0
        )  # Should be fast with optimizations
        assert perf_result["memory_peak"] < 100  # MB, should be reasonable
        assert (
            perf_result["ui_responsiveness"] > 0.8
        )  # Should maintain responsiveness

    def test_complete_end_to_end_workflow_integration(self) -> None:
        """Test complete end-to-end workflow integration combining components.

        Complete flow: Scan → Display → Select → Export → Download
        """
        # Execute complete integration workflow
        e2e_result = (
            self.interaction_workflow.execute_complete_integration_workflow(
                scan_directory=self.results_dir,
                export_directory=self.exports_dir,
                workflow_config={
                    "validation_level": "COMPREHENSIVE",
                    "enable_performance_monitoring": True,
                    "error_recovery_enabled": True,
                    "progress_tracking": True,
                },
            )
        )

        # Verify complete integration
        assert e2e_result["success"], f"E2E workflow failed: {e2e_result}"
        assert e2e_result["all_phases_completed"]
        assert e2e_result["data_consistency_maintained"]
        assert e2e_result["no_critical_errors"]

        # Verify individual workflow phases
        phases = ["scan", "display", "selection", "export", "download"]
        for phase in phases:
            assert e2e_result[f"{phase}_phase_successful"]
            assert e2e_result[f"{phase}_performance_acceptable"]

    def test_concurrent_component_operations(self) -> None:
        """Test concurrent operations across multiple components.

        Scenarios: Parallel scan + export, concurrent state updates
        """
        concurrent_config = {
            "parallel_operations": ["scan", "export"],
            "max_concurrent_workers": 2,
            "resource_sharing": True,
            "deadlock_prevention": True,
        }

        # Execute concurrent operations workflow
        concurrent_result = (
            self.interaction_workflow.execute_concurrent_operations_workflow(
                concurrent_config
            )
        )

        # Verify concurrent operation handling
        assert concurrent_result["success"]
        assert concurrent_result["no_deadlocks_detected"]
        assert concurrent_result["resource_conflicts_resolved"]
        assert concurrent_result["data_integrity_maintained"]
        assert concurrent_result["performance_benefits_achieved"]

    def test_component_lifecycle_management(self) -> None:
        """Test component lifecycle management and cleanup.

        Workflow: Initialize → Active → Suspend → Resume → Cleanup
        """
        lifecycle_phases = [
            "initialize",
            "activate",
            "suspend",
            "resume",
            "cleanup",
        ]

        # Execute lifecycle management workflow
        lifecycle_result = (
            self.interaction_workflow.execute_lifecycle_management_workflow(
                lifecycle_phases
            )
        )

        # Verify lifecycle management
        assert lifecycle_result["success"]
        assert lifecycle_result["all_phases_executed"]
        assert lifecycle_result["resource_cleanup_completed"]
        assert lifecycle_result["no_memory_leaks_detected"]

        # Verify individual lifecycle phases
        for phase in lifecycle_phases:
            assert lifecycle_result[f"{phase}_phase_successful"]
