# Duplicate Scan Report

Total duplicate groups: 31

## Group (count=4)

```python
    def feature_info(self) -> list[dict[str, Any]]:
        """Information about output features for each stage.

        Returns:
            List of dictionaries, each containing:
                - 'channels': Number of output channels
                - 'reduction': Spatial reduction factor from input
                - 'stage': Stage index
        """
        return self.get_feature_info()
```

- function feature_info @ src/crackseg/model/architectures/cnn_convlstm_unet.py:222-231
- function feature_info @ src/crackseg/model/encoder/cnn_encoder.py:126-135
- function feature_info @ src/crackseg/model/encoder/cnn_encoder.py:261-270
- function feature_info @ src/crackseg/model/encoder/swin/core.py:310-319

## Group (count=3)

```python
    def _create_empty_plot(self, title: str) -> Figure:
        from crackseg.evaluation.visualization.utils.plot_utils import (
            create_empty_plot,
        )

        fig, _ = create_empty_plot(
            f"No data available for {title}", figsize=(8, 6)
        )
        return fig
```

- function _create_empty_plot @ src/crackseg/evaluation/visualization/legacy/learning_rate_analysis.py:167-175
- function _create_empty_plot @ src/crackseg/evaluation/visualization/legacy/parameter_analysis.py:250-258
- function _create_empty_plot @ src/crackseg/evaluation/visualization/legacy/training_curves.py:201-209

## Group (count=3)

```python
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate checksum for {file_path}: {e}")
```

- function _calculate_checksum @ src/crackseg/utils/artifact_manager/storage.py:36-46
- function _calculate_checksum @ src/crackseg/utils/artifact_manager/validation.py:20-30
- function _calculate_checksum @ src/crackseg/utils/artifact_manager/versioning.py:147-157

## Group (count=3)

```python
def validate_checkpoint_completeness(
    checkpoint_data: dict[str, Any], spec: CheckpointSpec | None = None
) -> tuple[bool, list[str]]:
    """Validate that checkpoint contains all required fields.

    Args:
        checkpoint_data: Checkpoint dictionary to validate
        spec: Specification defining required fields

    Returns:
```

- function validate_checkpoint_completeness @ src/crackseg/utils/checkpointing/load.py:197-217
- function validate_checkpoint_completeness @ src/crackseg/utils/checkpointing/save.py:276-296
- function validate_checkpoint_completeness @ src/crackseg/utils/checkpointing/validation.py:95-115

## Group (count=3)

```python
class TrainingProcessError(Exception):
    """Custom exception for training process errors.

    Raised when training subprocess management fails due to:
    - Process already running when starting new training
    - Invalid command construction
    - Working directory doesn't exist
    - Process termination failures
    - Override validation errors

```

- class TrainingProcessError @ gui/utils/process/manager/core/manager_backup_original.py:30-45
- class TrainingProcessError @ gui/utils/process/manager/core/process_manager.py:28-43
- class TrainingProcessError @ gui/utils/process/manager/core/states.py:13-28

## Group (count=3)

```python
def find_matching_files(
    image_dir: Path, mask_dir: Path
) -> list[tuple[Path, Path]]:
    """
    Finds corresponding image-mask file pairs.

    Args:
        image_dir: Images directory
        mask_dir: Masks directory

```

- function find_matching_files @ scripts/data_processing/image_processing/crop_crack_images.py:151-181
- function find_matching_files @ scripts/data_processing/image_processing/crop_crack_images_configurable.py:177-207
- function find_matching_files @ scripts/data_processing/image_processing/crop_py_crackdb_images.py:260-290

## Group (count=3)

```python
def load_gitignore_matcher(project_root: Path):
    if pathspec is None:
        return lambda _p: False
    gi = project_root / ".gitignore"
    if not gi.exists():
        return lambda _p: False
    from scripts.utils.common.io_utils import read_text  # noqa: E402

    spec = pathspec.PathSpec.from_lines(
        "gitwildmatch", read_text(gi).splitlines()
```

- function load_gitignore_matcher @ scripts/utils/quality/scan_artifacts_and_binaries.py:99-118
- function load_gitignore_matcher @ scripts/utils/quality/scan_duplicates_unused.py:62-81
- function load_gitignore_matcher @ scripts/utils/quality/scan_language_compliance.py:92-111

## Group (count=2)

```python
    def __init__(self, style_config: dict[str, Any]) -> None:
        """Initialize the parameter analyzer.

        Args:
            style_config: Configuration for plot styling
        """
        self.style_config = style_config
```

- function __init__ @ src/crackseg/evaluation/visualization/analysis/parameter.py:22-28
- function __init__ @ src/crackseg/evaluation/visualization/legacy/parameter_analysis.py:23-29

## Group (count=2)

```python
    def visualize_parameter_distributions(
        self, model_path: Path, save_path: Path | None = None
    ) -> Figure | PlotlyFigure:
        """Visualize parameter distributions from model checkpoint.

        Args:
            model_path: Path to model checkpoint
            save_path: Path to save the visualization

        Returns:
```

- function visualize_parameter_distributions @ src/crackseg/evaluation/visualization/analysis/parameter.py:30-60
- function visualize_parameter_distributions @ src/crackseg/evaluation/visualization/legacy/parameter_analysis.py:31-61

## Group (count=2)

```python
    def _extract_parameter_statistics(
        self, model_state: dict[str, torch.Tensor]
    ) -> dict[str, dict[str, float]]:
        """Extract statistical information from model parameters.

        Args:
            model_state: Model state dictionary

        Returns:
            Dictionary mapping parameter names to statistics
```

- function _extract_parameter_statistics @ src/crackseg/evaluation/visualization/analysis/parameter.py:62-88
- function _extract_parameter_statistics @ src/crackseg/evaluation/visualization/legacy/parameter_analysis.py:63-89

## Group (count=2)

```python
    def _get_default_style(self) -> dict[str, Any]:
        """Get default styling configuration."""
        return {
            "figure_size": (12, 8),
            "dpi": 300,
            "color_palette": "viridis",
            "grid_alpha": 0.3,
            "line_width": 2,
            "font_size": 12,
            "title_font_size": 14,
```

- function _get_default_style @ src/crackseg/evaluation/visualization/training/advanced.py:61-72
- function _get_default_style @ src/crackseg/evaluation/visualization/training/core.py:56-67

## Group (count=2)

```python
    def connect_artifact_manager(
        self, artifact_manager: ArtifactManager
    ) -> None:
        """Connect with ArtifactManager for visualization storage.

        Args:
            artifact_manager: ArtifactManager instance to use for saving
        """
        self.artifact_manager = artifact_manager
        logger.info(
```

- function connect_artifact_manager @ src/crackseg/evaluation/visualization/training/advanced.py:74-85
- function connect_artifact_manager @ src/crackseg/evaluation/visualization/training/core.py:69-80

## Group (count=2)

```python
class PathMapping:
    """Represents a mapping between old and new paths."""

    old_path: str
    new_path: str
    mapping_type: str  # 'import', 'config', 'docs', 'artifact', 'checkpoint'
    description: str
    deprecated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
```

- class PathMapping @ src/crackseg/utils/mapping_registry_types.py:10-18
- class PathMapping @ scripts/utils/automation/simple_mapping_registry.py:14-22

## Group (count=2)

```python
def _has_nested_field(config: DictConfig, field_path: str) -> bool:
    """Check if a nested field exists in configuration."""
    try:
        keys = field_path.split(".")
        current = config
        for key in keys:
            # For DictConfig objects, use get() method which returns None if key doesn't exist
            if hasattr(current, "get"):
                value = current.get(key)
                if value is None:
```

- function _has_nested_field @ src/crackseg/utils/config/standardized_storage.py:255-274
- function _has_nested_field @ src/crackseg/utils/config/standardized_storage_utils.py:102-119

## Group (count=2)

```python
        def report_progress(stage: str, message: str, percent: float) -> None:
            if callback:
                elapsed = time.time() - start_time
                progress = AbortProgress(
                    stage=stage,
                    message=message,
                    progress_percent=percent,
                    elapsed_time=elapsed,
                    estimated_remaining=(
                        max(0, timeout - elapsed) if percent < 100 else 0
```

- function report_progress @ gui/utils/process/manager/cleanup/abort_system.py:92-104
- function report_progress @ gui/utils/process/manager/core/manager_backup_original.py:300-312

## Group (count=2)

```python
    def _build_command(
        self,
        config_path: Path,
        config_name: str,
        overrides: list[str] | None = None,
    ) -> list[str]:
        """Build the training command safely.

        Args:
            config_path: Path to configuration directory
```

- function _build_command @ gui/utils/process/manager/core/core.py:308-346
- function _build_command @ gui/utils/process/manager/core/manager_backup_original.py:504-542

## Group (count=2)

```python
    def _terminate_gracefully(self, timeout: float) -> None:
        """Attempt graceful process termination.

        Args:
            timeout: Maximum time to wait for termination
        """
        if self._process is None:
            return

        try:
```

- function _terminate_gracefully @ gui/utils/process/manager/core/core.py:379-401
- function _terminate_gracefully @ gui/utils/process/manager/core/manager_backup_original.py:612-634

## Group (count=2)

```python
    def _force_kill(self) -> None:
        """Force kill the process and its children."""
        if self._process is None:
            return

        try:
            if os.name == "nt":
                # Windows: Terminate process tree
                self._process.terminate()
            else:
```

- function _force_kill @ gui/utils/process/manager/core/core.py:403-421
- function _force_kill @ gui/utils/process/manager/core/manager_backup_original.py:636-654

## Group (count=2)

```python
    def _cleanup(self) -> None:
        """Clean up resources after process completion."""
        if self._process:
            try:
                # Ensure pipes are closed
                if self._process.stdout:
                    self._process.stdout.close()
                if self._process.stderr:
                    self._process.stderr.close()
                if self._process.stdin:
```

- function _cleanup @ gui/utils/process/manager/core/core.py:423-437
- function _cleanup @ gui/utils/process/manager/core/manager_backup_original.py:656-670

## Group (count=2)

```python
    def process_info(self) -> ProcessInfo:
        """Get current process information (thread-safe)."""
        with self._lock:
            return ProcessInfo(
                pid=self._process_info.pid,
                command=self._process_info.command.copy(),
                start_time=self._process_info.start_time,
                state=self._process_info.state,
                return_code=self._process_info.return_code,
                error_message=self._process_info.error_message,
```

- function process_info @ gui/utils/process/manager/core/manager_backup_original.py:87-98
- function process_info @ gui/utils/process/manager/core/process_manager.py:83-94

