"""Unit tests for capture storage functionality."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from tests.e2e.capture.storage import (
    CaptureStorage,
    CaptureStorageMixin,
    NamingConvention,
    RetentionPolicy,
    StorageConfig,
)


class TestStorageConfig:
    """Test suite for StorageConfig."""

    def test_default_configuration(self) -> None:
        """Test default storage configuration values."""
        config = StorageConfig()

        assert config.base_dir == Path("test-artifacts")
        assert config.screenshots_dir == "screenshots"
        assert config.videos_dir == "videos"
        assert config.regression_dir == "visual-regression"
        assert config.retention_policy == RetentionPolicy.KEEP_FAILURES
        assert config.retention_days == 7
        assert config.max_files_per_test == 10
        assert config.naming_convention == NamingConvention.DESCRIPTIVE

    def test_custom_configuration(self) -> None:
        """Test custom storage configuration."""
        custom_dir = Path("/tmp/custom-artifacts")
        config = StorageConfig(
            base_dir=custom_dir,
            retention_policy=RetentionPolicy.KEEP_ALL,
            max_files_per_test=5,
            naming_convention=NamingConvention.TIMESTAMP,
        )

        assert config.base_dir == custom_dir
        assert config.retention_policy == RetentionPolicy.KEEP_ALL
        assert config.max_files_per_test == 5
        assert config.naming_convention == NamingConvention.TIMESTAMP


class TestCaptureStorage:
    """Test suite for CaptureStorage."""

    @pytest.fixture
    def temp_dir(self) -> Generator[Path, None, None]:
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def storage_config(self, temp_dir: Path) -> StorageConfig:
        """Create test storage configuration."""
        return StorageConfig(
            base_dir=temp_dir,
            auto_cleanup=False,  # Disable auto cleanup for testing
        )

    @pytest.fixture
    def storage(self, storage_config: StorageConfig) -> CaptureStorage:
        """Create CaptureStorage instance for testing."""
        return CaptureStorage(storage_config)

    def test_initialization(
        self, storage: CaptureStorage, temp_dir: Path
    ) -> None:
        """Test storage initialization creates directories."""
        expected_dirs = [
            temp_dir / "screenshots",
            temp_dir / "videos",
            temp_dir / "visual-regression",
            temp_dir / "temp",
        ]

        for directory in expected_dirs:
            assert directory.exists()
            assert directory.is_dir()

    def test_store_artifact_creates_file(
        self, storage: CaptureStorage, temp_dir: Path
    ) -> None:
        """Test storing an artifact creates file in correct location."""
        # Create a temporary source file
        source_file = temp_dir / "test_source.png"
        source_file.write_text("test content")

        # Store the artifact
        stored_path = storage.store_artifact(
            source_file, "screenshot", "test_example"
        )

        # Verify file was stored correctly
        assert stored_path.exists()
        assert stored_path.parent.name == "screenshots"
        assert "test_example" in stored_path.name
        assert stored_path.read_text() == "test content"

    def test_store_artifact_with_metadata(
        self, storage: CaptureStorage, temp_dir: Path
    ) -> None:
        """Test storing artifact with metadata."""
        source_file = temp_dir / "test_source.png"
        source_file.write_text("test content")

        metadata = {"test_failed": True, "browser": "chrome"}

        stored_path = storage.store_artifact(
            source_file, "screenshot", "test_failure", metadata
        )

        # Check that metadata file was created
        meta_file = stored_path.with_suffix(stored_path.suffix + ".meta")
        assert meta_file.exists()

    def test_nonexistent_file_raises_error(
        self, storage: CaptureStorage
    ) -> None:
        """Test that storing nonexistent file raises FileNotFoundError."""
        nonexistent_file = Path("/nonexistent/file.png")

        with pytest.raises(FileNotFoundError, match="Source file not found"):
            storage.store_artifact(nonexistent_file, "screenshot", "test")

    def test_get_artifact_info(
        self, storage: CaptureStorage, temp_dir: Path
    ) -> None:
        """Test getting artifact information for a test."""
        # Create and store some test files
        for i in range(3):
            source_file = temp_dir / f"source_{i}.png"
            source_file.write_text(f"content {i}")
            storage.store_artifact(source_file, "screenshot", "test_info")

        info = storage.get_artifact_info("test_info")

        assert info["test_name"] == "test_info"
        assert info["total_files"] >= 3
        assert info["total_size"] > 0
        assert "screenshot" in info["artifacts"]
        assert info["artifacts"]["screenshot"]["count"] >= 3

    def test_filename_generation_descriptive(
        self, storage: CaptureStorage
    ) -> None:
        """Test descriptive filename generation."""
        # Test successful scenario
        filename = storage._generate_filename(
            "test_login", "screenshot", ".png", None
        )
        assert "test_login" in filename
        assert "success" in filename
        assert filename.endswith(".png")

        # Test failure scenario
        filename = storage._generate_filename(
            "test_login", "screenshot", ".png", {"test_failed": True}
        )
        assert "test_login" in filename
        assert "failure" in filename
        assert filename.endswith(".png")

    def test_sanitize_filename(self, storage: CaptureStorage) -> None:
        """Test filename sanitization."""
        dirty_name = 'test<>:"/\\|?*name'
        clean_name = storage._sanitize_filename(dirty_name)

        assert "<" not in clean_name
        assert ">" not in clean_name
        assert ":" not in clean_name
        assert '"' not in clean_name
        assert "/" not in clean_name
        assert "\\" not in clean_name
        assert "|" not in clean_name
        assert "?" not in clean_name
        assert "*" not in clean_name
        assert "test" in clean_name
        assert "name" in clean_name

    def test_cleanup_artifacts(
        self, storage: CaptureStorage, temp_dir: Path
    ) -> None:
        """Test cleaning up artifacts."""
        # Create test files
        source_file = temp_dir / "source.png"
        source_file.write_text("content")

        stored_path = storage.store_artifact(
            source_file, "screenshot", "test_cleanup"
        )
        assert stored_path.exists()

        # Clean up with force
        cleaned_count = storage.cleanup_artifacts("test_cleanup", force=True)

        assert cleaned_count > 0


class TestCaptureStorageMixin:
    """Test suite for CaptureStorageMixin."""

    def test_mixin_initialization(self) -> None:
        """Test mixin initialization."""

        class TestClass(CaptureStorageMixin):
            pass

        # Create instance with temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StorageConfig(base_dir=Path(tmpdir), auto_cleanup=False)
            instance = TestClass(storage_config=config)

            assert hasattr(instance, "_capture_storage")
            assert isinstance(instance._capture_storage, CaptureStorage)

    def test_mixin_methods(self) -> None:
        """Test mixin exposes storage methods."""

        class TestClass(CaptureStorageMixin):
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            config = StorageConfig(base_dir=Path(tmpdir), auto_cleanup=False)
            instance = TestClass(storage_config=config)

            # Test that methods are available
            assert hasattr(instance, "store_artifact")
            assert hasattr(instance, "cleanup_test_artifacts")
            assert hasattr(instance, "get_artifact_info")

            # Test method calls work
            info = instance.get_artifact_info("test")
            assert isinstance(info, dict)
            assert info["test_name"] == "test"
