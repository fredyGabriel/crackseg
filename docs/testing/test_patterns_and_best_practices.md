# Test Patterns and Best Practices

## CrackSeg Project Testing Standards

**Established:** January 6, 2025
**Coverage Achievement:** 66% (from 25%)
**Test Suite Size:** 866 tests (67 unit + 11 integration + legacy tests)

---

## Core Testing Principles

### 1. Type Safety First

All test code must include comprehensive type annotations and pass `basedpyright` with zero errors.

```python
from typing import Any
from unittest.mock import Mock, patch
import pytest
import torch

def test_model_forward_pass(mock_model: Mock) -> None:
    """Test model forward pass with proper type annotations."""
    input_tensor: torch.Tensor = torch.randn(1, 3, 224, 224)
    expected_output: torch.Tensor = torch.randn(1, 1, 224, 224)

    mock_model.forward.return_value = expected_output
    result: torch.Tensor = mock_model(input_tensor)

    assert isinstance(result, torch.Tensor)
    assert result.shape == expected_output.shape
```

### 2. Quality Gates Integration

Every test file must pass all quality checks:

```bash
# Pre-commit quality gates
basedpyright tests/
ruff check tests/ --fix
black tests/
```

### 3. Modern Python Patterns

Use Python 3.12+ built-in generics and modern syntax:

```python
# ✅ Modern approach
def create_mock_dataloaders() -> dict[str, DataLoader[Any]]:
    return {
        "train": DataLoader(mock_dataset, batch_size=32),
        "val": DataLoader(mock_dataset, batch_size=32),
    }

# ❌ Avoid legacy typing
from typing import Dict, List
def create_mock_dataloaders() -> Dict[str, DataLoader]:
    pass
```

---

## Unit Testing Patterns

### Pattern 1: Component Isolation Testing

**Use Case:** Testing individual functions/classes in isolation
**Example:** Model component testing

```python
class TestCBAMComponent:
    """Test CBAM attention component in isolation."""

    @pytest.fixture
    def cbam_config(self) -> dict[str, Any]:
        """Provide minimal CBAM configuration."""
        return {
            "in_channels": 64,
            "reduction_ratio": 16,
            "kernel_size": 7
        }

    def test_cbam_forward_shape(self, cbam_config: dict[str, Any]) -> None:
        """Test CBAM output shape matches input."""
        cbam = CBAM(**cbam_config)
        input_tensor = torch.randn(2, 64, 32, 32)

        output = cbam(input_tensor)

        assert output.shape == input_tensor.shape
        assert isinstance(output, torch.Tensor)
```

**Key Principles:**

- Use fixtures for reusable test data
- Test one specific behavior per test method
- Include shape and type assertions for tensors
- Use descriptive test names that explain the behavior

### Pattern 2: Configuration-Driven Testing

**Use Case:** Testing components with various configurations
**Example:** Factory instantiation testing

```python
class TestModelInstantiation:
    """Test model instantiation with different configurations."""

    @pytest.fixture
    def base_config(self) -> DictConfig:
        """Provide base model configuration."""
        config_dict = {
            "encoder": {"name": "cnn_encoder", "channels": [64, 128, 256]},
            "decoder": {"name": "cnn_decoder", "channels": [256, 128, 64]},
            "bottleneck": {"name": "cnn_bottleneck", "channels": 512}
        }
        return OmegaConf.create(config_dict)

    @pytest.mark.parametrize("encoder_type,expected_class", [
        ("cnn_encoder", CNNEncoder),
        ("swin_encoder", SwinTransformerEncoder),
    ])
    def test_encoder_instantiation(
        self,
        base_config: DictConfig,
        encoder_type: str,
        expected_class: type
    ) -> None:
        """Test different encoder types instantiate correctly."""
        base_config.encoder.name = encoder_type

        with patch('src.model.factory.instantiate_encoder') as mock_instantiate:
            mock_instantiate.return_value = expected_class()

            result = create_unet(base_config)

            assert isinstance(result.encoder, expected_class)
            mock_instantiate.assert_called_once()
```

**Key Principles:**

- Use `@pytest.mark.parametrize` for testing multiple scenarios
- Leverage OmegaConf for configuration testing
- Mock external dependencies appropriately
- Test both success and failure scenarios

### Pattern 3: Error Handling Validation

**Use Case:** Testing error conditions and edge cases
**Example:** Input validation testing

```python
class TestInputValidation:
    """Test input validation and error handling."""

    def test_invalid_tensor_dimensions_raises_error(self) -> None:
        """Test that invalid input dimensions raise appropriate error."""
        model = UNetModel(in_channels=3, out_channels=1)
        invalid_input = torch.randn(2, 4, 224, 224)  # Wrong channels

        with pytest.raises(ValueError, match="Expected 3 channels, got 4"):
            model(invalid_input)

    def test_empty_configuration_raises_error(self) -> None:
        """Test that empty configuration raises ConfigError."""
        empty_config = OmegaConf.create({})

        with pytest.raises(ConfigError, match="Missing required"):
            create_unet(empty_config)
```

**Key Principles:**

- Use `pytest.raises` with specific exception types
- Include regex patterns to validate error messages
- Test edge cases and boundary conditions
- Ensure error messages are helpful for debugging

---

## Integration Testing Patterns

### Pattern 4: Pipeline Integration Testing

**Use Case:** Testing component interactions and data flow
**Example:** Data pipeline integration

```python
class TestDataPipelineIntegration:
    """Test complete data loading pipeline integration."""

    @pytest.fixture
    def mock_dataset_components(self) -> dict[str, Mock]:
        """Create mock components for data pipeline."""
        return {
            "dataset": Mock(spec=CrackSegmentationDataset),
            "transform": Mock(),
            "dataloader": Mock(spec=DataLoader)
        }

    def test_complete_data_pipeline_flow(
        self,
        mock_dataset_components: dict[str, Mock]
    ) -> None:
        """Test complete data pipeline from config to dataloader."""
        # Arrange
        config = self._create_data_config()
        mock_dataset = mock_dataset_components["dataset"]
        mock_dataset.__len__.return_value = 100

        # Act
        with patch('src.data.factory.create_dataset') as mock_create:
            mock_create.return_value = mock_dataset
            result = create_dataloaders_from_config(config)

        # Assert
        assert "train" in result
        assert "val" in result
        mock_create.assert_called()
```

**Key Principles:**

- Test complete workflows, not just individual components
- Use comprehensive mocking for external dependencies
- Verify both data flow and component interactions
- Include realistic data sizes and configurations

### Pattern 5: Factory Integration Testing

**Use Case:** Testing factory systems and component creation
**Example:** Model factory integration

```python
class TestModelFactoryIntegration:
    """Test model factory integration with training components."""

    def test_factory_to_training_integration(self) -> None:
        """Test model created by factory integrates with training."""
        # Arrange
        config = self._create_model_config()

        # Act - Create model through factory
        model = create_unet(config)

        # Assert - Model works with training components
        assert isinstance(model, nn.Module)

        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)

        assert output.shape == (1, 1, 224, 224)
        assert output.requires_grad  # Trainable

        # Test with loss function
        target = torch.randn(1, 1, 224, 224)
        loss_fn = nn.MSELoss()
        loss = loss_fn(output, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
```

**Key Principles:**

- Test end-to-end component integration
- Verify compatibility with downstream systems
- Include realistic usage scenarios
- Test both creation and usage patterns

---

## Mock and Fixture Patterns

### Pattern 6: Reusable Mock Components

**Use Case:** Creating consistent mocks across test suites
**Example:** Model component mocks

```python
# conftest.py - Shared fixtures
@pytest.fixture
def mock_encoder() -> Mock:
    """Create a mock encoder with realistic behavior."""
    mock = Mock(spec=EncoderBase)
    mock.forward.return_value = torch.randn(1, 256, 32, 32)
    mock.get_feature_info.return_value = [
        {"channels": 64, "reduction": 4},
        {"channels": 128, "reduction": 8},
        {"channels": 256, "reduction": 16}
    ]
    return mock

@pytest.fixture
def mock_training_components() -> TrainingComponents:
    """Create mock training components for testing."""
    return TrainingComponents(
        model=Mock(spec=nn.Module),
        optimizer=Mock(spec=torch.optim.Optimizer),
        scheduler=Mock(),
        loss_fn=Mock(),
        device=torch.device("cpu")
    )
```

**Key Principles:**

- Use `spec` parameter to maintain interface contracts
- Provide realistic return values for mocked methods
- Share common fixtures via `conftest.py`
- Document fixture behavior and usage

### Pattern 7: Temporary Data Creation

**Use Case:** Creating test data that's automatically cleaned up
**Example:** File system testing

```python
@pytest.fixture
def temp_data_structure(tmp_path: Path) -> str:
    """Create temporary data directory structure."""
    data_root = tmp_path / "data"

    # Create directory structure
    for split in ["train", "val", "test"]:
        images_dir = data_root / split / "images"
        masks_dir = data_root / split / "masks"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        # Create sample files
        for i in range(3):
            create_sample_image(images_dir / f"img{i}.png")
            create_sample_mask(masks_dir / f"img{i}.png")

    return str(data_root)

def create_sample_image(path: Path) -> None:
    """Create a sample image file for testing."""
    image = Image.new("RGB", (64, 64), color="red")
    image.save(path)
```

**Key Principles:**

- Use `tmp_path` fixture for automatic cleanup
- Create realistic file structures
- Generate minimal but valid test data
- Document data structure and content

---

## Performance and Memory Testing

### Pattern 8: Performance Validation

**Use Case:** Ensuring tests don't introduce performance regressions
**Example:** Memory usage testing

```python
import psutil
import time

class TestPerformanceValidation:
    """Test performance characteristics of components."""

    def test_data_loading_performance(self) -> None:
        """Test data loading meets performance requirements."""
        config = self._create_performance_config()

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        # Execute data loading
        dataloaders = create_dataloaders_from_config(config)
        batch = next(iter(dataloaders["train"]))

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss

        # Performance assertions
        execution_time = end_time - start_time
        memory_usage = (end_memory - start_memory) / 1024 / 1024  # MB

        assert execution_time < 5.0, f"Data loading too slow: {execution_time}s"
        assert memory_usage < 100, f"Memory usage too high: {memory_usage}MB"
```

**Key Principles:**

- Set realistic performance thresholds
- Measure both time and memory usage
- Use appropriate tools for measurement
- Document performance requirements

---

## Test Organization and Structure

### Directory Structure

```text
tests/
├── unit/                    # Unit tests
│   ├── data/               # Data pipeline tests
│   ├── model/              # Model component tests
│   ├── training/           # Training system tests
│   └── utils/              # Utility function tests
├── integration/            # Integration tests
│   ├── data/               # Data pipeline integration
│   ├── model/              # Model factory integration
│   ├── training/           # Training workflow integration
│   └── end_to_end/         # Complete pipeline tests
├── conftest.py             # Shared fixtures
└── pytest.ini             # Test configuration
```

### Naming Conventions

- **Test Files:** `test_<module_name>.py`
- **Test Classes:** `Test<ComponentName>` or `Test<Functionality>`
- **Test Methods:** `test_<behavior_description>`
- **Fixtures:** `<component_name>_<type>` (e.g., `model_config`, `mock_encoder`)

### Test Documentation

```python
def test_model_forward_pass_with_attention(self) -> None:
    """Test model forward pass with attention mechanism enabled.

    This test verifies that:
    1. Model processes input tensors correctly
    2. Attention weights are computed and applied
    3. Output shape matches expected dimensions
    4. Gradients flow properly for training
    """
    # Test implementation
```

---

## Continuous Integration Patterns

### Quality Gates Configuration

```yaml
# .github/workflows/test.yml
- name: Run Quality Gates
  run: |
    basedpyright tests/
    ruff check tests/ --fix
    black tests/ --check

- name: Run Test Suite
  run: |
    pytest tests/ --cov=src --cov-report=xml --cov-report=html

- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

### Coverage Requirements

- **Minimum Coverage:** 66% (current baseline)
- **Target Coverage:** 85% (long-term goal)
- **Critical Modules:** >80% coverage required
- **New Code:** 100% coverage required

---

## Common Anti-Patterns to Avoid

### ❌ Avoid These Patterns

1. **Testing Implementation Details**

    ```python
    # ❌ Bad - testing internal implementation
    def test_model_uses_specific_activation(self) -> None:
        model = UNetModel()
        assert isinstance(model._encoder._layers[0], nn.ReLU)
    ```

2. **Overly Complex Test Setup**

    ```python
    # ❌ Bad - complex setup that's hard to understand
    def test_complex_scenario(self) -> None:
        # 50 lines of setup code
        # Unclear what's being tested
    ```

3. **Non-Deterministic Tests**

    ```python
    # ❌ Bad - random behavior that can cause flaky tests
    def test_random_behavior(self) -> None:
        result = some_function_with_randomness()
        assert result > 0  # May fail randomly
    ```

4. **Testing Multiple Behaviors**

    ```python
    # ❌ Bad - testing multiple things in one test
    def test_model_everything(self) -> None:
        # Tests instantiation, forward pass, training, saving, loading...
    ```

### ✅ Preferred Patterns

1. **Testing Behavior, Not Implementation**

    ```python
    # ✅ Good - testing observable behavior
    def test_model_produces_correct_output_shape(self) -> None:
        model = UNetModel(in_channels=3, out_channels=1)
        input_tensor = torch.randn(1, 3, 224, 224)
        output = model(input_tensor)
        assert output.shape == (1, 1, 224, 224)
    ```

2. **Clear, Focused Tests**

    ```python
    # ✅ Good - clear setup and single responsibility
    def test_encoder_feature_extraction(self) -> None:
        encoder = CNNEncoder(channels=[64, 128, 256])
        input_tensor = torch.randn(1, 3, 224, 224)
        features = encoder(input_tensor)
        assert len(features) == 3  # Three feature levels
    ```

---

## Future Testing Roadmap

### Short-Term Goals (Next Sprint)

1. Address 97 failing tests to achieve >95% success rate
2. Implement main entry point testing (`src/main.py`, `src/evaluate.py`)
3. Expand configuration system testing

### Medium-Term Goals (Next Quarter)

1. Achieve 85% overall coverage
2. Implement performance regression testing
3. Add end-to-end pipeline testing

### Long-Term Goals (Next 6 Months)

1. Implement property-based testing for critical algorithms
2. Add mutation testing for test quality validation
3. Establish automated test generation for new components

---

**Documentation Maintained by:** CrackSeg Development Team
**Last Updated:** January 6, 2025
**Next Review:** After achieving 85% coverage target
