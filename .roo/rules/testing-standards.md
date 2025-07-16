---
description:
globs: test*.py
alwaysApply: false
---
# Testing Quality Standards

These standards ensure tests validate real functionality while maintaining clean, maintainable production code. Tests should verify actual intended behavior without forcing artificial constraints on production systems.

## Core Testing Philosophy

- **Tests Validate Real APIs and Behavior:**
  - Test actual, intended behavior of production code - not implementation details
  - Focus on public APIs, inputs, outputs, and documented contracts
  - Verify business logic and functional requirements
  - Example:
    ```python
    # ✅ Test the actual API contract and business logic
    def test_model_prediction_shape():
        model = CrackSegmentationModel(num_classes=2)
        input_tensor = torch.randn(1, 3, 512, 512)
        output = model(input_tensor)
        assert output.shape == (1, 2, 512, 512)

    # ❌ Don't test internal implementation details
    def test_internal_layer_count():
        model = CrackSegmentationModel(num_classes=2)
        assert len(model._internal_layers) == 15  # Brittle internal detail
    ```

- **Production Code Drives Test Expectations:**
  - When tests fail due to incorrect expectations, adapt the test to match actual behavior
  - **Never modify production code just to satisfy arbitrary test requirements**
  - Only change production behavior for documented functional improvements
  - Example:
    ```python
    # ✅ Test adapts to actual production behavior
    def test_model_saves_checkpoint():
        trainer.save_checkpoint()
        # Test the actual path and format the system uses
        checkpoint_path = trainer.get_checkpoint_path()
        assert os.path.exists(checkpoint_path)
        assert checkpoint_path.endswith('.pth')

    # ❌ Don't force production code to create unnecessary artifacts
    def test_creates_summary_file():
        trainer.train()
        # Don't make trainer create files it doesn't need
        assert os.path.exists("unnecessary_summary.txt")
    ```

## Professional Testing Approach

- **Three-Option Analysis for Test Fixes:**
  - Before fixing failing tests, analyze at least 3 solutions:
    1. **Fix test expectations** (most common for incorrect assumptions)
    2. **Fix production code** (when actual bug exists)
    3. **Redesign the interface** (when design is fundamentally flawed)
  - Choose the most professional, robust, and maintainable approach
  - Document reasoning in commit messages or implementation logs

- **Quality Integration:**
  - Follow [coding-preferences.md](mdc:.roo/rules/coding-preferences.md) for test code quality
  - Use explicit type annotations for all test functions, fixtures, and helper methods
  - Apply Black formatting and pass Ruff linting for test files
  - Ensure test code passes `basedpyright .` with zero errors
  - Aim for >80% test coverage on core functionality

## Testing Strategy and Categories

### Unit Tests
Focus on individual components in complete isolation:

```python
# ✅ Pure unit test - isolated component testing
def test_dice_loss_calculation():
    loss_fn = DiceLoss()
    predictions = torch.randn(2, 2, 64, 64)
    targets = torch.randint(0, 2, (2, 64, 64))
    loss = loss_fn(predictions, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar loss
    assert 0.0 <= loss.item() <= 1.0  # Valid range for Dice loss
```

### Integration Tests
Test component interactions and workflows:

```python
# ✅ Integration test - multiple components working together
def test_training_pipeline_integration():
    model = CrackSegmentationModel(num_classes=2)
    trainer = ModelTrainer(model, config)
    sample_dataloader = create_sample_dataloader()

    metrics = trainer.train_epoch(sample_dataloader)

    assert "loss" in metrics
    assert "iou" in metrics
    assert isinstance(metrics["loss"], float)
    assert 0.0 <= metrics["iou"] <= 1.0
```

### Test Data and Fixtures Strategy

```python
# ✅ Realistic but minimal test data
@pytest.fixture
def sample_crack_image() -> torch.Tensor:
    """Generate a realistic test image for crack segmentation."""
    return torch.randn(3, 512, 512)  # RGB image

@pytest.fixture
def sample_binary_mask() -> torch.Tensor:
    """Generate a corresponding binary mask."""
    return torch.randint(0, 2, (512, 512))  # Binary mask

@pytest.fixture
def mock_config() -> ModelConfig:
    """Provide a minimal valid configuration for testing."""
    return ModelConfig(
        num_classes=2,
        input_channels=3,
        model_name="test_model"
    )
```

## Error Handling and Edge Cases

Test meaningful error scenarios and domain-specific edge cases:

```python
# ✅ Test meaningful error conditions
def test_model_handles_invalid_input_shape():
    model = CrackSegmentationModel(num_classes=2)
    invalid_input = torch.randn(1, 4, 512, 512)  # Wrong channels

    with pytest.raises(ValueError, match="Expected 3 channels"):
        model(invalid_input)

# ✅ Test domain-specific edge cases
def test_empty_mask_handling():
    """Test behavior with masks containing no positive pixels."""
    loss_fn = DiceLoss()
    predictions = torch.ones(1, 2, 64, 64)
    empty_target = torch.zeros(1, 64, 64)  # No cracks detected

    loss = loss_fn(predictions, empty_target)
    assert not torch.isnan(loss), "Loss should handle empty masks gracefully"
    assert loss.item() > 0, "Loss should be positive for wrong predictions"
```

## Test File Organization and Placement

**All test files must be organized according to their nature and scope within the `tests/` directory structure.**

### Directory Structure Requirements

- **Unit Tests (`tests/unit/`):**
  - Test individual components, functions, or classes in isolation
  - Mirror the `src/` directory structure exactly
  - One test file per source module (when feasible)
  - Example mapping:
    ```
    src/model/encoder/cnn_encoder.py → tests/unit/model/test_cnn_encoder.py
    src/training/losses/dice_loss.py → tests/unit/training/losses/test_dice_loss.py
    src/utils/config/validation.py  → tests/unit/utils/test_validation.py
    ```

- **Integration Tests (`tests/integration/`):**
  - Test component interactions and workflows
  - Organize by functional area or system boundary
  - Example organization:
    ```
    tests/integration/
    ├── model/               # Model component integration
    ├── training/            # Training pipeline integration
    ├── data/                # Data loading and processing
    ├── evaluation/          # Evaluation and metrics
    ├── config/              # Configuration system tests
    └── test_backward_compatibility.py  # Cross-system compatibility
    ```

### File Naming Conventions

- **Unit Test Files:** `test_{module_name}.py`
  - Mirror source module names exactly
  - Examples: `test_cnn_encoder.py`, `test_dice_loss.py`, `test_validation.py`

- **Integration Test Files:** `test_{functional_area}_{integration_type}.py`
  - Be descriptive about what integration is being tested
  - Examples: `test_model_training_pipeline.py`, `test_config_validation.py`, `test_backward_compatibility.py`

### Placement Decision Matrix

| Test Type | Scope | Location | Example |
|-----------|-------|----------|---------|
| **Single Function/Class** | Isolated component | `tests/unit/{module_path}/` | `CNNEncoder` → `tests/unit/model/test_cnn_encoder.py` |
| **Multiple Components** | Cross-component workflow | `tests/integration/{area}/` | Model + training → `tests/integration/training/test_trainer_integration.py` |
| **Configuration** | System configuration | `tests/integration/config/` | Hydra configs → `tests/integration/config/test_hydra_config.py` |
| **End-to-End** | Complete workflows | `tests/integration/` | Full pipeline → `tests/integration/test_full_pipeline.py` |
| **Backwards Compatibility** | System-wide compatibility | `tests/integration/` | Compatibility → `tests/integration/test_backward_compatibility.py` |

### Decision Process for Test Placement

**Follow this decision tree when creating new test files:**

1. **Is it testing a single module/class?** → `tests/unit/{mirror_src_path}/`
2. **Does it test interaction between components?** → `tests/integration/{functional_area}/`
3. **Is it testing configuration or system setup?** → `tests/integration/config/`
4. **Does it test backward compatibility across systems?** → `tests/integration/test_backward_compatibility.py`
5. **Is it a complete workflow test?** → `tests/integration/test_{workflow_name}.py`

### Special Cases and Guidelines

- **Shared Test Utilities:**
  - Global fixtures: `tests/conftest.py`
  - Area-specific fixtures: `tests/{area}/conftest.py`
  - Example: Model test fixtures in `tests/integration/model/conftest.py`

- **Mock Components:**
  - Keep mocks close to where they're used
  - Include in module-specific `conftest.py` files
  - Document mock behavior and limitations

- **Test Data:**
  - Small test data: inline in test files using fixtures
  - Large datasets: external files (not committed to repo)
  - Fixtures: use pytest fixtures in appropriate `conftest.py`

## Quality Standards for Test Files

```python
# ✅ Well-organized test file structure
"""
tests/unit/model/test_cnn_encoder.py

Unit tests for CNNEncoder component.
Tests encoder initialization, forward pass, and feature extraction.
"""
import pytest
import torch
from src.model.encoder.cnn_encoder import CNNEncoder

class TestCNNEncoder:
    """Test suite for CNNEncoder functionality."""

    def test_initialization(self) -> None:
        """Test encoder can be created with valid parameters."""
        encoder = CNNEncoder(in_channels=3, feature_dims=[64, 128, 256])
        assert encoder.in_channels == 3
        assert len(encoder.layers) == 3

    def test_forward_pass_shape(self) -> None:
        """Test output shapes match expected dimensions."""
        encoder = CNNEncoder(in_channels=3, feature_dims=[64, 128])
        input_tensor = torch.randn(2, 3, 224, 224)
        features = encoder(input_tensor)

        assert len(features) == 2
        assert features[0].shape[1] == 64  # First feature map channels
        assert features[1].shape[1] == 128  # Second feature map channels

    def test_feature_info_generation(self) -> None:
        """Test feature info metadata is correctly generated."""
        encoder = CNNEncoder(in_channels=3, feature_dims=[64, 128])
        feature_info = encoder.feature_info

        assert len(feature_info) == 2
        assert all('channels' in info for info in feature_info)
        assert all('reduction' in info for info in feature_info)
```

## When Tests Should Guide Code vs. When Code Should Guide Tests

### Legitimate Test-Driven Scenarios (Tests Guide Code):
- **Design phase**: Tests specify desired behavior for new features
- **Refactoring**: Tests protect against regressions and guide API improvements
- **Bug fixing**: Tests reproduce issues and verify fixes
- **Requirements clarification**: Tests document expected behavior

### Production-First Scenarios (Code Guides Tests):
- **Implementation details change**: Tests adapt to new internal organization
- **Output format evolution**: Tests update to match improved output formats
- **Performance optimizations**: Tests accommodate new efficient implementations
- **API improvements**: Tests update to reflect better interfaces

### Red Flags for Over-Adaptation:
- Test expects specific file names that aren't functionally necessary
- Test requires exact internal state that could change with implementation
- Test forces creation of outputs that serve no real purpose
- Test breaks when implementation improves without changing external behavior

## Integration with Project Workflow

### Pre-Commit Testing Workflow

```bash
# Run tests as part of quality gates (from workflow-preferences.md)
pytest tests/ --cov=src --cov-report=term-missing
basedpyright .
black .
ruff . --fix

# Only commit if all quality gates pass
```

### Test Organization Principles

- Follow [project-structure.md](mdc:.roo/guides/project-structure.md) for test layout
- Mirror source code structure in test directories
- Use descriptive test names that explain the scenario being tested
- Group related tests in classes when logical
- Include docstrings explaining test scope and purpose

### Examples from Current Project

- ✅ **Correct**: `tests/unit/model/test_feature_info_utils.py` - Tests single utility module
- ✅ **Correct**: `tests/integration/test_backward_compatibility.py` - Tests system-wide compatibility
- ✅ **Correct**: `tests/integration/config/test_hydra_config.py` - Tests configuration system
- ✅ **Correct**: `tests/integration/model/test_model_factory.py` - Tests model component integration

## References

- **Code Quality**: [coding-preferences.md](mdc:.roo/rules/coding-preferences.md)
- **Development Workflow**: [workflow-preferences.md](mdc:.roo/rules/workflow-preferences.md)
- **Project Structure**: [project-structure.md](mdc:.roo/guides/project-structure.md)
- **Architecture Guide**: [structural-guide.md](mdc:.roo/guides/structural-guide.md)

