# Project Tests

This directory contains the unit and integration tests for the crack segmentation project.

## Directory Structure

The test structure is organized by test type:

```
tests/
├── unit/          # Unit tests organized by module
│   ├── data/
│   ├── model/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── integration/   # Integration tests organized by module or workflow
│   ├── data/
│   ├── model/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── conftest.py    # Shared pytest fixtures
├── README.md      # This file
└── __init__.py    # Test package initialization
```

## Types of Tests

### Unit Tests
- Test individual components of each module
- Use mocking for dependencies
- Verify isolated behavior
- Located in `tests/unit/<module>/`

### Integration Tests
- Test the interaction between several modules or the full workflow
- Verify integration and configuration
- Located in `tests/integration/<module>/` or directly in `integration/` if they affect multiple modules

## Running Tests

### Run all tests:
```bash
pytest tests/
```

### Run only unit tests:
```bash
pytest tests/unit/
```

### Run only integration tests:
```bash
pytest tests/integration/
```

### Run tests for a specific module:
```bash
pytest tests/unit/model/
pytest tests/integration/training/
```

### Run a specific test:
```bash
pytest tests/unit/model/test_unet.py
```

### Run with coverage:
```bash
pytest --cov=src tests/
```

## Fixtures

Common fixtures are located in `conftest.py` at the root of `tests/`:
- Test data
- Mock configurations
- Shared utilities

## Best Practices

1. Keep tests independent and reproducible
2. Use descriptive names for test functions
3. Briefly document complex test cases
4. Keep test data small and representative
5. Update or add tests when modifying code
6. Clearly separate unit and integration tests

## Code Coverage

It is recommended to maintain at least 80% coverage in:
- Data modules
- Model architectures
- Training logic
- Critical utilities

## Test Organization

- Each module should have its own set of unit and integration tests
- Use test classes to group related cases
- Keep tests focused and specific
- Document special configurations

## Examples

```python
# Example unit test
def test_dataset_loading():
    dataset = CrackDataset(...)
    assert len(dataset) > 0
    assert dataset[0]['image'].shape == (3, 512, 512)

# Example integration test
def test_training_loop():
    trainer = Trainer(...)
    trainer.train()
    assert trainer.metrics['val_loss'] < initial_loss
``` 