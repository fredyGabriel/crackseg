# Tutorial Scripts

This directory contains tutorial and verification scripts for learning and validating CrackSeg components.

## Structure

- **tutorial_03_verification.py**: Comprehensive verification tutorial for CrackSeg components

## Usage

### Component Verification Tutorial

```bash
python tests/tutorials/tutorial_03_verification.py
```

## Features

- **Loss function verification**: Tests loss registry and creation
- **Configuration validation**: Verifies Hydra configuration system
- **Model component testing**: Validates model factory and components
- **Optimizer verification**: Tests optimizer creation and configuration
- **Integration validation**: End-to-end component verification

## Learning Objectives

This tutorial demonstrates:

1. How to create and register custom loss functions
2. How to configure experiments with Hydra
3. How to create and validate model components
4. How to set up optimizers and training configurations
5. How to verify component integration

## Integration

This tutorial integrates with:

- `src/crackseg/training/losses/`: Loss function registry
- `src/crackseg/model/factory/`: Model creation utilities
- `src/crackseg/utils/config/`: Configuration management
- `configs/`: Hydra configuration files

## Best Practices

1. Run verification tutorials after major changes
2. Use as a learning resource for new contributors
3. Validate component integration before deployment
4. Document any issues found during verification
