# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Dependency updates verification script (`scripts/check_updates.py`)
- Comprehensive dependency management documentation in README.md
- Directory-specific README.md files with detailed documentation
- Structured project layout with clear organization
- Data directory structure for training, validation, and testing

### Changed
- Directory structure reorganized for better organization:
  - Moved checkpoints/ to outputs/checkpoints/
  - Moved metrics/ to outputs/metrics/
  - Moved visualizations/ to outputs/visualizations/
- Documentation updated and translated to English
- Improved project structure documentation

### Dependencies
Current stable versions:
- PyTorch 2.2.1
- torchvision 0.17.1 (update to 0.22.0 available)
- albumentations 2.0.5 (update to 2.0.6 available)
- hydra-core 1.3.2
- pytest 8.3.5
- numpy 2.2.3 (update to 2.2.5 available)
- matplotlib 3.10.1
- scikit-image 0.25.2
- opencv-python (version TBD)
- python-dotenv (version TBD)

## [0.1.0] - 2025-05-01

### Added
- Initial project setup with conda environment
- Base repository structure following best practices
- Core U-Net implementation for crack segmentation
- Hydra configuration system for experiment management
- Data loading and preprocessing pipeline
- Basic training and evaluation scripts
- Logging and visualization utilities
- Test framework setup

### Initial Dependencies
- PyTorch 2.2.1
- torchvision 0.17.1
- albumentations 2.0.5
- hydra-core 1.3.2
- pytest 8.3.5
- python-dotenv (version TBD)
- opencv-python (version TBD)
- numpy 2.2.3
- matplotlib 3.10.1
- scikit-image 0.25.2 