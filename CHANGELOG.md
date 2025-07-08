# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **GUI Overhaul**: Complete redesign and implementation of the Streamlit GUI for a modern,
  intuitive, and functional user experience.
  - **Home Page**: New dashboard providing quick actions, experiment status, and dataset statistics.
  - **Configuration Page**: Reworked with interactive components:
    - File browser for project configurations.
    - File uploader for external configs.
    - Advanced YAML editor with real-time Hydra validation.
    - "Save As" dialog with validation and smart naming.
  - **Training Page**: Transformed from a placeholder into a functional training hub:
    - Non-blocking execution of the training script.
    - Live log viewer to monitor process output.
    - Real-time charting of key metrics like loss and validation loss.
  - **Header Component**: A consistent and reusable header with project logo and page title,
    applied across all pages.
- **Git repository cleanup and refactoring**:
  - Improved `.gitignore` to exclude artifacts, logs, outputs, coverage, site, and temporary files.
  - Removed all files and folders now ignored from git history.
- Dependency and configuration updates:
  - Updated `requirements.txt` for local GUI development, with clarifying comments for optional dependencies.
  - Updated `pyproject.toml`: expanded Black exclusion, added multiplatform comments.
- Documentation review and synchronization:
  - Updated `README.md` to reflect the real project state, test coverage, and CI/CD.
  - Reviewed and aligned configuration and rule documentation
    (`pyproject.toml`, `.roo/rules/coding-preferences.md`) with modern standards.
- Prioritized checklist of critical files/documents to keep synchronized.
- Reviewed and documented scripts/utilities for project structure and report organization maintenance.

### Changed

- Reinforced documentation policy: all `README.md`, guides, and specifications must be updated after
  code, structure, or dependency changes.
- Added clarifying comments in configuration files to avoid multiplatform errors.
- Standardized all Markdown headings to ATX style throughout documentation to comply with linters.

### Fixed

- Removed unnecessary files and folders from git that are now in `.gitignore`.
- Fixed potential dependency conflicts (`opencv-python` vs `opencv-python-headless`).
- Resolved Markdown linter warnings (MD003/heading-style).

### Notes

- The repository is now clean, synchronized, and aligned with best practices for version control
  and documentation.
- It is recommended to maintain the documentation and changelog update workflow after every relevant
  change.

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
