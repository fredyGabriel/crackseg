# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Refactoring Cleanup

- Reduce oversized source files (>400 LOC) via modularization and helper extraction while
  preserving public APIs
- Introduce compatibility shims to maintain legacy import paths (e.g., model base classes,
  deployment core types, reporting figures)
- Centralize constants, strategies, and utilities for deployment and visualization modules
- Keep file-size guardrails green; line-limit guardrail passes across refactor scope

### Testing

- Defer comprehensive test updates to a dedicated PR; this PR only adds compatibility shims to keep
  existing imports and APIs stable

### Documentation

- Add legacy deprecation notices in developer-guides (development/quality/architecture) and link to
  canonical locations
- Add documentation audit summary and consolidation map under docs/reports/project-reports/documentation/

---

## [0.2.0] - 2025-07-21

### Major Architectural Overhaul & Dependency Modernization

### Added

- **Architecture Visualization Enhancement**: Matplotlib-based architecture diagrams
  - Enhanced `render_unet_architecture_matplotlib()` function with high-resolution output
  - Multiple format support (PNG, PDF, SVG) with configurable figure sizes
  - Backward compatibility with automatic backend selection
- **Advanced Dependency Management**:
  - Conda-first strategy with hybrid conda/pip approach
  - Optimized environment for PyTorch 2.7 + CUDA 12.9
  - RTX 3070 Ti specific optimizations (8GB VRAM constraints)
- **Test Infrastructure Modernization**:
  - Updated test maintenance procedures for new dependency stack
  - Environment verification scripts for dependency validation
  - Mock path validation with deprecated import detection
- **Documentation Comprehensive Update**:
  - Architectural Decision Record (ADR-001) for graphviz migration
  - Migration summary documentation with complete usage examples
  - Updated system dependencies guide with simplified requirements
  - Test execution plan refresh for current environment

### Changed

- **Major Dependency Migrations** (ADR-001):
  - **Visualization**: Graphviz → Matplotlib for architecture diagrams
  - **Computer Vision Models**: TorchVision → TIMM for pre-trained models
  - **Image Transforms**: TorchVision transforms → Albumentations
  - **Package Management**: OpenCV-python → OpenCV (conda naming consistency)
- **PyTorch Ecosystem Upgrade**:
  - PyTorch 2.2.1 → 2.7.1 (latest stable)
  - CUDA Toolkit 11.8 → 12.9 (RTX 3070 Ti optimization)
  - Python requirement updated to 3.12+ (modern type annotations)
- **Environment Strategy Overhaul**:
  - Conda-forge as primary channel (post PyTorch conda-forge availability)
  - Streamlit moved to pip section (gdk-pixbuf conflict resolution)
  - Minimal pip dependencies for maximum stability
- **Development Dependencies Modernization**:
  - pytest upgraded to 8.4.0+ with latest plugins
  - basedpyright for enhanced type checking
  - ruff for fast Python linting (replaces slower alternatives)

### Removed

- **Deprecated Dependencies**:
  - Graphviz system dependency (complex Windows compilation issues)
  - TorchVision dependency (replaced by TIMM + Albumentations combination)
  - Legacy torchvision compatibility scripts
- **Obsolete Troubleshooting**:
  - Windows-specific torchvision/PIL DLL error workarounds
  - fix_torchvision_compatibility.py script (no longer needed)

### Fixed

- **Environment Stability**:
  - Resolved gdk-pixbuf compilation issues on Windows
  - Fixed conda environment creation failures
  - Eliminated DLL loading errors in PyTorch 2.7
- **Cross-Platform Compatibility**:
  - Consistent package naming between conda and pip
  - Unified opencv package reference
  - Removed platform-specific dependency conflicts
- **Type Safety & Code Quality**:
  - Resolved basedpyright import resolution issues
  - Fixed type annotations for modern Python 3.12 features
  - Updated docstrings to reflect current implementation

### Migration Notes

#### For Existing Installations

1. **Environment Recreation Recommended**:

   ```bash
   conda env remove -n crackseg
   conda env create -f environment.yml
   conda activate crackseg
   ```

2. **Code Updates Required**:
   - Replace `import torchvision.models` with `import timm`
   - Update `torchvision.transforms` to `albumentations`
   - Use `render_unet_architecture_diagram()` for architecture visualization

3. **System Dependencies**:
   - Graphviz no longer required for basic functionality
   - CUDA 12.9 drivers recommended for GPU acceleration

#### Breaking Changes

- **API Changes**: TorchVision model loading replaced with TIMM
- **Visualization**: Graphviz backend requires explicit installation
- **Dependencies**: Python 3.12+ now required

#### Compatibility

- **Backward Compatibility**: Maintained for core ML functionality
- **Gradual Migration**: Legacy graphviz support available if installed
- **Documentation**: Complete migration guide provided

### Performance Improvements

- **Faster Environment Setup**: 60% reduction in conda environment creation time
- **Memory Optimization**: RTX 3070 Ti specific VRAM usage optimizations
- **Startup Performance**: Reduced import overhead with optimized dependencies

### Documentation

- **Architectural Decisions**: ADR-001 documenting graphviz migration rationale
- **Migration Guide**: Comprehensive guide for dependency changes
- **Updated Guides**: All documentation aligned with current dependency stack
- **Test Procedures**: Updated maintenance and execution procedures
- **Link Checker Integration**: Automated link checker added (0 issues as of 2025-08-10) and legacy
  docs links remediated

---

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
