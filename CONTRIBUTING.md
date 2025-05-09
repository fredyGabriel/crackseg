# Contributing to Pavement Crack Segmentation Project

Thank you for your interest in contributing to this project! This guide will help you understand the contribution workflow, coding standards, and best practices to ensure your contributions align with the project's goals and quality standards.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Project Objectives and Technical Approach](#project-objectives-and-technical-approach)
- [Key Terminology](#key-terminology)
- [Coding Standards](#coding-standards)
- [Git Workflow](#git-workflow)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Code of Conduct](#code-of-conduct)

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** to your local machine
   ```bash
   git clone https://github.com/YOUR-USERNAME/crackseg.git
   cd crackseg
   ```
3. **Add the upstream repository** as a remote to sync with the main project
   ```bash
   git remote add upstream https://github.com/ORIGINAL-OWNER/crackseg.git
   ```
4. **Create a new branch** for your feature or bugfix
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Environment

1. **Set up Conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate torch
   ```

2. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env file with your required values
   ```

3. **Verify installation**
   ```bash
   # Run tests to ensure everything is set up correctly
   pytest
   ```

## Project Objectives and Technical Approach

This project focuses on exploring state-of-the-art (SOTA) deep learning models for semantic segmentation of cracks in asphalt pavement. Our approach involves:

- Developing and comparing various U-Net based architectures
- Experimenting with different encoders, bottlenecks, and decoders
- Integrating attention mechanisms to improve feature discrimination
- Benchmarking against established SOTA metrics in the pavement crack segmentation domain

When contributing, keep in mind the modular architecture design pattern that allows for interchangeable components:

- **Encoders**: CNN-based or Swin Transformer V2-based feature extractors
- **Bottlenecks**: Standard convolutions, ASPP, or ConvLSTM
- **Decoders**: CNN-based or Transformer-based upsampling paths
- **Attention Mechanisms**: Various attention modules such as CBAM

Performance goals include achieving high IoU (>0.8) and F1-scores on standard datasets, surpassing current SOTA benchmarks.

## Key Terminology

To help you navigate the codebase and documentation, here are some key terms used throughout the project:

- **ABC (Abstract Base Class)**: Python classes that define the contract for component implementations
- **AMP (Automatic Mixed Precision)**: Technique used to optimize training speed and memory usage
- **ASPP (Atrous Spatial Pyramid Pooling)**: A module used in bottlenecks for multi-scale context extraction
- **CBAM (Convolutional Block Attention Module)**: Attention mechanism for improving feature discriminability
- **CNN (Convolutional Neural Network)**: Standard deep learning architecture for image processing
- **F1-Score**: A key metric combining precision and recall for model evaluation
- **Hydra**: The configuration framework used for managing experiments and parameters
- **IoU (Intersection over Union)**: Primary evaluation metric for segmentation quality
- **SOTA (State Of The Art)**: Benchmark for comparing model performance against leading methods
- **U-Net**: The base encoder-decoder architecture with skip connections
- **VRAM**: Video RAM considerations for model design (project targets 8GB VRAM)

## Coding Standards

This project follows strict coding standards to ensure readability, maintainability, and reliability:

1. **Follow PEP 8** style guide for Python code, covering naming, spacing, layout, etc.
2. **Keep line length** within 79 characters where possible.
3. **Documentation and Comments**:
   - Write clear, concise, and **very brief** docstrings for all modules, classes, functions, and methods.
   - Use comments to clarify complex logic, but do not overuse them.
   - All docstrings and comments **must be in English**.
4. **Modular Design**:
   - Structure code into small, focused, and reusable modules.
   - Keep individual files manageable, ideally between 200-300 lines of code.
   - Never exceed 400 lines per file without justification.
   - Adhere to the Abstract Base Class (ABC) pattern for component interfaces.
5. **Configuration via Hydra**:
   - All configuration parameters must be defined and loaded using Hydra and YAML files.
   - Avoid hardcoding parameters in code.

## Git Workflow

1. **Keep branches updated** with the upstream main branch
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Make regular, small commits** that focus on a single change
   ```bash
   git add <files>
   git commit -m "your commit message"
   ```

3. **Push changes** to your fork
   ```bash
   git push origin feature/your-feature-name
   ```

## Commit Message Guidelines

**All commit messages must be written in clear, correct English**, regardless of the language used in code comments or documentation. This ensures consistency and accessibility for all contributors and reviewers.

### Commit Message Format

Follow this structure:
```
type(scope): Brief description of the change

- Bullet point details (optional)
- More details (optional)
```

### Types
- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation changes
- **style**: Formatting changes that don't affect code functionality
- **refactor**: Code changes that neither fix bugs nor add features
- **perf**: Performance improvements
- **test**: Adding or correcting tests
- **chore**: Changes to build process, dependencies, etc.

### Scope
Indicate the module, file, or component affected (e.g., model, encoder, training, data).

### Examples

✅ **DO**:
```bash
git commit -m "fix(model): Correct off-by-one error in encoder output

- Adjusted index calculation in forward method
- Added test for edge case"
```

```bash
git commit -m "feat(training): Add early stopping with patience parameter"
```

❌ **DON'T**:
```bash
git commit -m "fixed bug"
```

```bash
git commit -m "arreglo bug en el encoder"  # Not in English
```

## Testing Guidelines

1. **Write tests** for all new functionality and bug fixes
   ```bash
   # Unit tests go in tests/unit/
   # Integration tests go in tests/integration/
   ```

2. **Run tests locally** before submitting a PR
   ```bash
   pytest
   ```

3. **Test specific areas** when needed
   ```bash
   pytest tests/unit/model/
   ```

4. **Follow test conventions**:
   - Use descriptive test names
   - Set up required fixtures
   - Include both positive and negative test cases
   - Mock external dependencies

5. **Performance Testing**:
   - For model architecture changes, evaluate performance using standard metrics:
     - Intersection over Union (IoU)
     - F1-Score
   - Compare results against baseline models and SOTA benchmarks
   - Document performance improvements or regressions

## Documentation

1. **Update documentation** for any code changes
   - Modify relevant README.md files
   - Update docstrings
   - Maintain YAML configuration file comments

2. **Document new features** with examples
   - Include sample configuration options
   - Describe expected inputs and outputs

3. **Follow docstring conventions**:
   ```python
   def function_name(param1, param2):
       """Brief one-line description.

       Args:
           param1: Description of first parameter
           param2: Description of second parameter

       Returns:
           Description of the return value
       """
   ```

4. **SOTA Performance Documentation**:
   - When adding or improving model architectures, document:
     - Specific configuration used
     - Dataset(s) and preprocessing details
     - Performance metrics (IoU, F1-Score)
     - Comparison to previous results and published benchmarks
     - Training resources (VRAM usage, approximate training time)

## Pull Request Process

1. **Create a pull request** from your feature branch to the upstream main branch
2. **Fill in the PR template** with:
   - Description of changes
   - Related issue(s)
   - Type of change (bugfix, feature, etc.)
   - Checklist of completed tasks
3. **Ensure all checks pass**:
   - CI pipeline tests
   - Code coverage requirements
   - Linting rules
4. **Address review comments** promptly
5. **Update your PR** if requested with additional changes

## Code of Conduct

- Be respectful and professional in all communications
- Provide constructive feedback
- Focus on the code, not the contributor
- Value inclusivity and diverse perspectives
- Help maintain a positive community atmosphere

---

Thank you for contributing to the Pavement Crack Segmentation Project! 