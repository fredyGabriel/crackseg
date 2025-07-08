# Code Quality Gates Guide

## Overview

This guide documents the comprehensive code quality gates system implemented for the CrackSeg
project, with specific focus on the GUI codebase quality standards established in **Task #2**.

## Quality Gates Implementation

### 📋 Quality Standards Applied

All Python code in the CrackSeg project must pass the following quality gates:

1. **🔍 Ruff Linting** - Code style and error detection
2. **🎨 Black Formatting** - Consistent code formatting
3. **🔬 Basedpyright Type Checking** - Type safety verification
4. **🧪 Test Coverage** - Minimum 80% coverage requirement

### 🖥️ GUI Code Quality Results

**Status**: ✅ **ALL QUALITY GATES PASSING**

As of the latest implementation (Task #2.5), all **163 GUI Python files** pass all quality gates:

- **Ruff**: "All checks passed!" ✅
- **Black**: "163 files would be left unchanged" ✅
- **Basedpyright**: "0 errors, 0 warnings, 0 notes" ✅

## Environment Setup

### Prerequisites

```bash
# Activate the conda environment (REQUIRED for all commands)
conda activate crackseg
```

> **⚠️ CRITICAL**: All quality gate commands MUST be prefixed with `conda activate crackseg &&` due
> to PowerShell environment requirements.

### Local Quality Gate Commands

#### Core Code Quality Checks

```bash
# Run all quality gates on source code
conda activate crackseg && python -m ruff check src/ --fix
conda activate crackseg && black src/
conda activate crackseg && basedpyright src/
```

#### GUI Code Quality Checks

```bash
# Run all quality gates on GUI code
conda activate crackseg && python -m ruff check scripts/gui/ --fix
conda activate crackseg && black scripts/gui/
conda activate crackseg && basedpyright scripts/gui/
```

#### Combined Quality Gates

```bash
# Run all quality gates together (recommended for CI-like verification)
conda activate crackseg && python -m ruff check . && black . --check && basedpyright .
```

## CI/CD Integration

### GitHub Actions Workflow

The quality gates are enforced through GitHub Actions in `.github/workflows/quality-gates.yml`:

#### Workflow Structure

1. **Core Quality Gates** (`quality-gates-core`)
   - Ruff linting on `src/` directory
   - Black formatting check on `src/` directory
   - Basedpyright type checking on `src/` directory

2. **GUI Quality Gates** (`quality-gates-gui`)
   - Ruff linting on `scripts/gui/` directory
   - Black formatting check on `scripts/gui/` directory
   - Basedpyright type checking on `scripts/gui/` directory
   - Detailed quality reporting

3. **Test Coverage** (`test-coverage`)
   - Unit and integration test execution
   - Coverage analysis with 80% minimum threshold
   - Coverage reporting to Codecov

#### Automated Triggers

Quality gates run automatically on:

- Push to `main` or `develop` branches
- Pull requests targeting `main` or `develop`
- Changes to source code (`src/**`, `scripts/**`, `tests/**`)

## Quality Gate Details

### 🔍 Ruff Linting

**Purpose**: Detects code style issues, potential bugs, and enforces best practices

**Configuration**: Uses project's `pyproject.toml` settings

**Common Issues Fixed**:

- Unused imports removal
- Line length violations (>88 characters)
- Import organization
- Code style improvements

**Manual Fix Command**:

```bash
conda activate crackseg && python -m ruff check scripts/gui/ --fix
```

### 🎨 Black Formatting

**Purpose**: Ensures consistent code formatting across the project

**Standards**:

- Maximum line length: 88 characters
- Consistent indentation and spacing
- Standardized quote usage

**Manual Fix Command**:

```bash
conda activate crackseg && black scripts/gui/
```

### 🔬 Basedpyright Type Checking

**Purpose**: Ensures type safety and catches type-related errors

**Requirements**:

- Complete type annotations using Python 3.12+ features
- Proper import statements
- Type compatibility verification

**Common Issues Fixed**:

- Missing type annotations
- Incompatible function signatures
- Import resolution errors
- Generic type specifications

## Development Guidelines

### Pre-Commit Workflow

Before committing code, always run:

```bash
# Quick quality check
conda activate crackseg && python -m ruff check scripts/gui/ --fix && black scripts/gui/ && basedpyright scripts/gui/
```

### IDE Integration

Configure your IDE to run quality gates automatically:

#### VS Code Settings

Add to `.vscode/settings.json`:

```json
{
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.analysis.typeCheckingMode": "strict"
}
```

### File Size Guidelines

**Maximum file sizes** (automatically enforced):

- Python files: 300 lines (warning), 400 lines (hard limit)
- Large files require refactoring justification

## Troubleshooting

### Common Environment Issues

#### Problem: Commands not found

```txt
black: The term 'black' is not recognized
```

**Solution**: Always activate conda environment first

```bash
conda activate crackseg && black scripts/gui/
```

#### Problem: Type checking errors

```txt
basedpyright: Cannot resolve import
```

**Solutions**:

1. Check import paths are correct
2. Ensure all dependencies are installed
3. Verify `__init__.py` files exist in packages

### Quality Gate Failures

#### Ruff Failures

- Run with `--fix` flag to auto-fix most issues
- Review remaining issues manually
- Check line length violations

#### Black Failures

- Run Black to auto-format files
- Check for syntax errors preventing formatting

#### Type Check Failures

- Add missing type annotations
- Fix import statements
- Resolve type compatibility issues

## Quality Metrics

### Current Status (Task #2 Completion)

| Component | Files | Ruff | Black | Basedpyright |
|-----------|-------|------|-------|--------------|
| GUI Code  | 163   | ✅    | ✅     | ✅            |
| Core Code | ~200  | ✅    | ✅     | ✅            |

### Coverage Requirements

- **Minimum Coverage**: 80%
- **Target Coverage**: 90%+
- **Critical Components**: 95%+

## Implementation History

### Task #2: GUI Quality Gates Implementation

**Completed**: Successfully applied quality gates to all GUI code

**Key Achievements**:

- Fixed all type checking errors across 163 GUI files
- Standardized code formatting with Black
- Resolved linting issues with Ruff
- Implemented comprehensive CI/CD quality gates
- Created development documentation and guidelines

**Files Impacted**:

- `scripts/gui/pages/home_page.py` - Function signature standardization
- `scripts/gui/utils/log_parser.py` - Pandas DataFrame compatibility
- `scripts/gui/utils/performance_optimizer.py` - Type annotations
- `scripts/gui/utils/results/core.py` - Missing property implementation
- Multiple import and formatting fixes across GUI codebase

## Next Steps

1. **Extend coverage** to additional project components
2. **Monitor quality metrics** through CI/CD dashboards
3. **Regular audits** of quality gate effectiveness
4. **Developer training** on quality standards
5. **Tool updates** and configuration refinements

## References

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Black Documentation](https://black.readthedocs.io/)
- [Basedpyright Documentation](https://github.com/DetachHead/basedpyright)
- [Project Coding Standards](/.cursor/rules/coding-standards.mdc)
- [Development Workflow](/.cursor/rules/development-workflow.mdc)
