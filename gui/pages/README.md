# GUI Pages - CrackSeg Project

## Overview

This directory contains all GUI pages for the CrackSeg application, organized
following ML project best practices with modular architecture and clear separation
of concerns.

## Directory Structure

```bash
gui/pages/
├── core/                    # Core application pages
│   ├── home/               # Home page and landing
│   └── navigation/         # Page routing and navigation
├── ml/                     # ML-specific pages
│   ├── training/           # Training pages and workflows
│   ├── config/             # Configuration pages
│   └── architecture/       # Model architecture pages
├── data/                   # Data-related pages
│   ├── results/            # Results visualization
│   └── analysis/           # Data analysis pages
└── deprecated/             # Obsolete pages (for removal)
```

## Page Categories

### Core Pages (`core/`)

- **Home**: Landing page and application overview
- **Navigation**: Page routing and navigation logic

### ML Pages (`ml/`)

- **Training**: Model training workflows and monitoring
- **Config**: Configuration management and editing
- **Architecture**: Model architecture visualization and selection

### Data Pages (`data/`)

- **Results**: Training results visualization and analysis
- **Analysis**: Data analysis and exploration tools

## Migration Guide

### From Old Structure

- `home_page.py` → `core/home/main.py`
- `train_page.py` → `ml/training/legacy.py`
- `page_train.py` → `ml/training/main.py`
- `config_page.py` → `ml/config/basic.py`
- `advanced_config_page.py` → `ml/config/advanced.py`
- `architecture_page.py` → `ml/architecture/main.py`
- `results_page.py` → `data/results/legacy.py`
- `results_page_new.py` → `data/results/main.py`

### Deprecated Files

- `results_page.py` (obsoleto, reemplazado por results_page_new.py)
- `train_page.py` (obsoleto, reemplazado por page_train.py)

## Best Practices

1. **Single Responsibility**: Each page has one clear purpose
2. **Modular Design**: Pages are self-contained and reusable
3. **Type Safety**: Full type annotations using Python 3.12+ features
4. **Error Handling**: Comprehensive error states and recovery
5. **Performance**: Optimized for user experience
6. **Testing**: Each page has corresponding unit tests

## Usage Examples

```python
# Core pages
from gui.pages.core.home import HomePage
from gui.pages.core.navigation import PageRouter

# ML pages
from gui.pages.ml.training import TrainingPage
from gui.pages.ml.config import ConfigPage
from gui.pages.ml.architecture import ArchitecturePage

# Data pages
from gui.pages.data.results import ResultsPage
from gui.pages.data.analysis import AnalysisPage
```

## Quality Standards

- All pages must pass quality gates: `black`, `ruff`, `basedpyright`
- Maximum file size: 300 lines (preferred), 400 lines (absolute max)
- Complete type annotations and docstrings
- Comprehensive error handling
- Unit tests with >80% coverage
