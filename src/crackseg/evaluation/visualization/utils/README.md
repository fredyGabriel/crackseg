# Visualization Utilities

Utilities for advanced prediction visualization:

- `images.py`: load/save image I/O helpers
- `plot_utils.py`: grids, overlays, legends, axes helpers
- `gradients.py`: gradient norm computations

Principles:
- Thin wrappers in high-level visualizers; heavy ops live here
- Keep interfaces stable for downstream usage
