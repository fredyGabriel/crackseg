# Reporting Utilities

Shared helpers for reporting and figure generation.

- `data_loading.py`: load and validate reporting data
- `figures.py`: matplotlib setup and multi-format saving
- `templates/utils/html_common.py`: shared HTML/CSS helpers

Guidelines:

- Keep heavy plotting code in dedicated modules
- Re-export public helpers in package `__init__` if needed
