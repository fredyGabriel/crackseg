# Deployment Core

Utilities and primitives for deployment orchestration.

- `manager.py`: high-level orchestration
- `strategies.py`: blue/green, canary, rolling, recreate
- `types.py`: deployment types and results
- `__init__.py`: stable public API

Responsibilities:

- Provide strategy implementations
- Keep imports stable via re-exports
- Avoid inline templates (moved to packaging/helm_templates)
