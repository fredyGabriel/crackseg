# Development Guide

This guide provides instructions for developers working on the CrackSeg Professional GUI.

## Development Setup

Follow the instructions in the [Installation Guide](INSTALL.md) to set up the basic environment.

### Quality Tools

This project uses a strict set of quality gates. All code must pass these checks before being committed.

- **`black`**: For automated code formatting.
- **`ruff`**: For linting and style checks.
- **`basedpyright`**: For static type checking.

You can run all checks with the following commands:

```bash
black .
ruff . --fix
basedpyright .
```

### Testing

The project uses `pytest` for testing.

- **Run all tests**:

  ```bash
  pytest tests/
  ```

- **Run tests with coverage**:

  ```bash
  pytest tests/ --cov=src --cov-report=term-missing
  ```

## Project Structure

The GUI code is located in `gui/`. It follows a modular structure:

- `app.py`: Main application entry point.
- `components/`: Reusable Streamlit components.
- `services/`: Business logic decoupled from the UI.
- `utils/`: Helper functions and utilities.

## Contributing

Please see the [Contributing Guide](CONTRIBUTING.md) for details on our development workflow, pull
request process, and code review standards.
