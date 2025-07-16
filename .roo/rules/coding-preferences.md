---
description:
globs:
alwaysApply: true
---
# Code Quality Standards (Mandatory)

- **All Python code must pass three quality gates before commit:**
  - **Type checking**: `basedpyright .` with zero errors or warnings
  - **Formatting**: `black .` applied consistently
  - **Linting**: `ruff .` with no violations
  - **Configuration**: See `configs/linting/config.yaml` and `pyproject.toml`

- **Type annotations are mandatory for all code:**
  - Every function, method, class attribute, and module-level variable must have explicit type hints
  - No `Any` types without clear justification and documentation
  - **Use modern Python 3.12+ built-in generics**: `list[str]`, `dict[str, int]`, `tuple[int, ...]`,
    etc.
  - **Import from typing only when necessary**: `Optional`, `Union`, `Callable`, `Protocol`, etc.
  - Example:

    ```python
    from typing import Optional

    def process_data(items: list[str], threshold: Optional[int] = None) -> dict[str, int]:
        """Process data items and return statistics."""
        return {"count": len(items), "threshold": threshold or 0}
    ```

- **Modern Generic Type Syntax (Python 3.12+ PEP 695):**
  - Use new generic syntax for classes and functions when possible
  - Example:

    ```python
    # ✅ Modern approach (Python 3.12+)
    class Container[T]:
        def __init__(self, items: list[T]) -> None:
            self.items = items

        def get_first(self) -> T:
            return self.items[0]

    def process_batch[T](mdc:items: list[T], processor: Callable[[T], T]) -> list[T]:
        return [processor(item) for item in items]

    # ✅ Type aliases using new syntax
    type ProcessResult[T] = dict[str, T]
    type DataBatch[T] = list[T]
    ```

- **Built-in Generic Types (Python 3.9+ PEP 585):**
  - **Use built-in collections with subscript notation:**
    - `list[T]` instead of `typing.List[T]`
    - `dict[K, V]` instead of `typing.Dict[K, V]`
    - `tuple[T, ...]` instead of `typing.Tuple[T, ...]`
    - `set[T]` instead of `typing.Set[T]`
  - **Import from typing only for advanced types:**

    ```python
    # ✅ Modern imports
    from typing import Optional, Union, Callable, Protocol, TypeAlias
    from collections.abc import Iterable, Mapping, Sequence

    # ❌ Avoid these obsolete imports for Python 3.12
    # from typing import List, Dict, Tuple, Set
    ```

- **Pre-commit workflow:**

  ```bash
  black .
  ruff . --fix
  basedpyright .
  # Only commit if all three pass
  ```

## Code Structure and Organization

- **Modular Design:**
  - Functions/classes should have single, clear responsibilities
  - Modules should be focused and cohesive (200-300 lines preferred, max 400)
  - Place code in appropriate directories following `project-structure.md`
  - No one-off scripts in `src/` - use `scripts/` for utilities

- **PEP 8 Compliance:**
  - Follow PEP 8 naming conventions: `snake_case` for variables/functions, `PascalCase` for classes
  - Black handles line length and formatting - accept its decisions
  - Example:

    ```python
    class DataProcessor:  # PascalCase for classes
        def __init__(self, config_path: str) -> None:
            self.config_path = config_path  # snake_case for attributes

        def process_batch(self, data_items: list[str]) -> ProcessResult:  # snake_case for methods
            pass
    ```

- **DRY Principle:**
  - Extract common functionality into reusable functions/classes
  - Use inheritance or composition to avoid code duplication
  - Create utility modules for shared logic

## Documentation and Comments

- **Docstrings (Required):**
  - All modules, classes, and public functions must have English docstrings
  - Use Google-style or NumPy-style format consistently
  - Example:

    ```python
    def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """Calculate evaluation metrics for model predictions.

        Args:
            predictions: Model output tensor of shape (N, C, H, W)
            targets: Ground truth tensor of shape (N, H, W)

        Returns:
            Dictionary containing IoU, accuracy, and F1 scores
        """
    ```

- **Comments (Minimal):**
  - Only comment non-obvious business logic or algorithmic decisions
  - Avoid obvious comments: `# increment counter` for `counter += 1`

## Reliability and Error Handling

- **Error Handling:**
  - Use specific exception types, not bare `except:`
  - Handle expected errors gracefully with meaningful messages
  - Let unexpected errors bubble up with proper stack traces
  - Example:

    ```python
    def load_config(path: str) -> Config:
        try:
            with open(path, 'r') as f:
                return Config.from_dict(yaml.safe_load(f))
        except FileNotFoundError:
            raise ConfigError(f"Configuration file not found: {path}")
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {path}: {e}")
    ```

- **Test Coverage:**
  - Write unit tests for all public functions and methods
  - Include integration tests for key workflows
  - Aim for >80% coverage on core functionality

## Configuration and Dependencies

- **Dependency Management:**
  - **Prefer CONDA for package installation and environment management**
    - Use `conda install <package>` when available in conda-forge or main channels
    - Only use `pip install` for packages not available in conda repositories
    - Update `environment.yml` with conda packages and `requirements.txt` for pip-only packages
    - Example conda installation:
      `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`
  - Use `environment.yml` for conda packages and versions
  - Pin specific versions for reproducibility
  - Document any special installation requirements

- **Configuration Management:**
  - All parameters configurable via Hydra/YAML files
  - Store sensitive config in `.env` files (never committed)
  - Use type-safe config classes with validation
  - Example:

    ```python
    @dataclass
    class TrainingConfig:
        learning_rate: float
        batch_size: int
        epochs: int

        def __post_init__(self) -> None:
            if self.learning_rate <= 0:
                raise ValueError("Learning rate must be positive")
    ```

- **File Encoding:**
  - All Python files must use UTF-8 encoding
  - Include `# -*- coding: utf-8 -*-` only if needed for compatibility

## Advanced Type Patterns for Python 3.12+

- **Protocol Definition:**

  ```python
  from typing import Protocol

  class Drawable(Protocol):
      def draw(self) -> None: ...

  # Usage with modern generics
  class Canvas[T: Drawable]:
      def __init__(self, items: list[T]) -> None:
          self.items = items

      def render_all(self) -> None:
          for item in self.items:
              item.draw()
  ```

- **Advanced Generic Patterns:**

  ```python
  from collections.abc import Callable, Mapping
  from typing import TypeVar, ParamSpec, Concatenate

  # Only use typing imports for advanced features not available as built-ins
  P = ParamSpec('P')
  R = TypeVar('R')

  def with_logging[**P, R](mdc:func: Callable[P, R]) -> Callable[P, R]:
      def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
          print(f"Calling {func.__name__}")
          return func(*args, **kwargs)
      return wrapper
  ```

## References

- **Project Structure**: [project-structure.md](mdc:.roo/guides/project-structure.md)
- **Architecture Patterns**: [structural-guide.md](mdc:.roo/guides/structural-guide.md)
- **Development Process**: [development-guide.md](mdc:.roo/guides/development-guide.md)
- **Linting Configuration**: `configs/linting/config.yaml`, `pyproject.toml`
- **Project Context**: [general-context.md](mdc:.roo/guides/general-context.md)

### Quick References

- [general-context.md](mdc:.roo/guides/general-context.md): Project Proposal & Technical Discussion
- [project-structure.md](mdc:.roo/guides/project-structure.md): Directory Tree
- [structural-guide.md](mdc:.roo/guides/structural-guide.md): Structural Guide. Minimal structural outline
- [development-guide.md](mdc:.roo/guides/development-guide.md): outlines a suggested step-by-step
  process for developing the pavement crack segmentation project.
- [glossary.md](mdc:.roo/guides/glossary.md): Glossary of Key Terms

---
description: Require full type annotations, Black formatting, and Ruff linting for all Python code using modern Python 3.12+ built-in generics
globs: src/**/*.py, tests/**/*.py, scripts/**/*.py
alwaysApply: true
---

- **Mandatory Code Quality Standards:**
  - All Python code must include explicit type annotations for all functions, methods, and
    variables, sufficient to satisfy basedpyright with zero errors or warnings.
  - All code must be formatted using Black and linted with Ruff, with no outstanding issues.
  - **Use modern Python 3.12+ built-in generic types** (list[T], dict[K,V]) instead of typing
    module equivalents (List[T], Dict[K,V]).
- **Type annotations are not optional:**
  - Every new or modified function, method, and class must include complete type hints.
  - All code must be reviewed and updated to maintain full type coverage.
  - Use built-in generics: `list[str]`, `dict[str, int]`, `tuple[int, ...]`, `set[str]`
- **No code may be committed unless it passes Black, Ruff, and basedpyright.**
- **Motivation:**
  - This rule guarantees code consistency, readability, and robust static analysis, reducing bugs
    and onboarding time.
  - Modern built-in generics are more performant and align with Python 3.12+ best practices.
