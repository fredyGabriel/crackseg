---
description:
globs:
alwaysApply: false
---
# Git Standards and Practices

These standards ensure consistent, professional Git usage throughout the project development lifecycle.

## Commit Message Standards

- **All commit messages must be written in clear, correct English:**
  - This applies to all commits, including documentation, code, configuration, and rule updates
  - Both subject line and commit body must be in English
  - Use conventional commit format for consistency with project standards
  - Example:
    ```bash
    git commit -m "feat(model): Add Swin Transformer encoder implementation

    - Implement SwinEncoder class with configurable parameters
    - Add comprehensive type annotations and docstrings
    - Include unit tests with >90% coverage
    - Integrate with existing model factory pattern"
    ```

- **Conventional Commits Format:**
  - Use structured commit messages: `type(scope): description`
  - Common types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
  - Scope should reference project modules: `model`, `data`, `training`, `evaluation`
  - Examples:
    ```bash
    # ✅ Good commit messages
    git commit -m "feat(encoder): Implement Swin Transformer backbone"
    git commit -m "fix(loss): Correct Dice loss calculation for edge cases"
    git commit -m "test(data): Add integration tests for DataLoader pipeline"
    git commit -m "docs(api): Update model configuration documentation"

    # ❌ Avoid these patterns
    git commit -m "arreglo bug en el encoder"  # Non-English
    git commit -m "fix stuff"  # Too vague
    git commit -m "WIP"  # Work in progress without context
    ```

## Branch Management

- **Branch Naming Conventions:**
  - Use descriptive, English branch names with hyphens
  - Include issue/task numbers when applicable
  - Format: `feature/task-id-brief-description` or `fix/issue-description`
  - Examples:
    ```bash
    # ✅ Good branch names
    feature/task-15-swin-encoder-implementation
    fix/dice-loss-nan-handling
    refactor/model-factory-simplification
    docs/api-documentation-update

    # ❌ Avoid these patterns
    feature/tarea-15-encoder  # Mixed languages
    fix-bug  # Too generic
    my-branch  # Non-descriptive
    ```

- **Branch Workflow:**
  - Create feature branches from `main` or `develop`
  - Keep branches focused on single features or fixes
  - Use pull requests for code review before merging
  - Delete feature branches after successful merge

## Code Quality Integration

- **Pre-Commit Requirements:**
  ```bash
  # Ensure code quality before commits (from coding-preferences.md)
  black .
  ruff . --fix
  basedpyright .
  pytest tests/ --cov=src --cov-report=term-missing

  # Only commit if all quality gates pass
  git add . && git commit -m "feat(module): Description"
  ```

- **Commit Content Standards:**
  - Include only related changes in each commit
  - Avoid committing generated files or temporary artifacts
  - Update documentation when changing public APIs
  - Add tests for new functionality in the same commit when possible

## Collaboration Standards

- **Pull Request Requirements:**
  - Provide clear description of changes and rationale
  - Reference related issues or task numbers
  - Include testing information and coverage reports
  - Request review from appropriate team members

- **Merge Strategies:**
  - Use squash merges for feature branches to maintain clean history
  - Preserve commit history for collaborative development branches
  - Include comprehensive commit message when squashing

## Integration with Project Workflow

- **Task Master Integration:**
  - Reference task IDs in commit messages when applicable
  - Update task status through Git hooks or manual process
  - Link commits to specific implementation phases

- **Quality Gate Enforcement:**
  - Follow [coding-preferences.md](mdc:.roo/rules/coding-preferences.md) for all committed code
  - Ensure [testing-standards.md](mdc:.roo/rules/testing-standards.md) compliance
  - Apply [workflow-preferences.md](mdc:.roo/rules/workflow-preferences.md) process

## Internationalization Considerations

- **English-First Policy Rationale:**
  - Ensures accessibility for international collaborators
  - Maintains consistency with code comments and documentation
  - Facilitates code review and project maintenance
  - Aligns with industry standards for open source projects

- **Exceptions and Flexibility:**
  - If non-English content is required for legal/compliance reasons, include English translation
  - Domain-specific terms may retain original language with English explanation
  - User-facing content may be localized, but development artifacts remain in English

## References

- **Related Standards**: [coding-preferences.md](mdc:.roo/rules/coding-preferences.md)
- **Development Process**: [workflow-preferences.md](mdc:.roo/rules/workflow-preferences.md)
- **Testing Integration**: [testing-standards.md](mdc:.roo/rules/testing-standards.md)
- **Rule Evolution**: [self_improve.md](mdc:.roo/rules/self_improve.md)




