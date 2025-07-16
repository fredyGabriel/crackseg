---
description:
globs:
alwaysApply: true
---
# Development Workflow Guidelines

These guidelines establish a systematic approach to code development, emphasizing planning, quality, and responsible integration with the existing codebase.

## Planning and Analysis

- **Thorough Planning Before Implementation:**
  - Understand task requirements and project context before coding
  - Review [structural-guide.md](mdc:.roo/guides/structural-guide.md) for architectural patterns
  - Outline approach and integration strategy with existing codebase
  - For significant implementations, provide 2-3 paragraph summary explaining chosen approach and reasoning

- **Analyze Three Solution Options:**
  - Before making significant technical decisions (refactoring, architecture changes, test adaptations)
  - Clearly justify the chosen solution as the most professional, robust, and maintainable
  - Document reasoning in implementation logs, commit messages, or rule updates
  - This ensures thoughtful, high-quality decisions and prevents arbitrary fixes

## Implementation Principles

- **Prioritize Simple Solutions:**
  - Choose the simplest and most direct solution to problems
  - Avoid introducing unnecessary complexity or over-engineering

- **Maintain Focused Scope:**
  - Limit modifications strictly to areas directly relevant to the current task
  - **Do not modify unrelated code** to avoid unintended side effects
  - Consider impact on other modules, methods, and functionality

- **Iterative Improvement:**
  - Integrate with and improve existing patterns in the relevant area
  - Reference [structural-guide.md](mdc:.roo/guides/structural-guide.md) for consistency
  - Only introduce new patterns after existing options are exhausted and with explicit confirmation
  - Remove old or superseded logic after new implementation is in place

## Quality Assurance

- **Code Quality Integration:**
  - Follow [coding-preferences.md](mdc:.roo/rules/coding-preferences.md) standards
  - Ensure all code passes `basedpyright .`, `black .`, and `ruff .` before commit
  - Write comprehensive type annotations for all new code

- **Testing During Development:**
  - Write unit and integration tests for major functionality and components
  - Aim for >80% test coverage on core functionality
  - Focus test changes primarily on test files and units under test
  - **For external modifications needed for testability:**
    ```python
    # ✅ Prefer dependency injection
    class ModelTrainer:
        def __init__(self, loss_function: LossFunction) -> None:
            self.loss_fn = loss_function  # Injected, not created internally

    # ✅ Use mocking for external dependencies
    @patch('src.model.encoder.SwinEncoder')
    def test_model_integration(self, mock_encoder):
        mock_encoder.return_value.forward.return_value = torch.zeros(1, 256, 32, 32)
    ```

## Project Structure and Documentation

- **Respect Project Structure:**
  - Follow directory organization in [project-structure.md](mdc:.roo/guides/project-structure.md)
  - **Always request user confirmation** before adding, deleting, or restructuring files/directories
  - Update [project-structure.md](mdc:.roo/guides/project-structure.md) with status markers (✅, `[x]`, `(DONE)`) as implementation progresses

- **Artifact Cross-Referencing:**
  - All dependent tasks must explicitly reference artifacts from prerequisites
  - Include paths and filenames in task `details` or `description`
  - Example:
    ```markdown
    ## References to Previous Artifacts
    - outputs/analysis/model_evaluation_report.md
    - outputs/test_results/unit_test_coverage.json
    ```

## Error Resolution and Communication

- **Resolve Errors Directly:**
  - Address root causes rather than masking symptoms
  - Understand and fix linting, type checking, and test errors
  - Document resolution process for complex errors
  - Only use workarounds for third-party library issues with clear documentation

- **Communicate Ambiguities:**
  - Seek clarification on unclear requirements, code, or guidelines before implementation
  - Identify potential problems, conflicts, or alternative approaches early
  - Reference specific files and line numbers when discussing issues

## Workflow Integration

- **Pre-Commit Checklist:**
  ```bash
  # Quality gates (from coding-preferences.md)
  black .
  ruff . --fix
  basedpyright .

  # Testing
  pytest tests/ --cov=src --cov-report=term-missing

  # Only commit if all pass
  git add . && git commit -m "feat(module): Description"
  ```

- **Task Management:**
  - Use Task Master for project organization following [dev_workflow.md](mdc:.roo/rules/dev_workflow.md)
  - Update task status and add implementation notes during development
  - Reference completed artifacts in dependent tasks

## Professional Solution Analysis

- **Before making a significant technical decision (e.g., refactoring, rule update, pattern change, or test adaptation), you must analyze at least 3 possible options.**
    - Clearly justify the chosen solution, selecting the most professional, robust, and maintainable approach.
    - Document the reasoning in the implementation log, commit message, or rule update as appropriate.
    - This ensures thoughtful, high-quality decisions and prevents arbitrary or rushed fixes.
    - This principle applies to all code, rules, and workflow improvements, not just tests.

## References

- **Code Quality**: [coding-preferences.md](mdc:.roo/rules/coding-preferences.md)
- **Project Structure**: [project-structure.md](mdc:.roo/guides/project-structure.md)
- **Architecture**: [structural-guide.md](mdc:.roo/guides/structural-guide.md)
- **Task Management**: [dev_workflow.md](mdc:.roo/rules/dev_workflow.md)
- **Self-Improvement**: [self_improve.md](mdc:.roo/rules/self_improve.md)