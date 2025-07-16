---
description: Guidelines for continuously improving Roo Code rules based on emerging code patterns and best practices.
globs: **/*
alwaysApply: true
---
# Rule Improvement and Evolution Guidelines

These guidelines establish a systematic approach for continuously improving and evolving Roo Code rules based on emerging patterns, best practices, and project evolution.

## Rule Improvement Triggers

- **Pattern Recognition Indicators:**
  - New code patterns not covered by existing rules appearing in 3+ files
  - Repeated similar implementations across modules
  - Common error patterns that could be prevented by rules
  - New libraries or tools being used consistently
  - Emerging best practices in machine learning or PyTorch development

- **Quality and Maintenance Signals:**
  - Code reviews repeatedly mentioning the same feedback
  - Linting errors that could be prevented by rules
  - Test patterns that need standardization
  - Performance or security patterns requiring documentation

## Analysis Process

- **Code Pattern Analysis:**
  - Compare new implementations with existing rules in [coding-preferences.md](mdc:.roo/rules/coding-preferences.md)
  - Identify standardizable patterns in ML model architectures
  - Monitor data loading and preprocessing consistency
  - Check for evolving testing patterns in [testing-standards.md](mdc:.roo/rules/testing-standards.md)

- **Quality Assessment:**
  - Verify alignment with [workflow-preferences.md](mdc:.roo/rules/workflow-preferences.md)
  - Check consistency with type annotation requirements
  - Ensure Black, Ruff, and basedpyright compliance patterns
  - Review integration with Task Master workflow patterns

## Rule Update Strategies

- **Add New Rules When:**
  - PyTorch/ML patterns emerge in 3+ model components
  - Common crack segmentation domain patterns require standardization
  - New configuration or data handling patterns appear
  - Security or performance patterns for ML pipelines emerge
  - Example:
    ```python
    # If this pattern appears in multiple model files:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(f"Expected 4D tensor with 3 channels, got {x.shape}")
        return self._forward_impl(x)

    # Consider adding to coding-preferences.md:
    # - Standard input validation patterns for models
    # - Consistent error message formats
    ```

- **Modify Existing Rules When:**
  - Better examples exist in the current codebase
  - Edge cases are discovered during development
  - Related rules in the project have been updated
  - ML framework best practices evolve

## Specific Pattern Categories

- **Model Architecture Patterns:**
  - Consistent encoder-decoder interfaces
  - Standard tensor shape validation
  - Common activation and normalization patterns
  - Error handling for model components

- **Data Pipeline Patterns:**
  - Transform composition and configuration
  - DataLoader setup and optimization
  - Path handling and file I/O consistency
  - Validation and preprocessing standardization

- **Training and Evaluation Patterns:**
  - Loss function implementation standards
  - Metric calculation consistency
  - Checkpoint saving and loading patterns
  - Logging and monitoring approaches

## Rule Quality Maintenance

- **Quality Checks:**
  - Rules should include concrete Python/PyTorch examples
  - Examples should come from actual project code
  - References should point to existing project files
  - Patterns should be enforceable via basedpyright, Black, and Ruff

- **Documentation Synchronization:**
  - Keep examples aligned with current project structure
  - Update references to match [project-structure.md](mdc:.roo/guides/project-structure.md)
  - Maintain cross-references between related rules
  - Document changes in rule evolution

## Implementation Process

- **Professional Solution Analysis:**
  - Before adding or modifying rules, analyze at least 3 different approaches
  - Consider: (1) New standalone rule, (2) Extension of existing rule, (3) Integration into workflow
  - Document reasoning in rule updates and commit messages
  - Ensure chosen approach is most maintainable and enforceable

- **Integration Workflow:**
  ```bash
  # Rule update process aligned with quality standards
  # 1. Identify pattern
  # 2. Draft rule update
  # 3. Validate with quality tools
  black .roo/rules/
  ruff .roo/rules/ --fix
  # 4. Test rule effectiveness on existing code
  # 5. Commit with descriptive message
  git add .roo/rules/ && git commit -m "rules: Add ML model validation patterns"
  ```

## Continuous Monitoring

- **Development Feedback Loops:**
  - Monitor Task Master logs for recurring implementation challenges
  - Track common questions during development sessions
  - Review pull request feedback for rule gaps
  - Assess rule effectiveness through code quality metrics

- **Rule Evolution Tracking:**
  - Document major rule changes in project changelog
  - Maintain rule versioning for significant updates
  - Create migration guides for breaking rule changes
  - Archive deprecated patterns with replacement guidance

## Rule Deprecation Process

- **Deprecation Criteria:**
  - Patterns no longer relevant to current ML stack
  - Rules superseded by better practices
  - Framework changes making rules obsolete
  - Project evolution rendering rules unnecessary

- **Deprecation Steps:**
  - Mark rules as deprecated with clear notation
  - Provide migration path to new patterns
  - Update references in dependent rules
  - Remove after deprecation period with project stakeholder approval

## References and Integration

- **Core Rule Dependencies:**
  - **Code Quality**: [coding-preferences.md](mdc:.roo/rules/coding-preferences.md)
  - **Development Process**: [workflow-preferences.md](mdc:.roo/rules/workflow-preferences.md)
  - **Testing Standards**: [testing-standards.md](mdc:.roo/rules/testing-standards.md)
  - **Project Structure**: [project-structure.md](mdc:.roo/guides/project-structure.md)

- **Rule Format Reference:**
  - Follow [cursor_rules.md](mdc:.roo/rules/cursor_rules.md) for proper structure
  - Maintain consistency with established pattern formatting
  - Use `mdc:` links for internal cross-references
  - Include concrete examples from project domain