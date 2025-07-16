---
description: Guidelines for creating and maintaining Roo Code rules to ensure consistency and effectiveness.
globs: .roo/rules/*.md
alwaysApply: true
---
# Roo Code Rule Creation Guidelines

These meta-guidelines ensure consistency, quality, and effectiveness across all Roo Code rules in
the project.

## Required Rule Structure

- **Standard Rule Header:**

  ```markdown
  ---
  description: Clear, one-line description of what the rule enforces
  globs: path/to/files/*.ext, other/path/**/*
  alwaysApply: boolean
  ---
  ```

- **Content Organization:**

  ```markdown
  # Rule Title

  Brief introduction explaining the rule's purpose.

  ## Section 1
  - **Main Points in Bold**
    - Sub-points with details
    - Examples and explanations

  ## Section 2
  - **Another Main Point**
    - Supporting details
  ```

## Content Guidelines

- **Rule Quality Standards:**
  - Start with high-level overview explaining the rule's purpose
  - Include specific, actionable requirements that can be followed immediately
  - Provide concrete examples from the actual project codebase
  - Use consistent formatting and structure across all rules
  - Keep rules DRY by cross-referencing related rules

- **Code Examples:**
  - Use language-specific code blocks with proper syntax highlighting
  - Include both positive (✅ DO) and negative (❌ DON'T) examples
  - Ensure examples are relevant to the project domain (ML/PyTorch/crack segmentation)
  - Example format:

    ```python
    # ✅ Good: Clear tensor shape validation
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {x.shape[1]}")
        return self._process(x)

    # ❌ Bad: No validation or unclear error messages
    def forward(self, x):
        return self._process(x)
    ```

## File References and Links

- **Internal Rule References:**
  - Use `[rule-name.md](mdc:.roo/rules/rule-name.md)` format
  - Example: [coding-preferences.md](mdc:.roo/rules/coding-preferences.md)

- **Project File References:**
  - Use `[filename](mdc:path/to/file)` for project files
  - Example: [project-structure.md](mdc:.roo/guides/project-structure.md)

- **Cross-Reference Strategy:**
  - Create a web of interconnected rules that support each other
  - Reference specific sections when relevant
  - Maintain bidirectional references where appropriate

## Rule Categories and Organization

- **Core Rule Types:**
  - **Code Quality**: [coding-preferences.md](mdc:.roo/rules/coding-preferences.md) - Technical standards
  - **Development Process**: [workflow-preferences.md](mdc:.roo/rules/workflow-preferences.md) - Methodology
  - **Testing Standards**: [testing-standards.md](mdc:.roo/rules/testing-standards.md) - Quality assurance
  - **Git Standards**: [git-standards.md](mdc:.roo/rules/git-standards.md) - Version control practices
  - **Meta-Rules**: [self_improve.md](mdc:.roo/rules/self_improve.md) - Rule evolution

- **Specialized Rules:**
  - Domain-specific patterns (ML, data processing, etc.)
  - Tool-specific guidelines (Task Master, PyTorch, etc.)
  - Project-specific conventions

## Rule Maintenance Process

- **Regular Review Criteria:**
  - Update examples when codebase patterns evolve
  - Add new examples from actual implementation
  - Remove outdated patterns or deprecated practices
  - Ensure cross-references remain valid

- **Quality Assurance:**
  - Rules should be testable and enforceable
  - Examples should compile/run successfully
  - References should point to existing files
  - Formatting should be consistent across all rules

- **Evolution Integration:**
  - Follow [self_improve.md](mdc:.roo/rules/self_improve.md) for systematic updates
  - Document significant rule changes in commit messages
  - Maintain backward compatibility when possible
  - Provide migration guidance for breaking changes

## Best Practices for Rule Writing

- **Clarity and Actionability:**
  - Use clear, imperative language ("Use X", "Avoid Y")
  - Provide specific steps or commands where applicable
  - Include concrete examples over abstract descriptions
  - Make rules immediately applicable by developers

- **Project Context Integration:**
  - Tailor examples to crack segmentation domain
  - Reference actual project tools (PyTorch, basedpyright, Black, Ruff)
  - Align with project architecture and patterns
  - Consider ML/deep learning specific concerns

- **Professional Standards:**
  - Follow the three-option analysis approach for significant decisions
  - Justify rule choices with clear reasoning
  - Maintain consistency with industry best practices
  - Document edge cases and exceptions clearly

## References and Integration

- **Core Project Rules:**
  - **Quality Standards**: [coding-preferences.md](mdc:.roo/rules/coding-preferences.md)
  - **Development Workflow**: [workflow-preferences.md](mdc:.roo/rules/workflow-preferences.md)
  - **Testing Guidelines**: [testing-standards.md](mdc:.roo/rules/testing-standards.md)
  - **Git Practices**: [git-standards.md](mdc:.roo/rules/git-standards.md)
  - **Continuous Improvement**: [self_improve.md](mdc:.roo/rules/self_improve.md)

- **Project Documentation:**
  - **Architecture**: [structural-guide.md](mdc:.roo/guides/structural-guide.md)
  - **Project Structure**: [project-structure.md](mdc:.roo/guides/project-structure.md)
  - **Development Guide**: [development-guide.md](mdc:.roo/guides/development-guide.md)
