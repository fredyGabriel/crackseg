# Exact Duplication Mapping

**Consolidation Phase 1**: Precise identification of duplicated content

## 🔍 Duplications in `always_applied_workspace_rules`

### 1. **Code Quality Standards** (COMPLETE DUPLICATION)

**Source**: `coding-preferences.mdc` lines 1-200+
**Duplicated in**: `always_applied_workspace_rules` ~300 lines

**Exact duplicated content:**

- Section "Code Quality Standards (Mandatory)"
- Type checking rules: `basedpyright .`
- Formatting: `black .`
- Linting: `ruff .`
- "Type annotations are mandatory for all code"
- Section "Modern Generic Type Syntax (Python 3.12+ PEP 695)"
- "Built-in Generic Types (Python 3.9+ PEP 585)"
- Identical code examples
- "Pre-commit workflow" with exact bash commands
- Section "Code Structure and Organization"
- "Documentation and Comments"
- "Reliability and Error Handling"
- "Configuration and Dependencies"
- "Advanced Type Patterns for Python 3.12+"

### 2. **Development Workflow Guidelines** (MASSIVE DUPLICATION)

**Source**: `workflow-preferences.mdc` lines 1-125
**Duplicated in**: `always_applied_workspace_rules` ~200 lines

**Exact duplicated content:**

- "Development Workflow Guidelines" title and description
- "Planning and Analysis" full section
- "Analyze Three Solution Options" principle
- "Implementation Principles" full section
- "Quality Assurance" section
- "Project Structure and Documentation"
- "Error Resolution and Communication"
- "Workflow Integration" with Pre-Commit Checklist
- "Professional Solution Analysis" full paragraph

### 3. **Task Master Development Workflow** (MEGA DUPLICATION)

**Source**: `dev_workflow.mdc` lines 1-210
**Duplicated in**: `always_applied_workspace_rules` ~400 lines

**Exact duplicated content:**

- "Task Master Development Workflow" full title
- "Primary Interaction: MCP Server vs. CLI" full section
- "Standard Development Workflow Process" full list of ~20 items
- "Task Complexity Analysis" section
- "Task Breakdown Process"
- "Implementation Drift Handling"
- "Task Status Management"
- "Task Structure Fields" with all examples
- "Environment Variables Configuration" full list
- "Determining the Next Task"
- "Viewing Specific Task Details"
- "Managing Task Dependencies"
- "Iterative Subtask Implementation" 10-step process

### 4. **Cursor Rule Creation Guidelines** (PARTIAL DUPLICATION)

**Source**: `cursor_rules.mdc` lines 1-144
**Duplicated in**: `always_applied_workspace_rules` ~100 lines

**Duplicated content:**

- "Cursor Rule Creation Guidelines" principles
- "Required Rule Structure" standards
- "Content Guidelines" with code examples
- "Rule Categories and Organization"

## 📊 Duplication Metrics

| Source File | Source Lines | Duplicated Lines | % Duplication | Impact |
|-------------|-------------|------------------|--------------|--------|
| `coding-preferences.mdc` | 244 | ~300 | 123% | CRITICAL |
| `workflow-preferences.mdc` | 125 | ~200 | 160% | HIGH |
| `dev_workflow.mdc` | 210 | ~400 | 190% | MEGA |
| `cursor_rules.mdc` | 144 | ~100 | 69% | MODERATE |
| **TOTAL** | **723** | **~1000** | **138%** | **MASSIVE** |

## 🎯 Content that MUST remain in `always_applied_workspace_rules`

### Critical Minimal Rules (3-4 lines max)

```txt
- **Python code**: Must pass basedpyright, black, and ruff before commit
- **Type annotations**: Mandatory using modern Python 3.12+ generics
- **References**: See [consolidated-workspace-rules.mdc](mdc:.cursor/rules/consolidated-workspace-rules.mdc)
- **Respond in Spanish**: Use English only for code
```

### Everything else → REMOVE and reference specific files

## 🚀 Elimination Plan

1. **Reduce `always_applied_workspace_rules`** from ~1000 lines to ~50 lines
2. **Remove internal duplications** between .mdc files
3. **Optimize references** in `consolidated-workspace-rules.mdc`
4. **Validate complete navigation**

**Target reduction**: **95% less duplicated content**
