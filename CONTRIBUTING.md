# Contributing Guide

Thank you for contributing to CrackSeg. This guide summarizes the workflow and required quality gates.

## Workflow

1. Create a branch
   - feature/<short-title>
   - fix/<short-title>
   - chore/<short-title>
2. Implement changes following rules in `.cursor/rules/`
3. Run quality gates locally
4. Write/update tests
5. Open a PR with a clear description and links to tasks

## Quality Gates (must pass)

```bash
black .
python -m ruff check . --fix
basedpyright .
pytest -q
```

## Docs

- README: project overview and quickstart
- Training workflow: `docs/guides/operational-guides/workflows/legacy/WORKFLOW_TRAINING.md`
- Coding standards: `.cursor/rules/coding-standards.mdc`
- ML/PyTorch standards: `.cursor/rules/ml-pytorch-standards.mdc`

## Commit messages

- Conventional Commits (e.g., `feat(model): add hybrid decoder block`)
- Keep PRs small; include before/after notes for refactors

## Contact

Open an issue with a minimal repro if you need help.
