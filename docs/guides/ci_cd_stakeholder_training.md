# CI/CD and Quality Assurance: Stakeholder Training

**Document Version**: 1.0
**Last Updated**: July 14, 2025
**Scope**: Overview of automated quality processes for developers and reviewers.

## Introduction

This document provides a high-level overview of the newly implemented CI/CD and quality assurance
processes in the CrackSeg project. The goal of these automated systems is to improve code quality,
maintain high test coverage, and ensure project stability.

For more detailed information, please refer to the [Test Maintenance Procedures](test_maintenance_procedures.md).

## 1. The Automated Quality Gates

On every Pull Request to `main` or `develop`, a series of automated checks are performed. A PR
cannot be merged unless all checks pass.

### What is Checked?

1. **Code Quality**: `black`, `ruff`, and `basedpyright` ensure that all code is correctly
    formatted, lint-free, and type-safe.
2. **Test Coverage**: The test suite is run, and code coverage is measured. This is the most
    important check for developers to understand.
3. **Performance Benchmarks**: Key performance metrics are tracked to prevent regressions.

The full configuration can be seen in [`.github/workflows/quality-gates.yml`](../../.github/workflows/quality-gates.yml).

## 2. Understanding the Coverage Check

We have a strict **80% test coverage** requirement. This is enforced in two ways:

1. **Overall Project Coverage**: The entire project must maintain at least 80% coverage.
2. **New Code Coverage (Patch Coverage)**: Any new code submitted in a PR must *itself* be at
    least 80% covered. This is the most common reason a PR might be blocked.

### What to Do When Coverage Fails

If your PR is blocked due to a coverage failure, please follow these steps:

1. **Look at the PR comments**: An automated comment from Codecov will pinpoint which lines of your
    new code are not being tested.
2. **Run the coverage report locally**:

    ```bash
    conda activate crackseg
    pytest --cov=src --cov-report=html
    ```

    Open `htmlcov/index.html` to see the full report.
3. **Add more tests**: Write new tests to cover the untested lines in your code.

For a detailed guide, see the [Troubleshooting Coverage Failures](test_maintenance_procedures.md#troubleshooting-coverage-failures)
section in the maintenance guide.

## 3. Weekly Quality Reports

Every Monday, an automated workflow generates a series of quality reports, including:

* A full HTML coverage report.
* Code complexity and maintainability reports.

These reports are stored as artifacts in the [Weekly Quality Report workflow runs](https://github.com/your-org/your-repo/actions/workflows/weekly-quality-report.yml).

## 4. Documentation Deployment

All documentation in the `docs/` directory is automatically built and deployed as a website. This
ensures that our documentation is always up-to-date and easily accessible.

---

**Questions?** Please reach out to the development team or open an issue for clarification.
