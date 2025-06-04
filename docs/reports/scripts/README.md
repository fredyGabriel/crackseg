# Scripts and Examples - Reports

This folder contains example files and configuration related to reports and documentation for the CrackSeg project.

## Contents

### Example Files

- **`example_prd.txt`** - Example Product Requirements Document for Task Master
  - Serves as template for creating new PRDs
  - Compatible with `task-master parse-prd`

- **`hydra_examples.txt`** - Hydra command-line override examples
  - Shows how to modify configurations from command line
  - Useful for experimentation and debugging

## Usage

### For Task Master

```bash
# Use the example PRD to initialize a project
task-master parse-prd --input=docs/reports/scripts/example_prd.txt
```

### For Hydra

```bash
# View override examples
cat docs/reports/scripts/hydra_examples.txt
```

## Notes

- These files were moved from `scripts/reports/` to maintain a clear separation between analysis tools and documentation
- Analysis scripts remain in `scripts/reports/` as they are development tools, not documentation
- The `.taskmaster/` structure is kept intact for compatibility

## See Also

- [scripts/reports/](../../../scripts/reports/) - Model and import analysis tools
- [docs/reports/tasks/](../tasks/) - Task Master reports
- [docs/reports/](../) - Main reports index
