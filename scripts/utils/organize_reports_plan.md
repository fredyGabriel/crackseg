# Reports Reorganization Plan - CrackSeg

## Current Situation

### Scattered Report Locations

1. **docs/reports/** - Well-organized structure with subfolders
2. **scripts/reports/** - Utility scripts and example files mixed
3. **.taskmaster/reports/** - **KEEP INTACT** - Task Master may need these files
4. **.taskmaster/.taskmaster/reports/** - **KEEP INTACT** - Task Master internal structure

## Reorganization Proposal

### 1. Unified Structure

**Centralize documentation in `docs/reports/` while respecting Task Master structure**

```txt
docs/reports/
├── README.md                    # Keep
├── organization_summary.md     # Keep
├── documentation_checklist.md  # Keep
├── project/                     # Keep - general project reports
├── tasks/                       # Keep - Task Master reports (reference copies)
├── models/                      # Keep - model analysis
├── testing/                     # Keep - testing reports
├── coverage/                    # Keep - coverage reports
├── analysis/                    # Keep - various analysis
├── archive/                     # Keep - historical reports
└── scripts/                     # NEW - scripts and examples
```

### 2. Proposed Movements

#### A. From `scripts/reports/` → `docs/reports/scripts/`

- ✅ `example_prd.txt` → `docs/reports/scripts/example_prd.txt` (ALREADY MOVED)
- ✅ `hydra_examples.txt` → `docs/reports/scripts/hydra_examples.txt` (ALREADY MOVED)

#### B. From `scripts/reports/` → `docs/reports/models/`

- Check for duplicated model JSON files and consolidate only if necessary

#### C. ~~From `.taskmaster/reports/` → `docs/reports/tasks/`~~

- **CANCELLED** - Keep `.taskmaster/` intact for Task Master compatibility
- Only keep reference copies in `docs/reports/tasks/` if they already exist

#### D. ~~Clean duplicates~~

- **CANCELLED** - Don't delete `.taskmaster/.taskmaster/reports/` (Task Master may need it)

### 3. Analysis Scripts

**Keep in `scripts/reports/` only analysis tools:**

- `model_imports_autofix.py`
- `model_imports_validation.py`
- `model_imports_catalog.py`
- `model_imports_cycles.py`
- `model_pyfiles_inventory.py`
- `compare_model_structure.py`
- `autofix_backups/` (folder)

### 4. Advantages of this Organization

1. **Centralization**: One place to find documentation reports
2. **Clear separation**: Tools vs results
3. **Compatibility**: Respects Task Master needs
4. **Logical structure**: Maintains existing categories
5. **No conflicts**: Doesn't interfere with tool functionality

### 5. Implementation Steps

1. ✅ Create `docs/reports/scripts/`
2. ✅ Move example and configuration files
3. **SKIP** - Consolidate Task Master reports (keep original locations)
4. **SKIP** - Clean .taskmaster duplicates
5. Update references in documentation
6. Verify scripts continue working

### 6. Files to Keep in Current Locations

- **`.roomodes`** - KEEP in root
- **`.windsurfrules`** - KEEP in root
- **`.taskmaster/`** - **KEEP INTACT** - Task Master may need this structure
- **Analysis scripts in `scripts/reports/`** - They are tools, not reports

### 7. Special Considerations

- ⚠️ **IMPORTANT**: Don't modify `.taskmaster/` structure to avoid compatibility issues
- Task Master scripts will continue generating reports in `.taskmaster/reports/`
- `docs/reports/tasks/` may contain reference copies for documentation
- Keep .gitignore updated to ignore temporary reports

## Expected Result

A cleaner structure where:

- **docs/reports/** = All final reports and documentation (without interfering with Task Master)
- **scripts/reports/** = Only analysis tools and utility scripts
- **.taskmaster/** = **INTACT** - Complete structure preserved for Task Master
