# Legacy Folders Reorganization Plan - CrackSeg

## Current Situation

### Problematic Folders Identified

1. **`MagicMock/`** - Contains test mock data with unclear naming
   - Has experiment manager mock directories with numeric IDs
   - Structure suggests testing artifacts that should be in tests/fixtures/
   - Currently scattered in project root

2. **`old_stuff/`** - Contains archived documentation and scripts
   - `archived_scripts/` with valuable documentation files
   - Contains `.mdc` files that appear to be project guides
   - Poor naming convention for a professional project

## Reorganization Proposal

### 1. MagicMock/ → tests/fixtures/mocks/

**Rationale**: Test-related artifacts belong in the test directory

```txt
tests/fixtures/mocks/
├── experiment_manager/
│   ├── experiment_dirs/        # Mock experiment directories
│   │   ├── 2254896334992/      # Keep existing IDs for consistency
│   │   ├── 2254896338784/
│   │   └── ...
│   └── experiment_paths/       # Mock path operations
│       ├── truediv_operations/
│       └── ...
└── README.md                   # Document mock structure and usage
```

### 2. old_stuff/ → archive/

**Rationale**: More professional naming for archived content

```txt
archive/
├── legacy_docs/               # Renamed from archived_scripts/
│   ├── project-structure.mdc  # Keep valuable documentation
│   ├── structural-guide.mdc
│   ├── development-guide.mdc
│   └── README.md              # Document archive purpose
└── README.md                  # Main archive documentation
```

## Implementation Strategy

### Phase 1: Create New Structure

1. **Create tests/fixtures/ if not exists**

   ```bash
   mkdir -p tests/fixtures/mocks/experiment_manager
   ```

2. **Create archive/ structure**

   ```bash
   mkdir -p archive/legacy_docs
   ```

### Phase 2: Move Content

1. **Move MagicMock content**

   ```bash
   # Move to appropriate test fixtures location
   mv MagicMock/* tests/fixtures/mocks/experiment_manager/
   ```

2. **Move old_stuff content**

   ```bash
   # Move to professional archive location
   mv old_stuff/archived_scripts/* archive/legacy_docs/
   ```

### Phase 3: Documentation

1. **Create README files** explaining:
   - Purpose of mock fixtures
   - How to use the mocks in tests
   - Archive content and retrieval process

2. **Update .gitignore** if necessary:
   - Ensure mock data is properly handled
   - Archive content versioning decisions

### Phase 4: Cleanup

1. **Remove empty directories**

   ```bash
   rmdir MagicMock/
   rmdir old_stuff/archived_scripts/
   rmdir old_stuff/
   ```

2. **Update any references** in code or documentation

## Benefits

### For MagicMock/ → tests/fixtures/mocks/

1. **Logical Organization**: Test artifacts with tests
2. **Clear Purpose**: Mock data clearly identified
3. **Maintainability**: Easier to manage test fixtures
4. **Convention**: Follows standard Python project structure

### For old_stuff/ → archive/

1. **Professional Naming**: Better project presentation
2. **Clear Intent**: Archive purpose is evident
3. **Preservation**: Valuable documentation preserved
4. **Accessibility**: Easy to find and retrieve archived content

## Considerations

### Mock Data

- **Preserve structure**: Keep existing directory names for test compatibility
- **Document usage**: Clear README on how mocks are used
- **Test integration**: Ensure tests still find mock data

### Archived Documentation

- **Review content**: Determine if any should be restored to active docs
- **Index creation**: Catalog what's archived for easy retrieval
- **Migration notes**: Document why files were archived

## Risk Mitigation

1. **Backup first**: Copy folders before moving
2. **Test verification**: Run tests after moving mocks
3. **Gradual implementation**: Move in phases
4. **Rollback plan**: Keep backups until verification complete

## Final Structure

```txt
project_root/
├── tests/
│   └── fixtures/
│       └── mocks/
│           └── experiment_manager/    # Former MagicMock content
├── archive/
│   └── legacy_docs/                  # Former old_stuff content
└── ... (rest of project)
```

---

**Expected Result**: Clean project root with properly organized test fixtures and professional archive structure.
