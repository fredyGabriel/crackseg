# Architectural Decision Records (ADR)

## ADR-001: Replace Graphviz with Matplotlib for Model Architecture Visualization

**Date:** January 2025
**Status:** Accepted
**Context:** CrackSeg Project Environment Optimization

### Decision

We have decided to **replace Graphviz with Matplotlib** as the primary tool for model architecture
visualization in the CrackSeg project.

### Context

During environment setup for PyTorch 2.7 + CUDA 12.9 on Windows, we encountered significant
compilation issues with Graphviz:

1. **gdk-pixbuf Dependencies**: Graphviz installation via conda-forge requires `gdk-pixbuf`, which
  has complex compilation dependencies on Windows
2. **Build Tool Conflicts**: Multiple Visual Studio versions (2019/2022) caused path length issues
  and compilation failures
3. **Environment Complexity**: The build process significantly increased environment creation time
  and failure rate

### Analysis

**Graphviz Usage Assessment:**

- **Primary Use**: Model architecture diagram generation (U-Net visualization)
- **Frequency**: Development/debugging tool, not runtime critical
- **Criticality**: Nice-to-have feature, not essential for core ML functionality

**Alternative Evaluation:**

- **Matplotlib**: Already available in environment, supports custom graph layouts
- **Plotly**: Available for interactive visualizations, already used in test reporting
- **NetworkX**: Could be added easily for complex graph layouts if needed

### Decision Rationale

1. **Stability First**: Prioritize reliable environment creation over auxiliary features
2. **Existing Tools**: Leverage matplotlib which is already essential for data visualization
3. **Maintenance Cost**: Reduce complexity in build process and dependencies
4. **Functionality Preservation**: Maintain visualization capability with simpler implementation

### Implementation Strategy

**Phase 1**: Environment Stabilization

- Keep graphviz commented out in `environment.yml`
- Document the decision in configuration files
- Ensure core ML pipeline works without graphviz

**Phase 2**: Alternative Implementation

- Implement matplotlib-based architecture visualization
- Create fallback mechanism for existing graphviz code
- Update GUI components to handle new visualization format

**Phase 3**: Migration

- Update all references to prefer matplotlib implementation
- Maintain graphviz compatibility for users who have it installed
- Update documentation and examples

### Consequences

**Positive:**

- ✅ Faster, more reliable environment setup
- ✅ Reduced dependency complexity
- ✅ Better Windows compatibility
- ✅ Leverage existing matplotlib expertise
- ✅ Maintain visualization functionality

**Negative:**

- ⚠️ Custom implementation needed for complex graph layouts
- ⚠️ Some visual styling differences from graphviz
- ⚠️ Potential migration effort for existing users

**Neutral:**

- 🔄 Functionality equivalent for primary use cases
- 🔄 Learning curve for new visualization API

### Implementation Details

**Environment Configuration:**

```yaml
# environment.yml
# - graphviz             # REMOVED: Replaced with matplotlib-based visualization
# See docs/guides/architectural_decisions.md ADR-001 for rationale
```

**Code Changes:**

- `src/crackseg/model/common/utils.py`: Add matplotlib-based rendering
- `gui/utils/architecture_viewer.py`: Support matplotlib backend
- Update all import statements and error handling

**Fallback Strategy:**

```python
try:
    # Try matplotlib implementation first
    render_architecture_matplotlib(...)
except ImportError:
    try:
        # Fallback to graphviz if available
        render_architecture_graphviz(...)
    except ImportError:
        raise ImportError("No visualization backend available")
```

### Alternatives Considered

1. **Fix Graphviz Issues**: Attempted but too complex for cross-platform reliability
2. **Use Graphviz via Pip**: Similar dependency issues persist
3. **External Graphviz Installation**: Increases setup complexity for users
4. **Remove Visualization Entirely**: Reduces development and debugging capabilities

### Review Date

This decision should be reviewed if:

- Graphviz compilation issues are resolved in conda-forge
- A superior visualization library becomes available
- User feedback indicates strong preference for graphviz output
- Cross-platform compatibility improves significantly

### References

- Environment setup issues: Conda build failures with gdk-pixbuf
- Matplotlib visualization examples: `src/crackseg/utils/visualization/plots.py`
- Alternative visualization in project: `tests/e2e/performance/reporting/visualizations.py`
