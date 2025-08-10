# Engine Refactor Plan

## Overview

Refactor `src/crackseg/reporting/comparison/engine.py` (652 lines) into modular components.

## Current Structure Analysis

- **Total lines**: 652
- **Classes**: 1 (AutomatedComparisonEngine)
- **Functions**: 17 methods
- **Imports**: 7
- **Non-empty lines**: 539

## Proposed Split

### 1. `src/crackseg/reporting/comparison/core/engine.py` (~200 lines)

Main engine class with core comparison logic

- `AutomatedComparisonEngine` class
- `compare_experiments()` method
- `identify_best_performing()` method
- `generate_comparison_table()` method
- Core initialization and configuration

### 2. `src/crackseg/reporting/comparison/analysis/statistical.py` (~150 lines)

Statistical analysis utilities

- `_perform_statistical_analysis()` method
- `_perform_significance_tests()` method
- `_check_statistical_significance()` method
- `_calculate_confidence_level()` method
- Statistical computation helpers

### 3. `src/crackseg/reporting/comparison/analysis/ranking.py` (~120 lines)

Ranking and scoring logic

- `_generate_ranking_analysis()` method
- `_calculate_composite_scores()` method
- Ranking algorithms and scoring functions

### 4. `src/crackseg/reporting/comparison/analysis/trends.py` (~100 lines)

Performance trend analysis

- `_analyze_performance_trends()` method
- Trend detection and analysis utilities

### 5. `src/crackseg/reporting/comparison/analysis/anomalies.py` (~80 lines)

Anomaly detection

- `_detect_anomalies()` method
- Anomaly detection algorithms

### 6. `src/crackseg/reporting/comparison/utils/metrics.py` (~80 lines)

Metrics extraction and processing

- `_extract_comparison_metrics()` method
- `_extract_model_config()` method
- `_extract_training_config()` method
- Metrics processing utilities

### 7. `src/crackseg/reporting/comparison/utils/recommendations.py` (~60 lines)

Recommendation generation

- `_generate_comparison_recommendations()` method
- Recommendation logic and templates

### 8. `src/crackseg/reporting/comparison/utils/table_utils.py` (~60 lines)

Table generation utilities

- `_calculate_table_statistics()` method
- `_calculate_metric_correlations()` method
- Table formatting and statistics

## Migration Strategy

### Phase 1: Create new modules

1. Create directory structure: `src/crackseg/reporting/comparison/core/`
2. Create directory structure: `src/crackseg/reporting/comparison/analysis/`
3. Create directory structure: `src/crackseg/reporting/comparison/utils/`

### Phase 2: Extract components

1. Extract statistical analysis to `analysis/statistical.py`
2. Extract ranking logic to `analysis/ranking.py`
3. Extract trend analysis to `analysis/trends.py`
4. Extract anomaly detection to `analysis/anomalies.py`
5. Extract metrics utilities to `utils/metrics.py`
6. Extract recommendations to `utils/recommendations.py`
7. Extract table utilities to `utils/table_utils.py`

### Phase 3: Refactor main engine

1. Move core engine class to `core/engine.py`
2. Update imports to use new modules
3. Maintain public API compatibility

### Phase 4: Create shim

1. Convert original `engine.py` to shim
2. Re-export all public symbols from new locations
3. Add deprecation warnings for direct imports

## Acceptance Criteria

- [ ] All new files ≤ 300 lines
- [ ] Original `engine.py` becomes shim with re-exports
- [ ] All tests pass
- [ ] Public API remains unchanged
- [ ] Import mapping updated
- [ ] Quality gates pass (ruff, black, basedpyright)
- [ ] Link checker passes (0 issues)

## Risk Assessment

- **Low Risk**: Well-defined module boundaries
- **Mitigation**: Comprehensive testing and shim for backward compatibility
- **Rollback**: Original file preserved as shim

## Timeline

- **Phase 1**: 30 minutes (create structure)
- **Phase 2**: 60 minutes (extract components)
- **Phase 3**: 30 minutes (refactor main engine)
- **Phase 4**: 15 minutes (create shim)
- **Testing**: 30 minutes
- **Total**: ~3 hours
