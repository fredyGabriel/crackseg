# Coverage Validation Report - CrackSeg

**Generated:** 2025-06-01 19:15:01
**Validation Threshold:** 80.0%
**Failure Threshold:** 80.0%

## Summary

| Metric | Value | Status |
|--------|-------|--------|
| Overall Coverage | 23.2% |  |
| Total Statements | 8,045 | - |
| Covered Statements | 1,867 | - |
| Missing Statements | 6,178 | - |
| Modules Above Threshold | 32/131 | ⚠️ |

## Coverage Distribution

- ✅ Excellent 90 100: 32 modules
- ⚠️ Fair 60 79: 9 modules
-  Poor 40 59: 7 modules
-  Critical 0 39: 83 modules

## Critical Coverage Gaps

| File | Coverage | Priority | Missing Lines | Action Required |
|------|----------|----------|---------------|----------------|
| `src\evaluation\__main__.py` | 0.0% | 🚨 P0 | 165 | Immediate |
| `src\evaluation\setup.py` | 0.0% | 🚨 P0 | 28 | Immediate |
| `src\main.py` | 0.0% | 🚨 P0 | 180 | Immediate |
| `src\model\components\attention_decorator.py` | 0.0% | 🚨 P0 | 21 | Immediate |
| `src\model\components\registry_support.py` | 0.0% | 🚨 P0 | 96 | Immediate |
| `src\model\config\core.py` | 0.0% | 🚨 P0 | 93 | Immediate |
| `src\model\config\schemas.py` | 0.0% | 🚨 P0 | 30 | Immediate |
| `src\model\config\validation.py` | 0.0% | 🚨 P0 | 95 | Immediate |
| `src\model\config\instantiation.py` | 3.4% | 🚨 P0 | 199 | Immediate |
| `src\model\config\factory.py` | 4.2% | 🚨 P0 | 92 | Immediate |

## Recommendations

1. **Immediate Action Required:** Coverage is below the failure threshold
2. Focus on P0 priority files first
3. Implement the coverage improvement plan systematically
