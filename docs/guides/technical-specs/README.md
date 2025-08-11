# Technical Specifications - CrackSeg Documentation

> **Technical specifications for the CrackSeg project**
>
> This section contains detailed specifications, data formats and experiments for the crack
> segmentation system.

## 📋 Purpose

**Technical Specifications** are designed for technical teams who need to:

- **Understand** data formats and configurations
- **Implement** technical specifications
- **Replicate** experiments and benchmarks
- **Extend** functionalities according to specifications
- **Validate** compliance with standards

## 📁 Structure

```bash
technical-specs/
├── specifications/      # Technical specifications
│   └── legacy/         # Legacy specifications
└── experiments/         # Experiments and benchmarks
    └── legacy/         # Legacy experiments
```

## 🔬 Available Guides

### **Specifications**

- **checkpoint_format_specification.md** (legacy) - `specifications/legacy/checkpoint_format_specification.md`
- **configuration_storage_specification.md** (legacy) - `specifications/legacy/configuration_storage_specification.md`
- **performance_benchmarking_system.md** (legacy) - `specifications/legacy/performance_benchmarking_system.md`
- **traceability_data_model_specification.md** (legacy) - `specifications/legacy/traceability_data_model_specification.md`

### **Experiments**

- **README_swinv2_hybrid.md** - SwinV2 Hybrid experiments
- Specific experiment documentation
- Experiment configurations
- Results and analysis

## 📖 How to Use

1. **Implementation**: Check `specifications/` for formats
2. **Experiments**: Go to `experiments/` to replicate
3. **Benchmarking**: Use performance specifications

## 🔄 Migration

The `legacy/` folders contain previous documentation that is being progressively migrated to the
new structure. This documentation will be consolidated and updated progressively.

### Legacy pointers

- `specifications/legacy/`
- `experiments/legacy/`

---

**Last update**: $(Get-Date -Format "yyyy-MM-dd")
**Status**: Active migration
