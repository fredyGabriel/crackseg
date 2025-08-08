# Operational Guides - CrackSeg Documentation

> **Operational guides for the CrackSeg project**
>
> This section contains documentation for operations, deployment, monitoring and workflows of the
> crack segmentation system.

## 🚀 Purpose

**Operational Guides** are designed for operations teams who need to:

- **Deploy** the system in production
- **Monitor** performance and health
- **Configure** CI/CD pipelines
- **Manage** workflows
- **Maintain** infrastructure

## 📁 Structure

```bash
operational-guides/
├── deployment/         # Deployment guides
│   └── legacy/        # Legacy deployment
├── monitoring/         # Monitoring and observability
│   └── legacy/        # Legacy monitoring
├── cicd/              # CI/CD and pipelines
│   └── legacy/        # Legacy CI/CD
└── workflows/          # Workflows
    └── legacy/        # Legacy workflows
```

## 🔧 Available Guides

### **Experiments** ✅ **UPDATED**

- **Successful Experiments Guide**: `successful_experiments_guide.md`
  - SwinV2 360x360 (Crack500) - Fully verified
  - SwinV2 320x320 (PY-CrackDB) - Recently completed
  - Hydra configuration resolution
  - Bidirectional cropping algorithm
  - Quality gates and troubleshooting

### **Deployment** ⚠️ **NEEDS UPDATE**

- Deployment configuration
- System user guides
- Deployment troubleshooting
- Pipeline architecture

### **Monitoring** ✅ **UPDATED**

- **Monitoring Guide**: `monitoring/monitoring_guide.md`
  - Training monitoring with TensorBoard
  - System health monitoring
  - Performance tracking and alerts
  - Automated monitoring setup

### **CI/CD** ⚠️ **NEEDS UPDATE**

- Testing integration
- Integration guides
- Automated pipelines
- Environment configuration

### **Workflows** ✅ **UPDATED**

- **Training Workflow Guide**: `workflows/training_workflow_guide.md`
  - Complete training workflow from setup to execution
  - Quality gates and verification procedures
  - Troubleshooting and optimization
  - Advanced workflow customization

### **Deployment** ✅ **UPDATED**

- **Deployment Guide**: `deployment/deployment_guide.md`
  - Development environment setup
  - Configuration deployment
  - Production considerations
  - Monitoring and troubleshooting

## 📖 How to Use

1. **Experiments**: Start with `successful_experiments_guide.md` ✅ **READY**
2. **Deployment**: Start with `deployment/deployment_guide.md` ✅ **READY**
3. **Workflows**: Check `workflows/training_workflow_guide.md` ✅ **READY**
4. **Monitoring**: Check `monitoring/monitoring_guide.md` ✅ **READY**
5. **CI/CD**: Go to `cicd/` for pipelines ⚠️ **NEEDS UPDATE**

## 🔄 Migration

The `legacy/` folders contain previous documentation that is being progressively migrated to the
new structure. This documentation will be consolidated and updated progressively.

---

**Last update**: $(Get-Date -Format "yyyy-MM-dd")
**Status**: Active migration
