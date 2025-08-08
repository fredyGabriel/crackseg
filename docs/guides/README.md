# CrackSeg Documentation Guides

> **Documentation organized by functional domain for the CrackSeg project**
>
> This folder contains comprehensive documentation organized by target audience and functional domain
> to facilitate navigation and maintenance.

## 🎯 New Structure by Domain

The documentation has been reorganized following modern ML project best practices, organized by
target audience:

### 📁 **User Guides** (`user-guides/`)

**For end users** who need to use system functionalities.

- **Getting Started**: First steps and onboarding
- **Usage**: Guides for specific component usage
- **Troubleshooting**: Common problem solutions

### 💻 **Developer Guides** (`developer-guides/`)

**For developers** who contribute to code and architecture.

- **Development**: Contribution guides and standards
- **Quality**: Testing and quality maintenance
- **Architecture**: Design decisions and structure

### 🚀 **Operational Guides** (`operational-guides/`)

**For operations teams** who manage deployment and monitoring.

- **Deployment**: Configuration and deployment management
- **Monitoring**: Observability and metrics
- **CI/CD**: Pipelines and automation
- **Workflows**: Operational workflows
- **Experiments**: Successful experiment execution and verification

### 📋 **Technical Specifications** (`technical-specs/`)

**For technical teams** who implement specifications.

- **Specifications**: Data formats and configurations
- **Experiments**: Experiments and benchmarks

### 📊 **Reporting & Visualization** (`reporting-visualization/`)

**For analysis teams** who generate reports and visualizations.

- **Reporting**: Experiment report generation
- **Visualization**: Chart and dashboard creation

## 📖 How to Navigate

### **By User Role**

| Role | Start in | Focus |
|------|----------|-------|
| **End User** | `user-guides/` | System usage, troubleshooting |
| **Developer** | `developer-guides/` | Contribution, testing, architecture |
| **DevOps/Operations** | `operational-guides/` | Deployment, monitoring, CI/CD |
| **Researcher** | `technical-specs/` | Specifications, experiments |
| **Analyst** | `reporting-visualization/` | Reports, visualization |

### **By Specific Need**

- **🚀 New to project**: `user-guides/getting-started/`
- **🐛 Problems**: `user-guides/troubleshooting/`
- **💻 Contribute code**: `developer-guides/development/`
- **🧪 Testing**: `developer-guides/quality/`
- **🏗️ Architecture**: `developer-guides/architecture/`
- **🚀 Deployment**: `operational-guides/deployment/`
- **📊 Monitoring**: `operational-guides/monitoring/`
- **⚙️ CI/CD**: `operational-guides/cicd/`
- **🧪 Experiments**: `operational-guides/successful_experiments_guide.md`
- **📋 Specifications**: `technical-specs/specifications/`
- **🔬 Experiments**: `technical-specs/experiments/`
- **📈 Reports**: `reporting-visualization/reporting/`
- **📊 Visualization**: `reporting-visualization/visualization/`

## 🔄 Migration in Progress

### **Current Status**

- ✅ **Structure created**: New domain-based organization
- ✅ **Files moved**: Documentation reorganized
- ✅ **READMEs updated**: Documentation of new structure
- 🔄 **Migration in progress**: Legacy content consolidation

### **Legacy Folders**

Each section contains `legacy/` folders with previous documentation that is being progressively
migrated to the new structure.

## 📈 Benefits of the New Structure

### **1. Intuitive Navigation**

- Organization by target audience
- Easy location of relevant information
- Reduced search time

### **2. Simplified Maintenance**

- Clear separation of responsibilities
- Independent updates by domain
- Scalability for future additions

### **3. Improved User Experience**

- Role-oriented documentation
- More effective onboarding
- Reduced learning curve

### **4. Documentation Quality**

- Elimination of obsolete files (46% reduction)
- Consolidated duplicate content
- Consistent standards by domain

## 📝 Contributing

When adding new documentation:

1. **Identify the domain**: Is it for users, developers, operations, etc.?
2. **Place in appropriate section**: Use the domain structure
3. **Update README**: Keep documentation updated
4. **Follow conventions**: Use established format

## 📊 Reorganization Metrics

- **Total files**: 50 (reduced from 93)
- **Backup files eliminated**: 43
- **Content reduction**: 46%
- **New sections**: 5 main domains
- **Subsections**: 15 specific categories

---

**Last reorganization**: $(Get-Date -Format "yyyy-MM-dd")
**Status**: Active migration
**Next step**: Legacy content consolidation
