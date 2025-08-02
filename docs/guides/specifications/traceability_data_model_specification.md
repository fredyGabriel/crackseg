# Traceability Data Model Specification

This document defines the comprehensive data model for the artifact traceability system
in the CrackSeg project, including entities, relationships, and compliance requirements.

## Overview

The traceability data model provides a complete framework for tracking artifacts, experiments,
versions, and their relationships throughout the ML development lifecycle. It ensures
reproducibility, auditability, and compliance with regulatory requirements.

## Design Principles

### 1. **Comprehensive Coverage**

- Track all artifact types (models, checkpoints, metrics, visualizations, etc.)
- Maintain complete experiment metadata
- Support bidirectional lineage tracking
- Enable compliance and audit trails

### 2. **Data Integrity**

- SHA256 checksums for all artifacts
- Validation rules for all entities
- Version control with semantic versioning
- Immutable audit records

### 3. **Flexibility and Extensibility**

- Extensible metadata fields
- Configurable compliance levels
- Support for custom artifact types
- Query and filter capabilities

### 4. **Performance and Scalability**

- Efficient query patterns
- Pagination support
- Indexed relationships
- Export capabilities

## Core Entities

### ArtifactEntity

The central entity representing any artifact in the system.

**Key Features:**

- Unique identifier with validation
- Type classification (model, checkpoint, metrics, etc.)
- File metadata (path, size, checksum)
- Ownership and compliance tracking
- Version history and dependencies
- Bidirectional lineage relationships

**Validation Rules:**

- Artifact ID cannot be empty or contain spaces
- Checksum must be valid SHA256 hash (64 characters)
- File path must be valid and accessible
- Version must follow semantic versioning (x.y.z)

**Example:**

```python
artifact = ArtifactEntity(
    artifact_id="model-best-20241201",
    artifact_type=ArtifactType.MODEL,
    file_path=Path("outputs/experiments/exp1/model_best.pth"),
    file_size=1024000,
    checksum="a1b2c3d4e5f6...",
    name="Best Model - Experiment 1",
    description="Best performing model from experiment 1",
    owner="ml_engineer",
    experiment_id="exp-20241201-001",
    version="1.0.0"
)
```

### ExperimentEntity

Comprehensive experiment tracking with full metadata.

**Key Features:**

- Complete environment information
- Configuration tracking with hashing
- Training progress and metrics
- Git and system metadata
- Artifact associations
- Parent/child experiment relationships

**Metadata Categories:**

- **Basic**: ID, name, status, description, tags
- **Configuration**: Hash, summary, parameters
- **Environment**: Python, PyTorch, CUDA, platform
- **Training**: Epochs, metrics, timing
- **Git**: Commit, branch, dirty state
- **System**: Hostname, username, memory, GPU
- **Timestamps**: Created, started, completed, updated

**Example:**

```python
experiment = ExperimentEntity(
    experiment_id="exp-20241201-001",
    experiment_name="SwinV2-Hybrid-Experiment",
    status=ExperimentStatus.COMPLETED,
    config_hash="sha256:abc123...",
    python_version="3.12.0",
    pytorch_version="2.7.0",
    total_epochs=100,
    best_metrics={"iou": 0.85, "dice": 0.90},
    artifact_ids=["model-best", "metrics-final", "config-final"]
)
```

### VersionEntity

Version tracking for artifacts with change management.

**Key Features:**

- Semantic versioning (major.minor.patch)
- Change summaries and types
- Dependency tracking
- File integrity verification
- Metadata preservation

**Version Types:**

- **Major**: Breaking changes, incompatible updates
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, minor improvements

**Example:**

```python
version = VersionEntity(
    version_id="v1.2.3-model-best",
    artifact_id="model-best-20241201",
    version_number="1.2.3",
    file_path=Path("outputs/models/model_v1.2.3.pth"),
    checksum="sha256:def456...",
    change_summary="Improved attention mechanism",
    change_type="minor",
    dependencies={"dataset": "v2.1.0", "config": "v1.0.0"}
)
```

### LineageEntity

Relationship tracking between artifacts.

**Key Features:**

- Source and target artifact identification
- Relationship type classification
- Confidence scoring
- Bidirectional tracking
- Metadata preservation

**Relationship Types:**

- **derived_from**: Artifact created from another
- **depends_on**: Dependency relationship
- **evolves_from**: Evolution/improvement
- **validates**: Validation relationship
- **complements**: Complementary artifacts

**Example:**

```python
lineage = LineageEntity(
    lineage_id="lineage-001",
    source_artifact_id="dataset-v1.0",
    target_artifact_id="model-v1.2",
    relationship_type="derived_from",
    relationship_description="Model trained on dataset v1.0",
    confidence=0.95
)
```

## Query and Filter System

### TraceabilityQuery

Comprehensive query interface with multiple filter options.

**Filter Categories:**

- **Basic**: Artifact types, experiment IDs, tags, owners
- **Temporal**: Date ranges for creation/update
- **Status**: Verification status, compliance level
- **Lineage**: Include lineage, depth limits
- **Pagination**: Limit, offset for large result sets
- **Sorting**: Field and order specification

**Example:**

```python
query = TraceabilityQuery(
    artifact_types=[ArtifactType.MODEL, ArtifactType.CHECKPOINT],
    experiment_ids=["exp-20241201-001"],
    tags=["best", "production"],
    verification_status=[VerificationStatus.VERIFIED],
    created_after=datetime(2024, 1, 1),
    include_lineage=True,
    max_lineage_depth=3,
    limit=50,
    sort_by="created_at",
    sort_order="desc"
)
```

### TraceabilityResult

Structured query results with metadata.

**Features:**

- Original query preservation
- Result counts and timing
- Structured result sets
- Export capabilities

## Compliance and Audit System

### ComplianceRecord

Audit trail for compliance verification.

**Compliance Levels:**

- **BASIC**: Essential integrity checks
- **STANDARD**: Standard verification procedures
- **COMPREHENSIVE**: Full audit trail
- **AUDIT**: Regulatory compliance

**Audit Components:**

- Compliance level specification
- Auditor identification
- Pass/fail status
- Findings and recommendations
- Timestamp preservation

**Example:**

```python
compliance = ComplianceRecord(
    record_id="audit-20241201-001",
    artifact_id="model-best-20241201",
    compliance_level=ComplianceLevel.STANDARD,
    auditor="ml_engineer",
    passed=True,
    findings=["All checksums verified", "Dependencies documented"],
    recommendations=["Consider comprehensive audit for production"]
)
```

## Data Validation and Integrity

### Validation Rules

**ArtifactEntity:**

- Non-empty, space-free artifact IDs
- Valid SHA256 checksums (64 characters)
- Accessible file paths
- Valid artifact types

**ExperimentEntity:**

- Non-empty experiment IDs
- Valid status values
- Required environment metadata
- Consistent timestamp relationships

**VersionEntity:**

- Semantic version format (x.y.z)
- Integer version components
- Valid file paths and checksums

**LineageEntity:**

- Confidence values between 0.0 and 1.0
- Valid relationship types
- Non-circular relationships

### Integrity Checks

**Checksum Verification:**

- SHA256 calculation for all artifacts
- Automatic verification on access
- Integrity status tracking

**Relationship Validation:**

- Bidirectional relationship consistency
- Circular dependency detection
- Orphan artifact identification

**Version Consistency:**

- Semantic version progression
- Dependency version compatibility
- Change type validation

## Export and Serialization

### TraceabilityExport

Comprehensive data export with metadata.

**Export Features:**

- Multiple format support (JSON, YAML, CSV)
- Complete entity relationships
- Metadata preservation
- Summary statistics
- Timestamp tracking

**Export Components:**

- All entity types (artifacts, experiments, versions, lineage)
- Compliance records
- Query metadata
- Export statistics

## Integration Patterns

### With Existing Systems

**ExperimentTracker Integration:**

```python
# Convert ExperimentTracker metadata to ExperimentEntity
experiment_entity = ExperimentEntity(
    experiment_id=tracker.experiment_id,
    experiment_name=tracker.experiment_name,
    status=ExperimentStatus(tracker.metadata.status),
    # ... other fields
)
```

**ArtifactManager Integration:**

```python
# Convert ArtifactManager metadata to ArtifactEntity
artifact_entity = ArtifactEntity(
    artifact_id=metadata.artifact_id,
    artifact_type=ArtifactType(metadata.artifact_type),
    file_path=Path(metadata.file_path),
    # ... other fields
)
```

### Database Integration

**SQLite Schema:**

```sql
-- Artifacts table
CREATE TABLE artifacts (
    artifact_id TEXT PRIMARY KEY,
    artifact_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    checksum TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    owner TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    verification_status TEXT NOT NULL,
    compliance_level TEXT NOT NULL,
    experiment_id TEXT,
    version TEXT NOT NULL,
    metadata TEXT
);

-- Experiments table
CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    experiment_name TEXT NOT NULL,
    status TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    python_version TEXT NOT NULL,
    pytorch_version TEXT NOT NULL,
    -- ... other fields
);

-- Lineage table
CREATE TABLE lineage (
    lineage_id TEXT PRIMARY KEY,
    source_artifact_id TEXT NOT NULL,
    target_artifact_id TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL
);
```

## Performance Considerations

### Query Optimization

**Indexing Strategy:**

- Primary keys on all entity IDs
- Composite indexes for common queries
- Full-text search on descriptions
- Temporal indexes for date ranges

**Pagination:**

- Limit result sets to manageable sizes
- Use offset-based pagination
- Implement cursor-based pagination for large datasets

### Storage Optimization

**Compression:**

- Compress large metadata fields
- Use efficient serialization formats
- Implement data archival strategies

**Caching:**

- Cache frequently accessed entities
- Implement query result caching
- Use Redis for session data

## Security and Access Control

### Data Protection

**Encryption:**

- Encrypt sensitive metadata
- Secure checksum storage
- Protect compliance records

**Access Control:**

- Role-based access to entities
- Audit trail for all operations
- Secure export capabilities

### Compliance Features

**GDPR Compliance:**

- Data retention policies
- Right to be forgotten
- Data portability

**Industry Standards:**

- SOC 2 compliance support
- ISO 27001 alignment
- HIPAA considerations for medical data

## Future Extensions

### Planned Enhancements

**Advanced Lineage:**

- Graph-based lineage visualization
- Impact analysis capabilities
- Dependency resolution

**Machine Learning Integration:**

- Automated artifact classification
- Anomaly detection
- Predictive lineage analysis

**API Development:**

- RESTful API for external access
- GraphQL for complex queries
- Webhook support for real-time updates

## References

- **Pydantic Documentation**: <https://docs.pydantic.dev/>
- **Semantic Versioning**: <https://semver.org/>
- **SHA256 Specification**: RFC 6234
- **Data Lineage Standards**: DCAM, DAMA-DMBOK
