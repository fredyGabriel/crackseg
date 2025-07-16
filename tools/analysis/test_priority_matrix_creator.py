#!/usr/bin/env python3
"""
Priority Matrix Creator for Test Failure Analysis - Subtask 6.3

This module creates a systematic priority matrix for the 29 categorized test
failures based on impact assessment and implementation complexity to guide
systematic resolution approach.

Follows CrackSeg coding standards: type annotations, modular design,
comprehensive error handling, and file size compliance (<300 lines).
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ImpactLevel(Enum):
    """Impact level classification for test failures."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ComplexityLevel(Enum):
    """Implementation complexity classification."""

    SIMPLE = "simple"  # < 2 hours, straightforward fix
    MEDIUM = "medium"  # 2-8 hours, moderate refactoring
    COMPLEX = "complex"  # > 8 hours, significant changes


class PriorityPhase(Enum):
    """Priority phases for systematic resolution."""

    PHASE_1 = "phase_1"  # High Impact + Simple/Medium Complexity
    PHASE_2 = "phase_2"  # High Impact + Complex OR Medium Impact + Simple
    PHASE_3 = "phase_3"  # Medium Impact + Medium/Complex
    PHASE_4 = "phase_4"  # Low Impact (any complexity)


@dataclass
class TestFailurePriority:
    """Represents a test failure with priority assessment."""

    test_name: str
    test_file: str
    error_message: str
    category: str
    severity: str
    impact_level: ImpactLevel
    complexity_level: ComplexityLevel
    priority_phase: PriorityPhase
    impact_score: float = 0.0
    complexity_score: float = 0.0
    estimated_effort_hours: float = 0.0
    affected_components: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    rationale: str = ""


class PriorityMatrixCreator:
    """Creates priority matrix for test failure resolution."""

    def __init__(self, categorization_file: Path):
        """Initialize with categorization data file."""
        self.categorization_file = categorization_file
        self.categorization_data: dict[str, Any] = {}
        self.priority_matrix: list[TestFailurePriority] = []

        # Component criticality mapping for impact assessment
        self.critical_components = {
            "Advanced Config Page",
            "Config Page & Session State",
            "Train Page & Session State",
            "Training Abort System",
            "Process Lifecycle Management",
            "Session State Management",
        }

        self.moderate_components = {
            "GPU Detection System",
            "Quick Actions UI",
            "Home Page Rendering",
            "Training Statistics Extraction",
            "Error Categorization System",
        }

    def load_categorization_data(self) -> None:
        """Load test failure categorization data."""
        try:
            with open(self.categorization_file, encoding="utf-8") as f:
                self.categorization_data = json.load(f)
            categories_count = len(
                self.categorization_data.get("detailed_analysis", {})
            )
            logger.info(
                f"Loaded categorization data: {categories_count} categories"
            )
        except Exception as e:
            logger.error(f"Failed to load categorization data: {e}")
            raise

    def assess_impact_level(
        self, failure: dict[str, Any], category: str
    ) -> tuple[ImpactLevel, float]:
        """Assess impact level based on affected components and coverage."""
        affected_components = failure.get("affected_components", [])
        severity = failure.get("severity", "medium")

        # Critical components assessment
        critical_overlap = len(
            set(affected_components) & self.critical_components
        )
        moderate_overlap = len(
            set(affected_components) & self.moderate_components
        )

        # Calculate impact score (0-100)
        impact_score = 0.0

        # Component criticality (40% weight)
        if critical_overlap > 0:
            impact_score += 40 * (
                critical_overlap / len(self.critical_components)
            )
        elif moderate_overlap > 0:
            impact_score += 25 * (
                moderate_overlap / len(self.moderate_components)
            )
        else:
            impact_score += 10  # Base score for any component

        # Severity weight (30% weight)
        severity_weights = {"high": 30, "medium": 20, "low": 10}
        impact_score += severity_weights.get(severity, 15)

        # Category impact (30% weight)
        category_weights = {
            "assertion_failure": 25,  # Core logic issues
            "import_module_error": 30,  # Blocks functionality
            "mock_fixture_issue": 20,  # Test infrastructure
            "configuration_problem": 15,  # Environment setup
        }
        impact_score += category_weights.get(category, 10)

        # Determine impact level
        if impact_score >= 70:
            return ImpactLevel.HIGH, impact_score
        elif impact_score >= 40:
            return ImpactLevel.MEDIUM, impact_score
        else:
            return ImpactLevel.LOW, impact_score

    def assess_complexity_level(
        self, failure: dict[str, Any], category: str
    ) -> tuple[ComplexityLevel, float, float]:
        """Assess implementation complexity and effort estimation."""
        error_message = failure.get("error_message", "")
        test_file = failure.get("test_file", "")

        # Base complexity by category
        category_complexity = {
            "import_module_error": (
                ComplexityLevel.SIMPLE,
                1.5,
            ),  # Usually missing imports
            "configuration_problem": (
                ComplexityLevel.SIMPLE,
                2.0,
            ),  # Test setup fixes
            "mock_fixture_issue": (
                ComplexityLevel.MEDIUM,
                4.0,
            ),  # MockSessionState redesign
            "assertion_failure": (
                ComplexityLevel.MEDIUM,
                5.0,
            ),  # Logic analysis required
        }

        _, base_hours = category_complexity.get(
            category, (ComplexityLevel.MEDIUM, 3.0)
        )

        # Complexity modifiers
        complexity_score = 50.0  # Base score
        effort_hours = base_hours

        # Systemic issues (affects multiple tests)
        if "MockSessionState" in error_message or "temp_path" in error_message:
            complexity_score += 20
            effort_hours += 2.0

        # GUI/Streamlit specific issues
        if "gui" in test_file.lower() or "streamlit" in error_message.lower():
            complexity_score += 10
            effort_hours += 1.0

        # Integration vs unit test complexity
        if "integration" in test_file:
            complexity_score += 15
            effort_hours += 1.5

        # Determine final complexity level
        if complexity_score >= 70:
            final_complexity = ComplexityLevel.COMPLEX
            effort_hours = max(effort_hours, 8.0)
        elif complexity_score >= 40:
            final_complexity = ComplexityLevel.MEDIUM
            effort_hours = max(effort_hours, 2.0)
        else:
            final_complexity = ComplexityLevel.SIMPLE
            effort_hours = min(effort_hours, 2.0)

        return final_complexity, complexity_score, effort_hours

    def determine_priority_phase(
        self, impact: ImpactLevel, complexity: ComplexityLevel
    ) -> PriorityPhase:
        """Determine priority phase based on impact and complexity matrix."""
        if impact == ImpactLevel.HIGH:
            if complexity in [ComplexityLevel.SIMPLE, ComplexityLevel.MEDIUM]:
                return PriorityPhase.PHASE_1
            else:
                return PriorityPhase.PHASE_2
        elif impact == ImpactLevel.MEDIUM:
            if complexity == ComplexityLevel.SIMPLE:
                return PriorityPhase.PHASE_2
            else:
                return PriorityPhase.PHASE_3
        else:  # Low impact
            return PriorityPhase.PHASE_4

    def create_priority_matrix(self) -> None:
        """Create comprehensive priority matrix for all test failures."""
        detailed_analysis = self.categorization_data.get(
            "detailed_analysis", {}
        )

        for category, category_data in detailed_analysis.items():
            individual_failures = category_data.get("individual_failures", [])

            for failure in individual_failures:
                # Assess impact and complexity
                impact_level, impact_score = self.assess_impact_level(
                    failure, category
                )
                complexity_level, complexity_score, effort_hours = (
                    self.assess_complexity_level(failure, category)
                )
                priority_phase = self.determine_priority_phase(
                    impact_level, complexity_level
                )

                # Create priority entry
                priority_entry = TestFailurePriority(
                    test_name=failure.get("test_name", ""),
                    test_file=failure.get("test_file", ""),
                    error_message=failure.get("error_message", ""),
                    category=category,
                    severity=failure.get("severity", "medium"),
                    impact_level=impact_level,
                    complexity_level=complexity_level,
                    priority_phase=priority_phase,
                    impact_score=impact_score,
                    complexity_score=complexity_score,
                    estimated_effort_hours=effort_hours,
                    rationale=(
                        f"Impact: {impact_level.value} "
                        f"({impact_score:.1f}/100), "
                        f"Complexity: {complexity_level.value} "
                        f"({complexity_score:.1f}/100)"
                    ),
                )

                self.priority_matrix.append(priority_entry)

        # Sort by priority phase, impact score (desc), complexity score (asc)
        self.priority_matrix.sort(
            key=lambda x: (
                x.priority_phase.value,
                -x.impact_score,
                x.complexity_score,
            )
        )

        matrix_size = len(self.priority_matrix)
        logger.info(
            f"Created priority matrix with {matrix_size} test failures"
        )

    def generate_priority_report(self) -> dict[str, Any]:
        """Generate comprehensive priority analysis report."""
        # Phase statistics
        phase_stats = {}
        for phase in PriorityPhase:
            phase_tests = [
                t for t in self.priority_matrix if t.priority_phase == phase
            ]
            total_effort = sum(t.estimated_effort_hours for t in phase_tests)

            phase_stats[phase.value] = {
                "test_count": len(phase_tests),
                "total_effort_hours": round(total_effort, 1),
                "avg_effort_per_test": (
                    round(total_effort / len(phase_tests), 1)
                    if phase_tests
                    else 0
                ),
                "impact_breakdown": {
                    level.value: len(
                        [t for t in phase_tests if t.impact_level == level]
                    )
                    for level in ImpactLevel
                },
                "complexity_breakdown": {
                    level.value: len(
                        [t for t in phase_tests if t.complexity_level == level]
                    )
                    for level in ComplexityLevel
                },
            }

        # Create detailed test list for each phase
        phase_details = {}
        for phase in PriorityPhase:
            phase_tests = [
                t for t in self.priority_matrix if t.priority_phase == phase
            ]
            phase_details[phase.value] = [
                {
                    "test_name": t.test_name,
                    "test_file": t.test_file,
                    "category": t.category,
                    "severity": t.severity,
                    "impact_level": t.impact_level.value,
                    "complexity_level": t.complexity_level.value,
                    "estimated_effort_hours": t.estimated_effort_hours,
                    "impact_score": round(t.impact_score, 1),
                    "complexity_score": round(t.complexity_score, 1),
                    "error_message": (
                        t.error_message[:100] + "..."
                        if len(t.error_message) > 100
                        else t.error_message
                    ),
                    "rationale": t.rationale,
                }
                for t in phase_tests
            ]

        return {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_tests_analyzed": len(self.priority_matrix),
                "methodology": (
                    "Hybrid impact-complexity scoring with systematic "
                    "phase classification"
                ),
                "scoring_criteria": {
                    "impact_factors": [
                        "component_criticality",
                        "severity_level",
                        "category_weight",
                    ],
                    "complexity_factors": [
                        "category_base",
                        "systemic_scope",
                        "test_type",
                        "technology_stack",
                    ],
                },
            },
            "phase_summary": phase_stats,
            "detailed_priorities": phase_details,
            "implementation_roadmap": {
                "phase_1": "Quick wins: High impact, simple/medium complexity",
                "phase_2": "Strategic fixes: High impact complex OR medium",
                "phase_3": "Systematic completion: Medium impact tasks",
                "phase_4": "Final cleanup: Low impact (optimize resources)",
            },
        }

    def save_priority_report(self, output_file: Path) -> None:
        """Save priority matrix report to JSON file."""
        report = self.generate_priority_report()

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Priority matrix report saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save priority report: {e}")
            raise

    def execute_analysis(
        self, output_file: Path | None = None
    ) -> dict[str, Any]:
        """Execute complete priority matrix analysis."""
        logger.info("Starting priority matrix creation for test failures")

        # Load data and create matrix
        self.load_categorization_data()
        self.create_priority_matrix()

        # Generate and save report
        output_path = output_file or Path("test_priority_matrix_report.json")
        report = self.generate_priority_report()
        self.save_priority_report(output_path)

        logger.info("Priority matrix analysis completed successfully")
        return report


def main() -> None:
    """Main execution function for priority matrix creation."""
    categorization_file = Path("test_failure_categorization_report.json")
    output_file = Path("test_priority_matrix_report.json")

    if not categorization_file.exists():
        logger.error(f"Categorization file not found: {categorization_file}")
        return

    try:
        creator = PriorityMatrixCreator(categorization_file)
        report = creator.execute_analysis(output_file)

        # Print summary
        total_tests = report["metadata"]["total_tests_analyzed"]
        print("\n✅ Priority Matrix Created Successfully")
        print(f"📊 Total Tests Analyzed: {total_tests}")
        print(f"📁 Output File: {output_file}")

        # Print phase breakdown
        print("\n📋 Priority Phase Breakdown:")
        for phase, stats in report["phase_summary"].items():
            print(
                f"  {phase.upper()}: {stats['test_count']} tests, "
                f"{stats['total_effort_hours']}h effort"
            )

    except Exception as e:
        logger.error(f"Priority matrix creation failed: {e}")
        raise


if __name__ == "__main__":
    main()
