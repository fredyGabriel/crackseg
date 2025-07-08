"""Insights and recommendations generator for performance analysis.

This module generates actionable insights, risk assessments, and specific
recommendations based on performance data and analysis results.
"""

from __future__ import annotations

import logging
from typing import Any

from tests.e2e.performance.reporting.config import RISK_THRESHOLDS

logger = logging.getLogger(__name__)


class InsightsGenerator:
    """
    Generates actionable insights and recommendations from performance data
    """

    def __init__(self) -> None:
        """Initialize insights generator."""
        self.logger = logging.getLogger(__name__)

    def generate_insights(
        self,
        current_data: dict[str, Any],
        trend_analysis: dict[str, Any],
        regression_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate actionable insights and recommendations."""
        insights: dict[str, Any] = {
            "summary": "",
            "key_findings": [],
            "recommendations": [],
            "risk_assessment": "low",  # low, medium, high, critical
        }

        # Extract key metrics
        success_rate = current_data.get("overall_summary", {}).get(
            "average_success_rate", 0.0
        )
        total_violations = current_data.get("overall_summary", {}).get(
            "total_violations", 0
        )
        regressions_count = regression_analysis.get("regressions_detected", 0)

        # Perform risk assessment
        insights["risk_assessment"] = self._assess_risk(
            success_rate, total_violations, regressions_count
        )

        # Generate key findings
        insights["key_findings"] = self._generate_key_findings(
            success_rate, total_violations, regressions_count, trend_analysis
        )

        # Generate specific recommendations
        insights["recommendations"] = self._generate_recommendations(
            success_rate, total_violations, regressions_count
        )

        # Generate executive summary
        insights["summary"] = self._generate_summary(
            insights["risk_assessment"],
            success_rate,
            total_violations,
            regressions_count,
        )

        return insights

    def _assess_risk(
        self,
        success_rate: float,
        total_violations: int,
        regressions_count: int,
    ) -> str:
        """Assess overall risk level based on key metrics."""
        risk_factors: list[str] = []

        # Success rate risk assessment
        if success_rate < RISK_THRESHOLDS["success_rate"]["critical"]:
            risk_factors.append("critical")
        elif success_rate < RISK_THRESHOLDS["success_rate"]["high"]:
            risk_factors.append("high")
        elif success_rate < RISK_THRESHOLDS["success_rate"]["medium"]:
            risk_factors.append("medium")

        # Violations risk assessment
        if total_violations > RISK_THRESHOLDS["violations"]["critical"]:
            risk_factors.append("critical")
        elif total_violations > RISK_THRESHOLDS["violations"]["high"]:
            risk_factors.append("high")
        elif total_violations > RISK_THRESHOLDS["violations"]["medium"]:
            risk_factors.append("medium")

        # Regressions risk assessment
        if regressions_count > RISK_THRESHOLDS["regressions"]["critical"]:
            risk_factors.append("critical")
        elif regressions_count > RISK_THRESHOLDS["regressions"]["high"]:
            risk_factors.append("high")

        # Determine overall risk
        if "critical" in risk_factors:
            return "critical"
        elif "high" in risk_factors:
            return "high"
        elif "medium" in risk_factors:
            return "medium"
        else:
            return "low"

    def _generate_key_findings(
        self,
        success_rate: float,
        total_violations: int,
        regressions_count: int,
        trend_analysis: dict[str, Any],
    ) -> list[str]:
        """Generate key findings from performance analysis."""
        findings = [
            f"Overall success rate: {success_rate:.1f}%",
            f"Threshold violations: {total_violations}",
            f"Performance regressions: {regressions_count}",
        ]

        # Add trend insights
        if "trends" in trend_analysis:
            improving = sum(
                1
                for t in trend_analysis["trends"]
                if t["trend_direction"] == "improving"
            )
            degrading = sum(
                1
                for t in trend_analysis["trends"]
                if t["trend_direction"] == "degrading"
            )
            findings.append(
                f"Trending metrics: {improving} improving, {degrading} "
                "degrading"
            )

        # Add trend confidence if available
        if "confidence" in trend_analysis:
            confidence = trend_analysis["confidence"]
            findings.append(f"Trend analysis confidence: {confidence:.1%}")

        return findings

    def _generate_recommendations(
        self,
        success_rate: float,
        total_violations: int,
        regressions_count: int,
    ) -> list[str]:
        """Generate specific recommendations based on performance metrics."""
        recommendations = []

        # Success rate recommendations
        if success_rate < 90:
            recommendations.append(
                "🔴 Critical: Success rate below 90% - immediate intervention "
                "required"
            )
        elif success_rate < 95:
            recommendations.append(
                "🔴 Immediate attention needed: Success rate below 95%"
            )
        elif success_rate < 98:
            recommendations.append(
                "⚠️ Monitor closely: Success rate below optimal 98% threshold"
            )

        # Violations recommendations
        if total_violations > 5:
            recommendations.append(
                "🔴 High violation count detected - review all performance "
                "thresholds"
            )
        elif total_violations > 2:
            recommendations.append(
                "⚠️ Multiple threshold violations detected - review "
                "performance tuning"
            )
        elif total_violations > 0:
            recommendations.append(
                "ℹ️ Minor violations detected - monitor trends"
            )

        # Regressions recommendations
        if regressions_count > 3:
            recommendations.append(
                "🔴 Multiple regressions detected - roll back recent changes"
            )
        elif regressions_count > 0:
            recommendations.append(
                "📉 Performance regressions detected - investigate recent "
                "changes"
            )

        # Positive feedback
        if (
            success_rate > 98
            and total_violations == 0
            and regressions_count == 0
        ):
            recommendations.append(
                "✅ Excellent performance - maintain current practices"
            )

        # Default recommendation if no specific issues
        if not recommendations:
            recommendations.append(
                "📊 Performance is within acceptable ranges - continue "
                "monitoring"
            )

        return recommendations

    def _generate_summary(
        self,
        risk_level: str,
        success_rate: float,
        total_violations: int,
        regressions_count: int,
    ) -> str:
        """Generate executive summary."""
        risk_emoji = {
            "low": "🟢",
            "medium": "🟡",
            "high": "🟠",
            "critical": "🔴",
        }

        return (
            f"{risk_emoji[risk_level]} Performance Risk: {risk_level.title()}."
            f" Success Rate: {success_rate:.1f}%, "
            f"Violations: {total_violations}, "
            f"Regressions: {regressions_count}"
        )

    def generate_trend_insights(
        self, trend_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate specific insights from trend analysis."""
        insights: dict[str, Any] = {
            "trending_up": [],
            "trending_down": [],
            "stable_metrics": [],
            "confidence_summary": "",
        }

        if "trends" not in trend_analysis:
            return insights

        # Type-safe access to lists
        trending_up: list[Any] = insights["trending_up"]
        trending_down: list[Any] = insights["trending_down"]
        stable_metrics: list[Any] = insights["stable_metrics"]

        for trend in trend_analysis["trends"]:
            metric_name = trend["metric_name"]
            direction = trend["trend_direction"]
            confidence = trend["confidence"]

            trend_info = {
                "metric": metric_name,
                "change": trend["trend_percentage"],
                "confidence": confidence,
            }

            if direction == "improving":
                trending_up.append(trend_info)
            elif direction == "degrading":
                trending_down.append(trend_info)
            else:
                stable_metrics.append(trend_info)

        # Generate confidence summary
        overall_confidence = trend_analysis.get("confidence", 0.0)
        if overall_confidence > 0.8:
            insights["confidence_summary"] = (
                "High confidence in trend analysis"
            )
        elif overall_confidence > 0.6:
            insights["confidence_summary"] = (
                "Moderate confidence in trend analysis"
            )
        else:
            insights["confidence_summary"] = "Low confidence - need more data"

        return insights
