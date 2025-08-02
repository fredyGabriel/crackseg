"""Security scanning for packaging system.

This module handles vulnerability scanning and security analysis
for container images and dependencies.
"""

import json
import logging
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


class SecurityScanner:
    """Handles security scanning for container images."""

    def __init__(self, packaging_system: Any) -> None:
        """Initialize security scanner.

        Args:
            packaging_system: Reference to main packaging system
        """
        self.packaging_system = packaging_system
        self.logger = logging.getLogger(__name__)

    def perform_security_scan(self, image_name: str) -> dict[str, Any]:
        """Perform security scan on container image.

        Args:
            image_name: Name of the container image to scan

        Returns:
            Dictionary with security scan results
        """
        if not image_name:
            return {"error": "No image name provided"}

        try:
            # Check if trivy is available
            if not self._check_trivy_available():
                return {"error": "Trivy not available for security scanning"}

            # Perform vulnerability scan
            scan_results = self._run_trivy_scan(image_name)

            # Parse and categorize vulnerabilities
            parsed_results = self._parse_vulnerabilities(scan_results)

            self.logger.info(
                f"Security scan completed for {image_name}: "
                f"{parsed_results.get('total_vulnerabilities', 0)} "
                f"vulnerabilities found"
            )

            return parsed_results

        except Exception as e:
            self.logger.error(f"Security scan failed: {e}")
            return {"error": f"Security scan failed: {e}"}

    def _check_trivy_available(self) -> bool:
        """Check if trivy is available for scanning.

        Returns:
            True if trivy is available
        """
        try:
            result = subprocess.run(
                ["trivy", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _run_trivy_scan(self, image_name: str) -> dict[str, Any]:
        """Run trivy vulnerability scan.

        Args:
            image_name: Name of the container image

        Returns:
            Dictionary with scan results
        """
        try:
            # Run trivy scan with JSON output
            cmd = [
                "trivy",
                "image",
                "--format",
                "json",
                "--severity",
                "CRITICAL,HIGH,MEDIUM,LOW",
                image_name,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
            )

            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {"error": f"Trivy scan failed: {result.stderr}"}

        except subprocess.TimeoutExpired:
            return {"error": "Security scan timed out"}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON output from trivy"}
        except Exception as e:
            return {"error": f"Unexpected error during scan: {e}"}

    def _parse_vulnerabilities(
        self, scan_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Parse and categorize vulnerabilities.

        Args:
            scan_results: Raw scan results from trivy

        Returns:
            Dictionary with categorized vulnerability information
        """
        if "error" in scan_results:
            return scan_results

        try:
            # Extract vulnerability information
            vulnerabilities = scan_results.get("Results", [])

            total_vulnerabilities = 0
            critical_vulnerabilities = 0
            high_vulnerabilities = 0
            medium_vulnerabilities = 0
            low_vulnerabilities = 0

            # Process each result
            for result in vulnerabilities:
                vulns = result.get("Vulnerabilities", [])
                total_vulnerabilities += len(vulns)

                for vuln in vulns:
                    severity = vuln.get("Severity", "UNKNOWN")
                    if severity == "CRITICAL":
                        critical_vulnerabilities += 1
                    elif severity == "HIGH":
                        high_vulnerabilities += 1
                    elif severity == "MEDIUM":
                        medium_vulnerabilities += 1
                    elif severity == "LOW":
                        low_vulnerabilities += 1

            return {
                "total_vulnerabilities": total_vulnerabilities,
                "critical_vulnerabilities": critical_vulnerabilities,
                "high_vulnerabilities": high_vulnerabilities,
                "medium_vulnerabilities": medium_vulnerabilities,
                "low_vulnerabilities": low_vulnerabilities,
                "scan_timestamp": scan_results.get("Metadata", {}).get(
                    "UpdatedAt", ""
                ),
                "scanner_version": scan_results.get("Metadata", {}).get(
                    "ScannerVersion", ""
                ),
            }

        except Exception as e:
            return {"error": f"Failed to parse vulnerabilities: {e}"}

    def scan_dependencies(self, requirements_path: str) -> dict[str, Any]:
        """Scan Python dependencies for vulnerabilities.

        Args:
            requirements_path: Path to requirements.txt file

        Returns:
            Dictionary with dependency scan results
        """
        try:
            # Check if safety is available
            if not self._check_safety_available():
                return {
                    "error": "Safety not available for dependency scanning"
                }

            # Run safety check
            cmd = ["safety", "check", "-r", requirements_path, "--json"]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )

            if result.returncode == 0:
                return self._parse_safety_results(result.stdout)
            else:
                return {"error": f"Safety scan failed: {result.stderr}"}

        except Exception as e:
            return {"error": f"Dependency scan failed: {e}"}

    def _check_safety_available(self) -> bool:
        """Check if safety is available for dependency scanning.

        Returns:
            True if safety is available
        """
        try:
            result = subprocess.run(
                ["safety", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _parse_safety_results(self, safety_output: str) -> dict[str, Any]:
        """Parse safety scan results.

        Args:
            safety_output: JSON output from safety

        Returns:
            Dictionary with parsed safety results
        """
        try:
            results = json.loads(safety_output)

            total_vulns = len(results)
            high_vulns = len(
                [r for r in results if r.get("severity") == "high"]
            )
            medium_vulns = len(
                [r for r in results if r.get("severity") == "medium"]
            )
            low_vulns = len([r for r in results if r.get("severity") == "low"])

            return {
                "total_vulnerabilities": total_vulns,
                "high_vulnerabilities": high_vulns,
                "medium_vulnerabilities": medium_vulns,
                "low_vulnerabilities": low_vulns,
                "vulnerable_packages": [
                    {
                        "package": r.get("package"),
                        "installed_version": r.get("installed_version"),
                        "vulnerable_spec": r.get("vulnerable_spec"),
                        "severity": r.get("severity"),
                        "description": r.get("description"),
                    }
                    for r in results
                ],
            }

        except json.JSONDecodeError:
            return {"error": "Invalid JSON output from safety"}
        except Exception as e:
            return {"error": f"Failed to parse safety results: {e}"}

    def generate_security_report(
        self,
        image_scan_results: dict[str, Any],
        dependency_scan_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate comprehensive security report.

        Args:
            image_scan_results: Results from container image scan
            dependency_scan_results: Results from dependency scan

        Returns:
            Dictionary with comprehensive security report
        """
        report = {
            "scan_timestamp": image_scan_results.get("scan_timestamp", ""),
            "scanner_version": image_scan_results.get("scanner_version", ""),
            "image_vulnerabilities": image_scan_results,
            "dependency_vulnerabilities": dependency_scan_results,
        }

        # Calculate overall security score
        total_image_vulns = image_scan_results.get("total_vulnerabilities", 0)
        total_dep_vulns = dependency_scan_results.get(
            "total_vulnerabilities", 0
        )

        critical_image = image_scan_results.get("critical_vulnerabilities", 0)
        critical_dep = len(
            [
                v
                for v in dependency_scan_results.get("vulnerable_packages", [])
                if v.get("severity") == "high"
            ]
        )

        # Simple scoring: 0-100, higher is better
        if critical_image > 0 or critical_dep > 0:
            security_score = 0
        elif total_image_vulns + total_dep_vulns > 10:
            security_score = 30
        elif total_image_vulns + total_dep_vulns > 5:
            security_score = 60
        else:
            security_score = 100

        report["security_score"] = security_score
        report["recommendations"] = self._generate_security_recommendations(
            report
        )

        return report

    def _generate_security_recommendations(
        self, report: dict[str, Any]
    ) -> list[str]:
        """Generate security recommendations based on scan results.

        Args:
            report: Security report

        Returns:
            List of security recommendations
        """
        recommendations = []

        image_vulns = report.get("image_vulnerabilities", {})
        dep_vulns = report.get("dependency_vulnerabilities", {})

        # Image vulnerability recommendations
        if image_vulns.get("critical_vulnerabilities", 0) > 0:
            recommendations.append(
                "CRITICAL: Update base image to fix critical vulnerabilities"
            )

        if image_vulns.get("high_vulnerabilities", 0) > 0:
            recommendations.append(
                "HIGH: Update base image to fix high severity vulnerabilities"
            )

        # Dependency vulnerability recommendations
        vulnerable_packages = dep_vulns.get("vulnerable_packages", [])
        for pkg in vulnerable_packages:
            if pkg.get("severity") == "high":
                recommendations.append(
                    f"HIGH: Update {pkg.get('package')} from "
                    f"{pkg.get('installed_version')} to fix vulnerability"
                )

        # General recommendations
        if not recommendations:
            recommendations.append("No critical security issues found")
            recommendations.append("Regular security scans recommended")

        return recommendations
