#!/usr/bin/env python3
"""
Performance Maintenance Compatibility Wrapper. This script maintains
compatibility with existing workflows while delegating to the
refactored modular performance maintenance system. For new
development, import directly from scripts.performance module: from
scripts.performance import PerformanceMaintenanceManager
"""

import sys

# Import from the refactored modular system
try:
    from performance.maintenance_manager import main
except ImportError:
    # Fallback if the module is not available
    def main() -> None:
        print("Performance maintenance module not available")


if __name__ == "__main__":
    sys.exit(main())
