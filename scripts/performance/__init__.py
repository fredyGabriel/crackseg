"""
Performance Maintenance Module. Provides automated maintenance for the
CrackSeg performance system. Replaces the monolithic
performance_maintenance.py with modular components. Usage: from
scripts.performance import PerformanceMaintenanceManager manager =
PerformanceMaintenanceManager() health_results =
manager.health_check()
"""

from .maintenance_manager import PerformanceMaintenanceManager

__all__ = ["PerformanceMaintenanceManager"]
