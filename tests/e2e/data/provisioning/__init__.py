"""Test data provisioning package.

This package provides comprehensive test data provisioning functionality
including basic provisioning, suite generation, and database seeding.
"""

from .core import ProvisioningResult, TestDataProvisioner
from .database import get_provisioning_summary, seed_test_database
from .suites import (
    provision_basic_suite,
    provision_comprehensive_suite,
    provision_error_test_data,
)

# Bind methods to TestDataProvisioner class
TestDataProvisioner.provision_basic_suite = provision_basic_suite
TestDataProvisioner.provision_comprehensive_suite = (
    provision_comprehensive_suite
)
TestDataProvisioner.provision_error_test_data = provision_error_test_data
TestDataProvisioner.seed_test_database = seed_test_database
TestDataProvisioner.get_provisioning_summary = get_provisioning_summary

__all__ = [
    "ProvisioningResult",
    "TestDataProvisioner",
]
