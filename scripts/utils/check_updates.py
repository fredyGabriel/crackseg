#!/usr/bin/env python
"""
Script to verify available updates for main dependencies.
"""

import subprocess
import json
from packaging import version
import requests
import yaml


def get_pypi_version(package_name):
    """Gets the latest version available on PyPI."""
    try:
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
        if response.status_code == 200:
            return response.json()["info"]["version"]
        return None
    except Exception:
        return None


def get_conda_version(package_name):
    """Gets the latest version available on conda-forge."""
    try:
        result = subprocess.run(
            ["conda", "search", "-c", "conda-forge", package_name, "--json"],
            capture_output=True,
            text=True
        )
        data = json.loads(result.stdout)
        if package_name in data:
            versions = [entry["version"] for entry in data[package_name]]
            return max(versions, key=version.parse)
        return None
    except Exception:
        return None


def get_current_versions():
    """Reads current versions from environment.yml."""
    with open("environment.yml", "r") as f:
        env = yaml.safe_load(f)

    deps = {}
    for dep in env["dependencies"]:
        if isinstance(dep, str):
            name_ver = dep.split("=")
            if len(name_ver) > 1:
                deps[name_ver[0]] = name_ver[1]

    if "pip" in env["dependencies"][-1]:
        for pip_dep in env["dependencies"][-1]["pip"]:
            name_ver = pip_dep.split("==")
            if len(name_ver) > 1:
                deps[name_ver[0]] = name_ver[1]

    return deps


def main():
    """Main function that compares versions and shows available updates."""
    print("Checking for available updates...")
    print("\nMain dependencies:")
    print("-" * 60)

    current_versions = get_current_versions()

    # List of main packages to check
    main_packages = [
        "pytorch",
        "torchvision",
        "albumentations",
        "hydra-core",
        "pytest",
        "python-dotenv",
        "opencv-python",
        "numpy",
        "matplotlib",
        "scikit-image"
    ]

    updates_available = False

    for package in main_packages:
        current = current_versions.get(package, "Not installed")
        latest_pypi = get_pypi_version(package)
        latest_conda = get_conda_version(package)

        latest = latest_conda if latest_conda else latest_pypi

        if latest and current != "Not installed":
            if version.parse(latest) > version.parse(current):
                updates_available = True
                print(f"\n{package}:")
                print(f"  Current version: {current}")
                print(f"  Available version: {latest}")
                print("  ⚠️ Update available")
            else:
                print(f"\n{package}: ✓ Up to date ({current})")
        else:
            print(f"\n{package}: ❌ Could not verify")

    if not updates_available:
        print("\n✅ All main dependencies are up to date.")
    else:
        print("\n⚠️ Updates are available. Consider updating environment.yml")
        print("To update, modify the versions in environment.yml and run:")
        print("conda env update -f environment.yml --prune")


if __name__ == "__main__":
    main()
