#!/usr/bin/env python3
"""
Generate test inventory from pytest results.

This script analyzes pytest XML output and generates a comprehensive
test inventory with metadata, status, and categorization.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

# Output file path
OUTPUT_CSV = "artifacts/global/reports/test_inventory.csv"


def parse_pytest_xml(xml_path: str) -> list[dict]:
    """Parse pytest XML output and extract test information."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    tests = []
    for test in root.findall(".//testcase"):
        test_info = {
            "name": test.get("name", ""),
            "classname": test.get("classname", ""),
            "time": float(test.get("time", 0)),
            "status": "passed",
            "message": "",
            "type": "",
        }

        # Check for failures or errors
        failure = test.find("failure")
        error = test.find("error")
        skipped = test.find("skipped")

        if failure is not None:
            test_info["status"] = "failed"
            test_info["message"] = failure.get("message", "")
            test_info["type"] = "failure"
        elif error is not None:
            test_info["status"] = "error"
            test_info["message"] = error.get("message", "")
            test_info["type"] = "error"
        elif skipped is not None:
            test_info["status"] = "skipped"
            test_info["message"] = skipped.get("message", "")
            test_info["type"] = "skipped"

        tests.append(test_info)

    return tests


def categorize_test(test_name: str, classname: str) -> str:
    """Categorize test based on name and class."""
    name_lower = test_name.lower()
    class_lower = classname.lower()

    # Unit tests
    if "test_" in name_lower and (
        "unit" in class_lower or "test_" in class_lower
    ):
        return "unit"

    # Integration tests
    if "test_" in name_lower and (
        "integration" in class_lower or "e2e" in class_lower
    ):
        return "integration"

    # Model tests
    if any(
        keyword in name_lower
        for keyword in ["model", "network", "unet", "encoder", "decoder"]
    ):
        return "model"

    # Data tests
    if any(
        keyword in name_lower
        for keyword in ["data", "dataset", "loader", "transform"]
    ):
        return "data"

    # Training tests
    if any(
        keyword in name_lower
        for keyword in ["train", "loss", "optimizer", "scheduler"]
    ):
        return "training"

    # Evaluation tests
    if any(
        keyword in name_lower for keyword in ["eval", "metric", "iou", "dice"]
    ):
        return "evaluation"

    # GUI tests
    if any(
        keyword in name_lower for keyword in ["gui", "interface", "app", "web"]
    ):
        return "gui"

    # Default to unit
    return "unit"


def generate_test_inventory(xml_path: str) -> pd.DataFrame:
    """Generate comprehensive test inventory."""
    print(f"Parsing pytest XML: {xml_path}")

    # Parse XML
    tests = parse_pytest_xml(xml_path)

    # Convert to DataFrame
    df = pd.DataFrame(tests)

    # Add categorization
    df["category"] = df.apply(
        lambda row: categorize_test(row["name"], row["classname"]), axis=1
    )

    # Add file path extraction
    df["file_path"] = df["classname"].apply(
        lambda x: x.replace(".", "/") + ".py" if x else ""
    )

    # Add priority based on category and status
    priority_map = {
        "unit": 1,
        "model": 2,
        "data": 3,
        "training": 4,
        "evaluation": 5,
        "integration": 6,
        "gui": 7,
    }
    df["priority"] = df["category"].apply(lambda x: priority_map.get(x, 1))

    # Add status priority
    status_priority = {"failed": 1, "error": 2, "skipped": 3, "passed": 4}
    df["status_priority"] = df["status"].apply(
        lambda x: status_priority.get(x, 4)
    )

    # Sort by priority and status
    df = df.sort_values(
        ["priority", "status_priority", "time"], ascending=[True, True, False]
    )

    return df


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate test inventory from pytest XML"
    )
    parser.add_argument("xml_path", help="Path to pytest XML output file")
    parser.add_argument(
        "--output", default=OUTPUT_CSV, help="Output CSV file path"
    )

    args = parser.parse_args()

    # Generate inventory
    df = generate_test_inventory(args.xml_path)

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Test inventory saved to: {output_path}")

    # Print summary
    print("\nTest Inventory Summary:")
    print(f"Total tests: {len(df)}")
    print(f"Passed: {len(df[df['status'] == 'passed'])}")
    print(f"Failed: {len(df[df['status'] == 'failed'])}")
    print(f"Errors: {len(df[df['status'] == 'error'])}")
    print(f"Skipped: {len(df[df['status'] == 'skipped'])}")

    print("\nBy Category:")
    for category in df["category"].unique():
        count = len(df[df["category"] == category])
        print(f"  {category}: {count}")


if __name__ == "__main__":
    main()
