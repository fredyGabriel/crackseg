import csv
import xml.etree.ElementTree as ET

# ruff: noqa: E501
CSV_PATH = "outputs/prd_project_refinement/test_suite_evaluation/reports/test_inventory.csv"
XML_PATH = "outputs/prd_project_refinement/test_suite_evaluation/reports/test_results.xml"


# Parse test results from XML
def parse_test_results(xml_path: str) -> dict[tuple[str, str], str]:
    """Parses test results from an XML file and returns a mapping from (classname, test name) to status."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    results = {}
    for suite in root.findall(".//testsuite"):
        for case in suite.findall("testcase"):
            classname = case.attrib.get("classname", "")
            name = case.attrib.get("name", "")
            key = (classname, name)
            if case.find("failure") is not None:
                results[key] = "failed"
            elif case.find("error") is not None:
                results[key] = "error"
            elif case.find("skipped") is not None:
                results[key] = "skipped"
            else:
                results[key] = "passed"
    return results


def update_inventory(
    csv_path: str, results: dict[tuple[str, str], str]
) -> None:
    """Actualiza el inventario de tests en el CSV con el estado a partir de los resultados XML."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        explanation = reader[0]
        header = reader[1]
        data = reader[2:]
    # Find column indices
    idx_file = header.index("File Path")
    idx_name = header.index("Test Name")
    idx_status = header.index("Status")
    # Update rows
    updated = [explanation, header]
    for row in data:
        file_path = row[idx_file].replace("/", ".")
        # Remove .py extension and leading 'tests.'
        if file_path.endswith(".py"):
            file_path = file_path[:-3]
        if file_path.startswith("tests."):
            classname = file_path
        else:
            classname = "tests." + file_path
        test_name = row[idx_name]
        key = (classname, test_name)
        status = results.get(key, "not found")
        row[idx_status] = status
        updated.append(row)
    # Write back
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(updated)


if __name__ == "__main__":
    results = parse_test_results(XML_PATH)
    update_inventory(CSV_PATH, results)
