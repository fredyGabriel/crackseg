import csv
import xml.etree.ElementTree as ET

# ruff: noqa: E501
CSV_PATH = "outputs/prd_project_refinement/test_suite_evaluation/reports/test_inventory.csv"
XML1_PATH = "outputs/prd_project_refinement/test_suite_evaluation/reports/test_results.xml"
XML2_PATH = "outputs/prd_project_refinement/test_suite_evaluation/reports/test_results_run2.xml"


# Parse test results from XML
def parse_test_results(xml_path):
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
                results[key] = "failed"
            elif case.find("skipped") is not None:
                results[key] = "skipped"
            else:
                results[key] = "passed"
    return results


def categorize_status(status1, status2):
    # Estrictamente: passing (ambas passed), failing (ambas failed/error),
    # flaky (cualquier otro caso)
    passing = {"passed"}
    failing = {"failed"}
    if status1 in passing and status2 in passing:
        return "passing"
    if status1 in failing and status2 in failing:
        return "failing"
    return "flaky"


def update_inventory(csv_path, results1, results2):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        explanation = reader[0]
        header = reader[1]
        data = reader[2:]
    idx_file = header.index("File Path")
    idx_name = header.index("Test Name")
    idx_status = header.index("Status")
    updated = [explanation, header]
    for row in data:
        file_path = row[idx_file].replace("/", ".")
        if file_path.endswith(".py"):
            file_path = file_path[:-3]
        if file_path.startswith("tests."):
            classname = file_path
        else:
            classname = "tests." + file_path
        test_name = row[idx_name]
        key = (classname, test_name)
        status1 = results1.get(key, "failed")
        status2 = results2.get(key, "failed")
        row[idx_status] = categorize_status(status1, status2)
        updated.append(row)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(updated)


if __name__ == "__main__":
    results1 = parse_test_results(XML1_PATH)
    results2 = parse_test_results(XML2_PATH)
    update_inventory(CSV_PATH, results1, results2)
