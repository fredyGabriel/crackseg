import csv
import xml.etree.ElementTree as ET

# ruff: noqa: E501
CSV_PATH = "outputs/prd_project_refinement/test_suite_evaluation/reports/test_inventory.csv"
XML_PATH = "outputs/prd_project_refinement/test_suite_evaluation/reports/test_results.xml"
REPORT_PATH = "outputs/prd_project_refinement/test_suite_evaluation/reports/manual_intervention_required.txt"

# Palabras clave para intervención manual
target_keywords = [
    "N/A",
    "setup failed",
    "external dependency",
    "permission",
    "manual",
    "not implemented",
    "requires user",
    "requires manual",
    "not supported",
    "not available",
    "not configured",
    "not found",
    "OSError",
    "PermissionError",
    "FileNotFoundError",
    "ModuleNotFoundError",
    "ImportError",
    "EnvironmentError",
    "missing",
    "cannot import",
    "failed to",
    "requires configuration",
]


def parse_test_errors(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    errors = {}
    for suite in root.findall(".//testsuite"):
        for case in suite.findall("testcase"):
            name = case.attrib.get("name", "")
            classname = case.attrib.get("classname", "")
            key = (classname, name)
            # Buscar errores/failures
            for tag in ("failure", "error"):
                node = case.find(tag)
                if node is not None:
                    msg = (
                        node.attrib.get("message", "")
                        + " "
                        + (node.text or "")
                    )
                    errors[key] = msg
    return errors


def main():
    errors = parse_test_errors(XML_PATH)
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        header = reader[1]
        data = reader[2:]
    idx_status = header.index("Status")
    idx_file = header.index("File Path")
    idx_name = header.index("Test Name")
    results = []
    for row in data:
        status = row[idx_status].strip()
        file_path = row[idx_file]
        test_name = row[idx_name]
        classname = file_path.replace("/", ".").replace(".py", "")
        key = (classname, test_name)
        justification = ""
        # Status N/A o error
        if status == "N/A":
            justification = "Status N/A: not categorized by automation."
        elif key in errors:
            msg = errors[key].lower()
            if any(k in msg for k in target_keywords):
                justification = (
                    f"Error message suggests manual intervention: "
                    f"{errors[key]}"
                )
        if justification:
            results.append(f"- {test_name} ({file_path}): {justification}")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        if results:
            f.write("Tests requiring manual intervention:\n")
            f.write("\n".join(results))
        else:
            f.write(
                "No tests require manual intervention based on current "
                "criteria."
            )


if __name__ == "__main__":
    main()
