import xml.etree.ElementTree as ET

# ruff: noqa: E501
XML_PATH = "outputs/prd_project_refinement/test_suite_evaluation/reports/test_results.xml"
REPORT_PATH = "outputs/prd_project_refinement/test_suite_evaluation/reports/slow_tests.txt"


# Parse test durations from XML
def get_test_durations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    durations = []
    for suite in root.findall(".//testsuite"):
        for case in suite.findall("testcase"):
            classname = case.attrib.get("classname", "")
            name = case.attrib.get("name", "")
            time = float(case.attrib.get("time", "0"))
            durations.append((time, classname, name))
    return sorted(durations, reverse=True)


def write_slow_tests_report(durations, report_path, top_n=10):
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Top 10 slowest tests (by duration, descending)\n")
        f.write("Duration (s)\tClassname\tTest Name\n")
        for _i, (time, classname, name) in enumerate(durations[:top_n]):
            f.write(f"{time:.3f}\t{classname}\t{name}\n")


if __name__ == "__main__":
    durations = get_test_durations(XML_PATH)
    write_slow_tests_report(durations, REPORT_PATH)
