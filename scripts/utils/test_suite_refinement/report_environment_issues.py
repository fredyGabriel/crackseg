import xml.etree.ElementTree as ET

# ruff: noqa: E501
XML_PATH = "outputs/prd_project_refinement/test_suite_evaluation/reports/test_results.xml"
REPORT_PATH = "outputs/prd_project_refinement/test_suite_evaluation/reports/environment_issues.txt"

# Palabras clave para identificar problemas de entorno
env_keywords = [
    "FileNotFoundError",
    "No such file or directory",
    "Config directory not found",
    "Hydra initialize/compose failed",
    "ModuleNotFoundError",
    "ImportError",
    "OSError",
    "Permission denied",
    "EnvironmentError",
    "missing",
    "not found",
    "failed to import",
    "cannot import",
    "dependency",
    "not installed",
    "not available",
    "CUDA",
    "device",
    "path",
    "directory",
    "config",
    "environment variable",
    "KeyError",
]


def extract_environment_issues(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    issues = []
    for suite in root.findall(".//testsuite"):
        for case in suite.findall("testcase"):
            name = case.attrib.get("name", "")
            classname = case.attrib.get("classname", "")
            # Buscar errores y fallos
            for tag in ["failure", "error"]:
                node = case.find(tag)
                if node is not None:
                    msg = (
                        node.attrib.get("message", "")
                        + "\n"
                        + (node.text or "")
                    )
                    for kw in env_keywords:
                        if kw.lower() in msg.lower():
                            issues.append(
                                f'- {classname}.{name}: {
                                    msg.strip().replace(chr(10), "; ")}'
                            )
                            break
    return issues


def write_report(issues, report_path):
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Environment-specific issues detected in test suite\n")
        if not issues:
            f.write("No environment-related issues detected.\n")
        else:
            for issue in issues:
                f.write(issue + "\n")


if __name__ == "__main__":
    issues = extract_environment_issues(XML_PATH)
    write_report(issues, REPORT_PATH)
