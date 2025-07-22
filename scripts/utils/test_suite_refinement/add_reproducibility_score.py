"""
Add reproducibility score to test inventory. This script calculates
and adds a reproducibility score to the test inventory based on test
consistency across multiple runs.
"""

import csv

# Type definitions
type CsvRow = list[str]
type CsvData = list[CsvRow]

CSV_PATH = (
    "outputs/prd_project_refinement/test_suite_evaluation/reports/"
    "test_inventory.csv"
)

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = list(csv.reader(f))
    explanation: CsvRow = reader[0]
    header: CsvRow = reader[1]
    data: CsvData = reader[2:]

# Document the criteria in the explanation
explanation = [
    "# Reproducibility Score: 1.0 if test is always passing or "
    "always failing in both runs; 0.5 if flaky (result changes "
    "between runs)."
]

# Add column if it doesn't exist
if "Reproducibility Score" not in header:
    header.append("Reproducibility Score")
    add_score = True
    idx_score = len(header) - 1  # Last index after append
else:
    add_score = False
    idx_score = header.index("Reproducibility Score")

idx_status = header.index("Status")

updated: CsvData = [explanation, header]
for row in data:
    status = row[idx_status].strip().lower()
    if status in ("passing", "failing"):
        score = "1.0"
    elif status == "flaky":
        score = "0.5"
    else:
        score = "N/A"
    if add_score:
        row.append(score)
    else:
        row[idx_score] = score
    updated.append(row)

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(updated)
