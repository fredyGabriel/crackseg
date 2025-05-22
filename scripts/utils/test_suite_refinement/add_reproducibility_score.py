import csv

CSV_PATH = "outputs/prd_project_refinement/test_suite_evaluation/reports/"
CSV_PATH += "test_inventory.csv"

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = list(csv.reader(f))
    explanation = reader[0]
    header = reader[1]
    data = reader[2:]

# Documentar el criterio en la explicación
explanation = "# Reproducibility Score: 1.0 if test is always passing or "
explanation += "always failing in both runs; 0.5 if flaky (result changes "
explanation += "between runs)."

# Añadir columna si no existe
if "Reproducibility Score" not in header:
    header.append("Reproducibility Score")
    add_score = True
else:
    add_score = False
    idx_score = header.index("Reproducibility Score")

idx_status = header.index("Status")

updated = [explanation, header]
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
