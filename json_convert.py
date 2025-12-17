import json
import csv

INPUT_FILE = "qwen25_math_outputs.json"
OUTPUT_FILE = "qwen25_math_outputs.csv"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Define CSV columns explicitly (stable + readable)
fieldnames = [
    "section",
    "problem",
    "model_output"
]

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for row in data:
        writer.writerow({
            "section": row.get("section", ""),
            "problem": row.get("problem", ""),
            "model_output": row.get("model_output", "")
        })

print(f"Saved CSV to {OUTPUT_FILE}")
