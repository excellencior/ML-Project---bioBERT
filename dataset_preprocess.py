import json
import pandas as pd
import ast
import csv

# Load JSON file with evidence descriptions
with open("modifications/Clean/release_evidences_cleaned.json", "r") as f:
    evidence_data = json.load(f)

# Load original dataset to extract metadata
data_row = pd.read_csv("dataset/validate.csv")

# Function to map codes to descriptions
def get_description(evidence_code):
    parts = evidence_code.split("_@_")
    e_code = parts[0]
    v_code = parts[1] if len(parts) > 1 else None

    if e_code in evidence_data:
        question = evidence_data[e_code].get("question_en", "")
        value_desc = (
            evidence_data[e_code]["value_meaning"].get(v_code, {}).get("en", "")
            if v_code else ""
        )
        return f"Question: {question} Answer: {value_desc}"
    return None

# Function to parse dataset and generate BioBERT input
def parse_dataset(data_row, save_to):
    patient_id = 1
    
    # Prepare to write BioBERT inputs
    with open(save_to, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "patient_id", "bio_bert_input", "age", "sex", "pathology"
        ])
        
        for index, row in data_row.iterrows():
            evidences = row['EVIDENCES']
            evidence_list = ast.literal_eval(evidences) if isinstance(evidences, str) else []

            # Map evidence codes to descriptions
            formatted_evidences = [get_description(ev.strip()) for ev in evidence_list]
            formatted_evidences = [ev for ev in formatted_evidences if ev]  # Remove None values

            # Prepare BioBERT input
            bio_bert_inputs = [f"[CLS] {evidence} [SEP]" for evidence in formatted_evidences]

            # Save BioBERT inputs with patient ID and metadata
            for bio_bert_input in bio_bert_inputs:
                writer.writerow([
                    patient_id, 
                    bio_bert_input, 
                    row['AGE'], 
                    row['SEX'], 
                    row['PATHOLOGY'], 
                    # str(row['DIFFERENTIAL_DIAGNOSIS']), 
                ])

            # Increment patient ID for each row
            patient_id += 1

    print("BioBERT inputs with metadata saved")

# Path for saving processed CSV and JSON
save_to = "processed_dataset/csv/validate_biobert.csv"
output_json_path = "processed_dataset/json/validate_biobert.json"

# Parse dataset
parse_dataset(data_row, save_to)

# Convert to JSON
data = {}

# Read the CSV file
with open(save_to, mode="r", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        patient_id = row["patient_id"]
        
        # Group inputs by patient_id
        if f"patient-{patient_id}" not in data:
            data[f"patient-{patient_id}"] = {
                "bio_bert_inputs": [],
                "metadata": {
                    "age": row["age"],
                    "sex": row["sex"],
                    "pathology": row["pathology"],
                    # "differential_diagnosis": row["differential_diagnosis"],
                }
            }
        
        data[f"patient-{patient_id}"]["bio_bert_inputs"].append(row["bio_bert_input"])

# Save to JSON
with open(output_json_path, mode="w", encoding="utf-8") as json_file:
    json.dump(data, json_file, indent=4, ensure_ascii=False)

print(f"JSON file with metadata saved to {output_json_path}")