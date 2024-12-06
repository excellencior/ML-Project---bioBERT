import json
import csv

# Load JSON data from file
with open('release_evidences.json', 'r') as file:
    data = json.load(file)

# Prepare the CSV file
output_file = 'modifications//Clean//questions.csv'
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # Write header row
    csv_writer.writerow(['Code', 'Question (en)', 'Possible Values (en)'])
    
    # Iterate through each question in the JSON
    for code, details in data.items():
        question = details.get('question_en', '')
        possible_values = details.get('possible-values', [])
        
        # Translate possible values to their 'en' meanings
        value_meaning = details.get('value_meaning', {})
        possible_values_en = [
            value_meaning[val]['en'] for val in possible_values if val in value_meaning
        ]
        
        # Write the row to the CSV
        csv_writer.writerow([code, question, '; '.join(possible_values_en)])

print(f"Data has been written to {output_file}")
