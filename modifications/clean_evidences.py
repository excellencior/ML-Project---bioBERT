import json

def remove_french_data(input_file, output_file):
    # Load the JSON data from the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Iterate over the items in the JSON
    for key, value in data.items():
        # Remove 'question_fr' if present
        if 'question_fr' in value:
            del value['question_fr']
        
        # Remove 'fr' keys from value_meaning if present
        if 'value_meaning' in value:
            for v_key in value['value_meaning']:
                if 'fr' in value['value_meaning'][v_key]:
                    del value['value_meaning'][v_key]['fr']
    
    # Save the cleaned data back to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Input and output file names
input_file = 'release_evidences.json'
output_file = 'modifications//Clean//release_evidences_cleaned.json'

# Run the function
remove_french_data(input_file, output_file)

print(f"French data removed. Cleaned JSON saved to {output_file}")
