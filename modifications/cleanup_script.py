import json

# File paths
input_file = 'release_conditions.json'
output_file = 'modifications//Clean//release_conditions_cleaned.json'

# Function to remove `cond-name-fr` from all objects
def remove_cond_name_fr(data):
    for key in data:
        if "cond-name-fr" in data[key]:
            del data[key]["cond-name-fr"]
        if "cond-name-eng" in data[key]:
            del data[key]["cond-name-eng"]
    return data

# Read, process, and save the JSON data
try:
    # Read the JSON file
    with open(input_file, 'r') as f:
        json_data = json.load(f)
    
    # Remove `cond-name-fr`
    cleaned_data = remove_cond_name_fr(json_data)
    
    # Save the cleaned JSON to a new file
    with open(output_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    print(f"`cond-name-fr` and 'cond-name-eng'  keys removed. Cleaned JSON saved to {output_file}")
except Exception as e:
    print(f"An error occurred: {e}")
