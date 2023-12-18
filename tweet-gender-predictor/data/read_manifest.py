import json

# Path to the JSON Lines file
manifest_data = r"C:\Users\Peant\OneDrive\Desktop\Narratize Data\tweet-gender-predictor\data\manifest.jl"

# Open the file and read each line
with open(manifest_data, 'r', encoding='utf-8') as file:
    for line in file:
        # Parse the JSON object in each line
        data = json.loads(line)
        # Print the parsed data
        print(data)