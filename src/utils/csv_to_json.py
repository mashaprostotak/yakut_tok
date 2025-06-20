import json
import pandas as pd

CSV_FILE = "corrected_data.csv"
JSON_FILE = "processed/synthetic_dataset_yakut.json"
OUTPUT_FILE = "updated_data.json"

df = pd.read_csv(CSV_FILE)
with open(JSON_FILE, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# Check if the number of records in CSV and JSON match
if len(json_data) != len(df):
    raise ValueError("CSV and JSON do not have the same number of records!")

# Update JSON data with corrected fields from CSV
for i, item in enumerate(json_data):
    item['context'] = df.loc[i, 'corrected yakut context']
    item['question'] = df.loc[i, 'corrected yakut question']
    
    # Handle options parsing
    options = df.loc[i, 'corrected yakut options']
    if isinstance(options, str):
        try:
            item['options'] = json.loads(options)
        except json.JSONDecodeError:
            # Fallback: split by comma
            item['options'] = [opt.strip() for opt in options.split(',')]
    else:
        item['options'] = options
    
    item['answer'] = df.loc[i, 'corrected yakut answer']

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print(f"Updated JSON saved to {OUTPUT_FILE}")
