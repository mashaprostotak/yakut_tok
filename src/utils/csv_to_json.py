import json
import pandas as pd

# === FILE PATHS ===
csv_file = 'corrected_data.csv'
json_file = 'processed/synthetic_dataset_yakut.json'
output_file = 'updated_data.json'

# === READ FILES ===
df = pd.read_csv(csv_file)
with open(json_file, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# === CHECK LENGTH ===
if len(json_data) != len(df):
    raise ValueError("CSV and JSON do not have the same number of records!")

# === REPLACE FIELDS ===
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

# === WRITE UPDATED JSON ===
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print(f"Updated JSON saved to {output_file}")
