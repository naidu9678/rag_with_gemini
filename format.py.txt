import json
import pandas as pd

# 1. Load your raw annotations JSON
with open('annotations.json', 'r', encoding='utf-8') as f:
    raw = json.load(f)

# 2. Flatten into a simple list of dicts
rows = []
for entry in raw:
    text = entry.get('text', '')
    classes = entry.get('annotations', {}).get('classes', [])
    label = classes[0]['name'] if classes else ''
    rows.append({'text': text, 'label': label})

# 3. Create a DataFrame and save to CSV
df = pd.DataFrame(rows)
df.to_csv('train.csv', index=False)

print(df.head())
