import json
import re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

# --- User‑adjustable paths ---
dataset_path = '/Users/beike/Desktop/Workspace/GitChameleon/dataset/final_fix_dataset.jsonl'
output_figure_pdf = '/Users/beike/Desktop/Workspace/GitChameleon/release_year_distribution.pdf'

# 1. Read all release_date strings
all_dates = []
with open(dataset_path, 'r') as f:
    for line in f:
        obj = json.loads(line)
        rd = obj.get('release_date')
        if rd:
            all_dates.append(rd)

# 2. Extract 4‑digit year from each string
year_pattern = re.compile(r'(\d{4})')
years = []
missing = []
for idx, raw in enumerate(all_dates):
    m = year_pattern.search(raw)
    if m:
        years.append(int(m.group(1)))
    else:
        missing.append((idx, raw))

# 3. Report any that had no 4‑digit year
if missing:
    print("\nStrings without a 4‑digit year (will be skipped):")
    for idx, raw in missing:
        print(f"  Index {idx:3d}: '{raw}'")
else:
    print("\nAll strings yielded a year!")

# 4. Count per year
year_counts = pd.Series(years).value_counts().sort_index()
print("\nRaw year counts:")
print(year_counts.to_string())

# 5. Align to 2014–2023
years_range = list(range(2014, 2024))
counts = [year_counts.get(y, 0) for y in years_range]
print(f"\nCounts 2014–2023: {dict(zip(years_range, counts))}")

# 6. Plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.bar(years_range, counts, color='royalblue', width=0.75, edgecolor='black', alpha=0.9)
ax.set_xticks(years_range)
ax.set_xticklabels(years_range, fontsize=14)
ax.set_xlabel('Version Release Year', fontsize=16)
ax.set_ylabel('Sample Count', fontsize=16)
ax.set_title('Release Year Distribution', fontsize=18)
ax.grid(axis='y', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig(output_figure_pdf, bbox_inches='tight')
plt.show()

print(f"\nFigure saved to: {output_figure_pdf}")
