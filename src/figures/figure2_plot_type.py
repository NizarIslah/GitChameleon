import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import re

# Path to your JSONL dataset
dataset = '/Users/beike/Desktop/Workspace/GitChameleon/dataset/final_fix_dataset.jsonl'

def categorize(change: str) -> str:
    """
    Map any raw change description into one of:
      • Argument or Attribute change
      • Function Name change
      • Semantics or Function Behavior change
      • New feature or additional dependency-based change
      • Other/Unmatched
    """
    s = change.strip().lower()

    # 1) Argument or Attribute change
    if re.search(r'\b(argument|attribute|param)\b', s):
        return 'Argument or Attribute change'

    # 2) New feature or additional dependency-based change
    elif re.search(r'\b(new|feature|introduc|dependency|additional)\b', s):
        return 'New feature or additional dependency-based change'

    # 3) Function Name change
    elif re.search(r'\b(name change|rename|function|func|method|class)\b', s):
        return 'Function Name change'

    # 4) Semantics or Function Behavior change
    elif re.search(
        r'\b(semantic|behaviour|behavior|runtime|breaking|deprecate|deprecation|output|return)\b',
        s
    ):
        return 'Semantics or Function Behavior change'

    # 5) Anything else
    else:
        return 'Other/Unmatched'


# ---- Step 1: List all unique raw values ----
unique_types = set()
with open(dataset, 'r') as f:
    for line in f:
        raw = json.loads(line).get('type_of_change', '').strip()
        if raw:
            unique_types.add(raw)

print("Unique raw 'type_of_change' values:")
for t in sorted(unique_types):
    print(f" - {t}")


# ---- Step 2: Categorize & count ----
counter = Counter()
total = 0
with open(dataset, 'r') as f:
    for line in f:
        raw = json.loads(line).get('type_of_change', '')
        cat = categorize(raw)
        counter[cat] += 1
        total += 1

print(f"\nTotal samples processed: {total}")
print("Counts by mapped category:")
for cat in [
    'Argument or Attribute change',
    'Function Name change',
    'Semantics or Function Behavior change',
    'New feature or additional dependency-based change',
    'Other/Unmatched'
]:
    print(f" - {cat}: {counter.get(cat, 0)}")


# ---- Step 3: Plot the distribution ----
categories = [
    'Argument or Attribute change',
    'Function Name change',
    'Semantics or Function Behavior change',
    'New feature or additional dependency-based change',
]
# Pull counts in that order
counts = [counter.get(cat, 0) for cat in categories]

# Define short labels for plotting
short_labels = ['Argument', 'Function name', 'Semantics', 'New feature']

# Plot
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

ax.bar(
    np.arange(len(categories)),
    counts,
    width=0.6,
    color='royalblue',
    edgecolor='k',
    alpha=0.9
)

ax.set_xticks(np.arange(len(categories)))
ax.set_xticklabels(short_labels, fontsize=14)
ax.set_xlabel('Change Category', fontsize=16)
ax.set_ylabel('Sample Count', fontsize=16)
ax.tick_params(axis='y', labelsize=14)
ax.grid(axis='y', linestyle='--', alpha=0.5)

output_png = 'type_change.png'
plt.savefig(output_png, bbox_inches='tight', transparent=True)

plt.show()