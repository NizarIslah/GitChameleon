import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

# --- User‑adjustable paths ---
dataset_path = '/Users/beike/Desktop/Workspace/GitChameleon/dataset/final_fix_dataset.jsonl'
output_png   = 'samples_per_library.pdf'

# 1. Build a mapping: library → count of samples
counts_per_lib = defaultdict(int)
with open(dataset_path, 'r') as f:
    for line in f:
        obj = json.loads(line)
        lib = obj.get('library')
        if lib:
            counts_per_lib[lib] += 1

# 2. Prepare data for plotting
libraries = list(counts_per_lib.keys())
sample_counts = [counts_per_lib[lib] for lib in libraries]

# 3. Plot
fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_minor_locator(MultipleLocator(1))

x = np.arange(len(libraries))
ax.bar(x, sample_counts, width=0.75, color='royalblue', edgecolor='k', alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(libraries, fontsize=12, rotation=90)
ax.set_xlabel('Library', fontsize=20)
ax.set_ylabel('Number of Samples', fontsize=20)
ax.yaxis.set_tick_params(labelsize=15)
ax.grid(axis='y', color='grey', linestyle='--', linewidth=1, alpha=0.5)

plt.savefig(output_png, bbox_inches='tight', transparent=False)
plt.show()

print(f"\nFigure saved to: {output_png}")
