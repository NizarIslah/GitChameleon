import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

# --- User‑adjustable paths ---
dataset_path = '/Users/beike/Desktop/Workspace/GitChameleon/dataset/final_fix_dataset.jsonl'
output_pdf   = 'version_count.pdf'

# 1. Build a mapping: library → set of versions
versions_per_lib = defaultdict(set)
with open(dataset_path, 'r') as f:
    for line in f:
        obj = json.loads(line)
        lib     = obj.get('library')
        version = obj.get('version')
        if lib and version:
            versions_per_lib[lib].add(version)

# 2. Prepare data for plotting
libraries = list(versions_per_lib.keys())
unique_counts = [len(versions_per_lib[lib]) for lib in libraries]

# 3. Plot
fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_minor_locator(MultipleLocator(1))

x = np.arange(len(libraries))
ax.bar(x, unique_counts, width=0.75, color='royalblue', edgecolor='k', alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(libraries, fontsize=12, rotation=90)
ax.set_xlabel('Library', fontsize=20)
ax.set_ylabel('Number of Unique Versions', fontsize=20)
ax.yaxis.set_tick_params(labelsize=15)
ax.grid(axis='y', color='grey', linestyle='--', linewidth=1, alpha=0.5)

plt.savefig(output_pdf, bbox_inches='tight')
plt.show()

print(f"\nFigure saved to: {output_pdf}")
