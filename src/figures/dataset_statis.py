import json
import re
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import pandas as pd

# Increase all font sizes by default
plt.rcParams.update({
    'font.size': 22,            # base font size
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
})

# --- User‑adjustable path ---
dataset_path = '/Users/beike/Desktop/Workspace/GitChameleon/dataset/final_fix_dataset.jsonl'
purple = '#8e44ad'

# 1. Samples per library
counts_per_lib = defaultdict(int)
with open(dataset_path, 'r') as f:
    for line in f:
        lib = json.loads(line).get('library')
        if lib:
            counts_per_lib[lib] += 1
libs = list(counts_per_lib.keys())
samples = [counts_per_lib[l] for l in libs]

plt.figure(figsize=(16, 10))
x = np.arange(len(libs))
plt.bar(x, samples, width=0.7, color=purple, alpha=0.9)
plt.xticks(x, libs, rotation=30, ha='right')
plt.xlabel('Library')
plt.ylabel('Number of Samples')
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
plt.grid(True, which='major', axis='y', linestyle='--', linewidth=1, alpha=0.5)
plt.tight_layout()
plt.savefig('samples_per_library.pdf', dpi=300)
plt.clf()

# 2. Unique versions per library
versions_per_lib = defaultdict(set)
with open(dataset_path, 'r') as f:
    for line in f:
        obj = json.loads(line)
        if obj.get('library') and obj.get('version'):
            versions_per_lib[obj['library']].add(obj['version'])
libs_v = list(versions_per_lib.keys())
unique_counts = [len(versions_per_lib[l]) for l in libs_v]

plt.figure(figsize=(16, 10))
x = np.arange(len(libs_v))
plt.bar(x, unique_counts, width=0.7, color=purple, alpha=0.9)
plt.xticks(x, libs_v, rotation=30, ha='right')
plt.xlabel('Library')
plt.ylabel('Number of Unique Versions')
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.grid(True, which='major', axis='y', linestyle='--', linewidth=1, alpha=0.5)
plt.tight_layout()
plt.savefig('version_count.pdf', dpi=300)
plt.clf()

# 3. Release year distribution (2014–2023)
year_pat = re.compile(r'(\d{4})')
years = []
with open(dataset_path, 'r') as f:
    for line in f:
        rd = json.loads(line).get('release_date','')
        m = year_pat.search(rd)
        if m:
            years.append(int(m.group(1)))
yc = pd.Series(years).value_counts().sort_index()
yrange = list(range(2014, 2024))
year_dist = [yc.get(y, 0) for y in yrange]

plt.figure(figsize=(16, 10))
plt.bar(yrange, year_dist, width=0.75, color=purple, alpha=0.9)
plt.xticks(yrange, yrange, rotation=45)
plt.xlabel('Version Release Year')
plt.ylabel('Sample Count')
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.grid(True, which='major', axis='y', linestyle='--', linewidth=1, alpha=0.5)
plt.tight_layout()
plt.savefig('version_year.pdf', dpi=300)
plt.clf()

# 4. Change type distribution
def categorize(t):
    s = t.strip().lower()
    if re.search(r'\b(argument|attribute|param)\b', s):
        return 'Argument'
    if re.search(r'\b(name change|rename|function|method|class)\b', s):
        return 'Function name'
    if re.search(r'\b(semantic|behaviour|behavior|breaking|deprecate)\b', s):
        return 'Semantics'
    if re.search(r'\b(new|feature|dependency)\b', s):
        return 'New feature'
    return 'Other'

counter = Counter()
with open(dataset_path, 'r') as f:
    for line in f:
        raw = json.loads(line).get('type_of_change','')
        if raw:
            counter[categorize(raw)] += 1

labels = ['Argument', 'Function name', 'Semantics', 'New feature', 'Other']
counts = [counter.get(l, 0) for l in labels]

plt.figure(figsize=(16, 10))
x = np.arange(len(labels))
plt.bar(x, counts, width=0.7, color=purple, alpha=0.9)
plt.xticks(x, labels, rotation=30, ha='right')
plt.xlabel('Change Category')
plt.ylabel('Sample Count')
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.grid(True, which='major', axis='y', linestyle='--', linewidth=1, alpha=0.5)
plt.tight_layout()
plt.savefig('type_change.pdf', dpi=300)
plt.clf()

print("All four PDFs generated with increased font sizes.")
