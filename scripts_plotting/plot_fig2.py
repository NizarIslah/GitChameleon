import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

def plot_library_counts(jsonl_path, save_path):
    # Count occurrences of each library
    counts = Counter()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            lib = rec.get("library")
            if lib:
                counts[lib] += 1

    # Sort libraries alphabetically (or by count: .most_common())
    libs = sorted(counts)
    vals = [counts[lib] for lib in libs]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(libs, vals, color="royalblue")
    ax.set_xlabel("Library")
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(libs)))
    ax.set_xticklabels(libs, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    fig.savefig(save_path)

def plot_unique_versions(jsonl_path, save_path):
    save_path = str(save_path)[:-4] + "_unique.pdf"
    # Count unique versions of each library
    version_counts = Counter()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            lib = rec.get("library")
            version = rec.get("version")
            if lib and version:
                version_counts[(lib, version)] += 1

    # Sort libraries and versions
    libs = sorted(set(lib for lib, _ in version_counts))
    vals = [len({ver for lib, ver in version_counts if lib == l}) for l in libs]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(libs, vals, color="royalblue")
    ax.set_xlabel("Library")
    ax.set_ylabel("Unique Versions")
    ax.set_xticks(range(len(libs)))
    ax.set_xticklabels(libs, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    fig.savefig(save_path)

def plot_type_of_change(jsonl_path, save_path):
    # Count occurrences of each type of change
    save_path = str(save_path)[:-4] + "_type_of_change.pdf"
    change_counts = Counter()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            change_type = rec.get("type_of_change")
            if change_type:
                change_counts[change_type] += 1

    # Sort types of changes
    types = sorted(change_counts)
    vals = [change_counts[t] for t in types]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(types, vals, color="royalblue")
    ax.set_xlabel("Type of Change")
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    fig.savefig(save_path)

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Read a JSONL with 'library' fields and plot a bar chart of counts."
    )
    p.add_argument(
        "jsonl_file",
        type=Path,
        help="Path to the input JSONL file (one record per line, with a 'library' key)."
    )
    p.add_argument(
        "--output",
        type=Path,
        default="scripts_plotting/fig2d.pdf",
        help="Path to save the output plot (default: fig2d.pdf)."
    )
    args = p.parse_args()
    plot_library_counts(args.jsonl_file, args.output)
    plot_unique_versions(args.jsonl_file, args.output)
    plot_type_of_change(args.jsonl_file, args.output)
