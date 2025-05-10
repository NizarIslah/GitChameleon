import argparse
import json
import matplotlib.pyplot as plt
import numpy as np


def load_counts(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object mapping categories to counts in {json_path}")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Plot error category frequencies for baseline vs. error-feedback from two JSON count files."
    )
    parser.add_argument(
        "--baseline", required=True,
        help="Path to JSON file with baseline counts (category->count)."
    )
    parser.add_argument(
        "--feedback", required=True,
        help="Path to JSON file with error-feedback counts (category->count)."
    )
    parser.add_argument(
        "--title", default="Error Category Comparison",
        help="Plot title."
    )
    parser.add_argument(
        "--output", help="File path to save the figure (e.g. .png). If omitted, shows interactively."
    )
    args = parser.parse_args()

    # Load counts
    baseline_counts = load_counts(args.baseline)
    feedback_counts = load_counts(args.feedback)

    # normalize counts to frequencies
    total_baseline = sum(baseline_counts.values())
    total_feedback = sum(feedback_counts.values())
    baseline_counts = {k: v / total_baseline for k, v in baseline_counts.items()}
    feedback_counts = {k: v / total_feedback for k, v in feedback_counts.items()}

    # Sort categories by baseline frequency, keep only top 10
    baseline_counts = dict(sorted(baseline_counts.items(), key=lambda item: item[1], reverse=True)[:10])

    # Determine ordered categories (union of keys, preserve baseline order then others)
    categories = list(baseline_counts.keys())

    baseline = [baseline_counts.get(cat, 0) for cat in categories]
    feedback = [feedback_counts.get(cat, 0) for cat in categories]

    # Bar positions
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars
    ax.bar(
        x - width/2, baseline, width,
        label='Baseline', color='#e0d5f9', edgecolor='black', alpha=0.8
    )
    ax.bar(
        x + width/2, feedback, width,
        label='+ Self Debug', color='#8e44ad', alpha=0.9
    )

    # Labels and ticks
    ax.set_xlabel('Error Categories', fontsize=16)
    ax.set_ylabel('Frequency', fontsize=16)
    # ax.set_title(args.title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=14)

    # Legend
    ax.legend(frameon=True, fontsize=14)

    # Grid
    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=300)
    else:
        plt.show()


if __name__ == '__main__':
    main()
