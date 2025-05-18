import argparse
import json
import matplotlib.pyplot as plt
import numpy as np


def load_counts(json_path):
    """Load a JSON file mapping categories to counts."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected JSON object mapping categories to counts in {json_path}"
        )
    return data


def main():
    # ------------------------------------------------------------------
    # Use CLI args if desired (commented). Currently hard-coded paths.
    # ------------------------------------------------------------------
    # parser = argparse.ArgumentParser(
    #     description="Plot error-category frequencies for baseline vs. self-debug from two JSON files."
    # )
    # parser.add_argument("--baseline", required=True, help="Baseline counts JSON (category→count)")
    # parser.add_argument("--feedback", required=True, help="Self-debug counts JSON (category→count)")
    # parser.add_argument("--output", help="Save figure to this path instead of showing interactively.")
    # args = parser.parse_args()

    baseline_counts = load_counts("gpt_errors.json")  # replace with args.baseline
    feedback_counts = load_counts("gpt_debug_errors.json")  # replace with args.feedback

    # ------------------------------------------------------------------
    # Focus on top-10 baseline categories for readability.
    # ------------------------------------------------------------------
    baseline_counts = dict(
        sorted(baseline_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    )
    categories = list(baseline_counts.keys())
    baseline = [baseline_counts.get(cat, 0) for cat in categories]
    feedback = [feedback_counts.get(cat, 0) for cat in categories]

    # Positive → reduction (% fewer errors); negative → increase
    percentage_delta = [
        (baseline[i] - feedback[i]) / baseline[i] * 100 if baseline[i] else 0
        for i in range(len(baseline))
    ]

    # ------------------------------------------------------------------
    # Build a single-axis figure to minimise vertical whitespace.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(
        x - width / 2,
        baseline,
        width,
        label="Greedy Decoding",
        color="#e0d5f9",
        edgecolor="black",
        alpha=0.85,
    )
    bars_sd = ax.bar(
        x + width / 2,
        feedback,
        width,
        label="Self-Debug",
        color="#8e44ad",
        alpha=0.9,
    )

    # Add vertical Δ% labels centered above each Self-Debug bar.
    max_count = max(max(baseline), max(feedback))
    for idx, bar in enumerate(bars_sd):
        height = bar.get_height()
        delta_text = f"{percentage_delta[idx]:.1f}%"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max_count * 0.03,
            delta_text,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=16,
            fontweight="bold",
        )

    # Axis styling
    ax.set_ylabel("Total Errors", fontsize=24)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=20)
    ax.set_xlabel("Error Categories", fontsize=24)
    ax.tick_params(axis="y", labelsize=20)
    ax.legend(frameon=True, fontsize=20)
    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)

    # Extend y-limit so delta labels fit inside figure.
    ax.set_ylim(0, max_count * 1.25)

    plt.tight_layout()
    fig.savefig("error_category_comparison_with_delta.pdf", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
