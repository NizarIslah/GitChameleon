import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

# --- CONFIGURATION ----------------------------------------------------------

# Define colors based on the image
BASELINE_COLOR = "#dadaeb"  # Light purple/grey
ERROR_FEEDBACK_COLOR = "#54278f" # Dark purple

# Define models for the right-hand plots based on the image
models_for_grid = [
    "Qwen2-72B",
    "Codestral-22B",
    "GPT-4o",
]

# --- SYNTHETIC DATA GENERATION ----------------------------------------------
# This function generates data that roughly simulates the distributions
# shown in your reference image.

def generate_synthetic_data(n_samples_total=1000):
    """Generates synthetic pass@10 data for plotting."""
    data = {
        "Baseline": [],
        "Error Feedback": []
    }

    # Simulate data for the left histogram (overall difficulty)
    # Baseline: skewed towards higher pass@10 but with a peak around 30-40
    data["Baseline"].extend(np.random.normal(45, 20, int(n_samples_total * 0.6)).tolist())
    data["Baseline"].extend(np.random.normal(15, 10, int(n_samples_total * 0.4)).tolist())

    # Error Feedback: skewed towards higher pass@10, peak around 40-50
    data["Error Feedback"].extend(np.random.normal(55, 25, int(n_samples_total * 0.7)).tolist())
    data["Error Feedback"].extend(np.random.normal(25, 15, int(n_samples_total * 0.3)).tolist())

    # Clamp data to [0, 100]
    data["Baseline"] = [max(0, min(100, x)) for x in data["Baseline"]]
    data["Error Feedback"] = [max(0, min(100, x)) for x in data["Error Feedback"]]

    # Simulate data for the right grid plots (specific models, binned data)
    # This is a simplification to match the *counts* and *bins* in the image (0, 50, 100)
    model_data = {
        "Qwen2-72B": {"Baseline": {0: 68, 50: 6, 100: 42}, "Error Feedback": {0: 66, 50: 6, 100: 44}},
        "Codestral-22B": {"Baseline": {0: 71, 50: 2, 100: 43}, "Error Feedback": {0: 67, 50: 5, 100: 44}},
        "GPT-4o": {"Baseline": {0: 62, 50: 15, 100: 39}, "Error Feedback": {0: 61, 50: 5, 100: 50}},
    }
    # Convert counts to a list of values for histogram plotting
    model_hist_data = {}
    for model, conditions in model_data.items():
        model_hist_data[model] = {}
        for cond, counts in conditions.items():
            values = []
            for bin_val, count in counts.items():
                # Append the bin value 'count' times
                values.extend([bin_val] * count)
            model_hist_data[model][cond] = values


    return data, model_hist_data

# --- PLOTTING -------------------------------------------------------------

def main():
    overall_data, model_hist_data = generate_synthetic_data(n_samples_total=1000) # Generate data

    # --- Figure 1: Overall Difficulty Histogram ---
    fig1, ax_left = plt.subplots(1, 1, figsize=(8, 6)) # Single plot

    bins = np.arange(0, 101, 10) # Bins from 0 to 100, step 10
    ax_left.hist(
        [overall_data["Baseline"], overall_data["Error Feedback"]],
        bins=bins,
        color=[BASELINE_COLOR, ERROR_FEEDBACK_COLOR],
        label=["Baseline", "+ Error Feedback"],
        edgecolor='black',
        alpha=0.9,
        density=False # Show counts, not density
    )
    ax_left.set_title("Difficulty of samples", fontsize=20, pad=10, fontweight='bold')
    ax_left.set_xlabel("Success Rate (%)", fontsize=12, fontweight='bold')
    ax_left.set_ylabel("Frequency", fontsize=12, fontweight='bold')
    ax_left.set_xlim(0, 100)
    ax_left.grid(axis='y', linestyle='--', alpha=0.5) # Grid on y-axis as in the image
    ax_left.legend(fontsize=14)

    plt.tight_layout() # Adjust layout
    plt.savefig("difficulty_overall.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig1) # Close the first figure

    # --- Figure 2: Model-Specific Grid Histograms ---
    fig2, axs_right = plt.subplots(3, 2, figsize=(10, 12), sharey=False) # 3x2 grid, removed sharey

    grid_bin_labels = ['0%', '50%', '100%']

    conditions = ["Baseline", "Error Feedback"]

    for i, model_label in enumerate(models_for_grid):
        for j, condition in enumerate(conditions):
            ax = axs_right[i, j] # Access subplot using row, column index

            # Categories we want to plot bars for
            categories_percent = [0, 50, 100]
            # Get the counts for each category from the synthetic data
            counts = [model_hist_data[model_label][condition].count(cat) for cat in categories_percent]

            # Positions for the center of the bars on the x-axis
            bar_positions = np.arange(len(categories_percent))
            bar_width = 0.8 # Width of the bars

            rects = ax.bar(
                bar_positions,
                counts,
                bar_width,
                color=BASELINE_COLOR if condition == "Baseline" else ERROR_FEEDBACK_COLOR,
                edgecolor='black',
                alpha=0.9,
                label=condition
            )

            # Add value labels on top of bars
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    ax.text(rect.get_x() + rect.get_width()/2., height + 1,
                            '%d' % int(height),
                            ha='center', va='bottom', fontsize=10)


            ax.set_title(f"{model_label} ({condition if condition == 'Baseline' else '+ Error Feedback'})", fontsize=12, fontweight='bold')
            ax.set_ylabel("Frequency", fontsize=10)
            # Adjust y-limit based on the counts in the *current* subplot
            ax.set_ylim(0, max(counts) * 1.2 if max(counts) > 0 else 5) # Ensure a minimum y-limit

            # Set x-axis ticks to be at the center of the bars and labels
            ax.set_xticks(bar_positions)
            ax.set_xticklabels(grid_bin_labels, fontsize=10)
            ax.set_xlabel("Success Rate (%)", fontsize=10)
            ax.grid(axis='y', linestyle='--', alpha=0.5) # Grid on y-axis

            # Set x-limits to frame the bars nicely
            ax.set_xlim(-bar_width/2, len(categories_percent) - bar_width/2)


    # Create a legend for the second figure (optional, can add to fig1 instead)
    # If adding to fig2, need to place it outside the grid.
    # For now, relying on the legend in the first figure is likely sufficient based on the image.
    # If a legend is needed here, use fig2.legend(...) similar to the original code.

    plt.tight_layout() # Adjust layout
    plt.savefig("difficulty_models_grid.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig2) # Close the second figure

    # Display the plots (optional - useful for interactive sessions)
    # plt.show() # Uncomment this if you want to see the plots displayed after saving

if __name__ == "__main__":
    main()