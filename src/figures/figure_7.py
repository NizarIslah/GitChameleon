import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # Needed for custom legend handles
import random

# --- CONFIGURATION ----------------------------------------------------------

# Define colors based on the image
BASELINE_COLOR = "#dadaeb"  # Light purple/grey
ERROR_FEEDBACK_COLOR = "#54278f" # Dark purple

# Define the error categories based on the x-axis labels in the image
error_categories = [
    'Name',
    'Indentation',
    'Syntax',
    'Attribute',
    'Import',
    'AssertionError',
    'ModuleNotFoundError',
    'TimeoutError',
]

# --- SYNTHETIC DATA GENERATION ----------------------------------------------
# This function generates frequency data that roughly simulates the bar heights
# shown in your reference image.

def generate_synthetic_data():
    """Generates synthetic frequency data for error categories."""
    # Manually set frequencies based on estimating bar heights from the image
    synthetic_frequencies = {
        'Name': {'Baseline': 37, 'Error Feedback': 31},
        'Indentation': {'Baseline': 14, 'Error Feedback': 11.5},
        'Syntax': {'Baseline': 11, 'Error Feedback': 12},
        'Attribute': {'Baseline': 8, 'Error Feedback': 6.5},
        'Import': {'Baseline': 7.5, 'Error Feedback': 7},
        'AssertionError': {'Baseline': 5.5, 'Error Feedback': 5},
        'ModuleNotFoundError': {'Baseline': 4.5, 'Error Feedback': 3},
        'TimeoutError': {'Baseline': 1, 'Error Feedback': 14.5},
    }
    return synthetic_frequencies

# --- PLOTTING -------------------------------------------------------------

def main():
    frequencies = generate_synthetic_data() # Generate data

    fig, ax = plt.subplots(1, 1, figsize=(10, 7)) # Single plot

    n_categories = len(error_categories)
    bar_width = 0.35 # Width of each individual bar in a group
    # Positions for the left edge of the first bar in each group
    group_positions = np.arange(n_categories)

    # Plotting the bars
    # Offset for the 'Baseline' bars (slightly left of the group position center)
    rects1 = ax.bar(
        group_positions - bar_width/2,
        [frequencies[cat]['Baseline'] for cat in error_categories],
        bar_width,
        label='Baseline',
        color=BASELINE_COLOR,
        edgecolor='black',
        alpha=0.9
    )

    # Offset for the '+ Error Feedback' bars (slightly right of the group position center)
    rects2 = ax.bar(
        group_positions + bar_width/2,
        [frequencies[cat]['Error Feedback'] for cat in error_categories],
        bar_width,
        label='+ Error Feedback',
        color=ERROR_FEEDBACK_COLOR,
        edgecolor='black',
        alpha=0.9
    )

    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_xlabel('Error Categories', fontsize=12, fontweight='bold')
    ax.set_title('Frequency of Error Categories', fontsize=14, fontweight='bold') # Example title

    # Set the position of the x ticks to be in the middle of the groups
    ax.set_xticks(group_positions)
    # Set the labels for the x ticks and rotate them
    ax.set_xticklabels(error_categories, rotation=45, ha='right', fontsize=10) # Rotate for readability

    # Set y-axis limits - based on the max frequency in the synthetic data
    max_freq = max(max(freq.values()) for freq in frequencies.values())
    ax.set_ylim(0, max_freq * 1.1) # Add 10% padding at the top

    # Add grid lines (only on y-axis as in the image)
    ax.grid(axis='y', linestyle='--', alpha=0.6)


    # Create legend using patches
    baseline_patch = mpatches.Patch(facecolor=BASELINE_COLOR, edgecolor='black', label='Baseline')
    error_feedback_patch = mpatches.Patch(facecolor=ERROR_FEEDBACK_COLOR, edgecolor='black', label='+ Error Feedback')
    ax.legend(handles=[baseline_patch, error_feedback_patch], loc='upper right', fontsize=10) # Position legend


    # Adjust layout to prevent labels from overlapping
    plt.tight_layout()

    # Save the figure to a PDF
    plt.savefig("error_category_bars.pdf", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()