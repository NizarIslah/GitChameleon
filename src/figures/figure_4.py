import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines # Needed for custom legend handles
import random
import itertools # For cycling through markers and colors

# --- CONFIGURATION ----------------------------------------------------------

# Define the benchmarks for the x-axes
benchmarks_x = ["HumanEval", "EvalPlus", "BigCodeBench"]
benchmark_y = "GitChameleon" # The common metric for the y-axis

# Define available markers and colors for random assignment
AVAILABLE_MARKERS = ['o', 's', '^', 'P', 'D', '*', 'X', '+', 'v', '<', '>', 'p', 'h', 'H']
# Using a set of distinct colors (you can customize this list)
AVAILABLE_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

# --- SYNTHETIC DATA GENERATION ----------------------------------------------
# This function generates data that roughly simulates the scatter plot
# positions in your reference image.

def generate_synthetic_data():
    """Generates synthetic success rate data for plotting."""
    # Manually set data points to roughly match the image positions
    # This is a simplification; real data would come from evaluation results.
    # Format: {model_name: {benchmark_name: success_rate, ...}, ...}
    synthetic_data = {
        "CodeQwen1.5-Chat": {benchmark_y: 21.5, "HumanEval": 50, "EvalPlus": 42, "BigCodeBench": 15},
        "StarCoder-2":      {benchmark_y: 22.5, "HumanEval": 40, "EvalPlus": 50, "BigCodeBench": 20},
        "LLama-3.1":        {benchmark_y: 16,   "HumanEval": 82, "EvalPlus": 65, "BigCodeBench": 15},
        "Phi-3.5-Mini":     {benchmark_y: 27,   "HumanEval": 80, "EvalPlus": 60, "BigCodeBench": 18},
        "Code-Llama":       {benchmark_y: 20.5, "HumanEval": 45, "EvalPlus": 45, "BigCodeBench": 16},
        "Qwen2.5-Coder":    {benchmark_y: 21.5, "HumanEval": 78, "EvalPlus": 78, "BigCodeBench": 21},
        "Qwen2":            {benchmark_y: 25,   "HumanEval": 85, "EvalPlus": 80, "BigCodeBench": 25},
        "granite-code":     {benchmark_y: 28,   "HumanEval": 70, "EvalPlus": 70, "BigCodeBench": 29},
        # Add some more synthetic models to test random assignment
        "NewModelA":        {benchmark_y: 18, "HumanEval": 60, "EvalPlus": 55, "BigCodeBench": 10},
        "NewModelB":        {benchmark_y: 24, "HumanEval": 75, "EvalPlus": 70, "BigCodeBench": 22},
    }
    return synthetic_data

def assign_model_styles(model_names):
    """Assigns a unique marker and color to each model name."""
    marker_cycle = itertools.cycle(AVAILABLE_MARKERS)
    color_cycle = itertools.cycle(AVAILABLE_COLORS)

    model_styles_dict = {}
    for name in model_names:
        assigned_marker = next(marker_cycle)
        assigned_color = next(color_cycle)
        # Store style as (marker, color) - no outlined styles for random assignment simplicity
        model_styles_dict[name] = (assigned_marker, assigned_color)

    # Convert to a list of tuples (model_name, marker, color)
    model_styles_list = []
    for name in model_names:
        marker, color = model_styles_dict[name]
        model_styles_list.append((name, marker, color))

    return model_styles_list


# --- PLOTTING -------------------------------------------------------------

def main():
    data = generate_synthetic_data() # Generate data
    model_names = list(data.keys()) # Get model names from data
    model_styles = assign_model_styles(model_names) # Dynamically assign styles

    n_benchmarks = len(benchmarks_x)
    fig, axs = plt.subplots(1, n_benchmarks, figsize=(6 * n_benchmarks, 6), sharey=True) # 3 plots horizontally, shared y-axis

    # Ensure axs is an array even if n_benchmarks is 1
    if n_benchmarks == 1:
        axs = [axs]

    # Use tight_layout for subplot spacing and adjust for legend space and margins
    # rect=[left, bottom, right, top] normalized figure coordinates
    # Adding margin on left, right, and top by adjusting the rect
    plt.tight_layout(rect=[0.05, 0.15, 0.95, 0.95])


    for i, benchmark in enumerate(benchmarks_x):
        ax = axs[i]

        # Plot data points for each model on the current benchmark
        for model_name, marker, color in model_styles: # Use dynamically assigned styles
            x_val = data.get(model_name, {}).get(benchmark)
            y_val = data.get(model_name, {}).get(benchmark_y)

            if x_val is not None and y_val is not None:
                # Use the assigned marker and color directly
                ax.scatter(
                    x_val, y_val,
                    marker=marker,
                    color=color,
                    s=100, # Marker size
                    label=model_name # Label for legend
                )


        # Set title and labels
        ax.set_title(f"{benchmark} Success Rate (↑)", fontsize=14, fontweight='bold')
        ax.set_xlabel(f"{benchmark} Success Rate (↑)", fontsize=10, fontweight='bold')
        if i == 0: # Only set y-label for the first plot
            ax.set_ylabel(f"{benchmark_y} Success Rate (↑)", fontsize=10, fontweight='bold')

        # Set axis limits - trying to roughly match ranges, can adjust based on data
        # Dynamic limits based on data could also be implemented here
        x_values = [data.get(name, {}).get(benchmark) for name in model_names if data.get(name, {}).get(benchmark) is not None]
        y_values = [data.get(name, {}).get(benchmark_y) for name in model_names if data.get(name, {}).get(benchmark_y) is not None]

        if x_values:
             # Add a small padding to the limits
             x_min, x_max = min(x_values), max(x_values)
             x_padding = (x_max - x_min) * 0.1
             ax.set_xlim(x_min - x_padding, x_max + x_padding)

        if y_values:
             # Add a small padding to the limits
             y_min, y_max = min(y_values), max(y_values)
             y_padding = (y_max - y_min) * 0.1
             ax.set_ylim(y_min - y_padding, y_max + y_padding)


        ax.grid(True, linestyle='--', alpha=0.6) # Add grid

    # Create a unified legend below the plots
    legend_handles = []
    for model_name, marker, color in model_styles: # Use dynamically assigned styles
         # Create a Line2D object purely for the legend
         handle = mlines.Line2D(
             [], [],
             color=color,
             marker=marker,
             linestyle='None',
             markersize=8 # Match scatter size roughly in legend
         )
         legend_handles.append(handle)

    # Place the legend below the subplots, anchored relative to the figure bottom.
    # With tight_layout rect bottom=0.15, space from y=0 to y=0.15 is available for the legend.
    # bbox_to_anchor=(0.5, 0) with loc='lower center' places the legend's bottom center
    # at the figure's bottom center.
    fig.legend(
        handles=legend_handles,
        labels=[m[0] for m in model_styles],
        loc="lower center", # Anchor point of the legend
        ncol=4, # Adjust number of columns as needed to fit
        frameon=False,
        prop={'size': 10},
        # Anchor at the bottom center of the figure (y=0).
        # tight_layout with rect bottom ensures subplots are above y=0.15.
        # The legend will sit in the space between y=0 and y=0.15.
        bbox_to_anchor=(0.5, 0)
    )


    # Use bbox_inches="tight" when saving to include the legend
    # combined with rect and bbox_to_anchor=(0.5, 0), this should work
    plt.savefig("benchmark_scatter.pdf", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()