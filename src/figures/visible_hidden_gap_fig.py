import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # Needed for custom legend handles
import random
from matplotlib.lines import Line2D # For custom legend elements
import matplotlib.cm as cm
# --- CONFIGURATION ----------------------------------------------------------

# Define styles for the two groups (mainly for color and group label)
STYLE_WITH_DEBUG    = {'color': '#ff7f0e', 'label': 'With Self-Debug'}    # Orange

# Define a list of markers for individual models
# Ensuring enough unique markers for the number of models (e.g., 25)
MARKERS = [
    '.', ',', 'o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'D', 'd',
    '|', '_', '1', '2', '3', '4', '8', 'P', 'X', 'Y'
] # List of 25 distinct markers


# --- SYNTHETIC DATA GENERATION ----------------------------------------------
# This function generates synthetic data for models with their hidden/visible
# success rates and self-debug capability.

def generate_synthetic_data(n_models=20):
    """Generates synthetic hidden/visible success rate data for plotting."""
    data = []
    # Removed: model_names = [f"Model_" for i in range(n_models)] as it created non-unique names

    for i in range(n_models):
        model_name = f"Model {i+1}" # Create unique model names

        # Randomly assign self-debug capability
        has_debug = random.choice([True, False])
        debug_status = "With Self-Debug" if has_debug else "Without Self-Debug"

        # Generate synthetic hidden and visible rates (as percentages 0-100)
        # Visible rate is often slightly higher than hidden rate, especially with debug
        base_hidden = random.uniform(40, 80) # Base performance level
        # Ensure visible rate is plausible relative to hidden
        hidden_rate = base_hidden + random.uniform(-5, 5) # Add some variation
        visible_rate = base_hidden + random.uniform(0, 10) # Visible often higher

        # If debugging is present, potentially boost visible rate or reduce variation
        if has_debug:
             visible_rate += random.uniform(0, 8) # Debug might improve visible performance more
             hidden_rate += random.uniform(0, 3) # Debug might slightly improve hidden too

        # Clamp rates to 0-100
        hidden_rate = max(0, min(100, hidden_rate))
        visible_rate = max(0, min(100, visible_rate))

        data.append({
            'model_name': model_name,
            'debug_status': debug_status,
            'hidden_rate': hidden_rate,
            'visible_rate': visible_rate
        })

    return data


def get_data():
    raw_data = """
    model_name & context & hidden & visible & hitrate & cot_hidden & cot_visible & cot_hitrate & debug_hidden & debug_visible & debug_hitrate \\
    Claude 3.7 Sonnet & 2025-02-24 & 48.78 & 55.79 & 19.7 & 45.12 & 56.10 & 23.0 & 64.63 & 82.62 & 33.3 \\
    Gemini 1.5 Pro & 2024-09-24 & 45.12 & 51.52 & 25.3 & 43.29 & 50.00 & 25.3 & 62.50 & 75.30 & 27.9 \\
    Gemini 2.0 Flash & 2025-02-05 & 44.21 & 50.61 & 26.7 & 35.98 & 42.07 & 25.5 & 58.54 & 75.00 & 28.3 \\
    Gemini 2.5 Pro & 2025-03-25 & 50.0 & 60.98 & 25.1 & 49.39 & 61.59 & 25.4 & 64.02 & 83.84 & 27.9 \\
    Gemini 2.5 Flash & 2025-04-17 & 38.11 & 41.77 & 32.0 & 30.79 & 35.67 & 29.9 & 61.28 & 75.91 & 30.3 \\
    GPT-4.1-nano & 2025-04-14 & 33.84 & 35.06 & 24.0 & 11.89 & 14.33 & 22.3 & 40.85 & 49.09 & 27.4 \\
    GPT-4.1-mini & 2025-04-14 & 44.21 & 50.00 & 26.8 & 24.09 & 28.96 & 25.3 & 60.67 & 78.35 & 28.4 \\
    GPT-4.1 & 2025-04-14 & 48.48 & 49.09 & 18.6 & 47.87 & 57.93 & 22.5 & 67.07 & 86.59 &  25.7 \\
    GPT-4o-mini & 2024-07-18 & 37.20 & 46.34 & 18.4 & 35.98 & 42.68 & 17.9 & 46.95 & 59.76 & 18.6 \\
    GPT-4o & 2024-08-06 & 49.09  & 53.96 & 23.1 & 50.30 & 58.23 & 15.9 & 60.76 & 71.04 & 26.8 \\
    GPT-4.5 & 2025-02-27 & 40.85 & 46.04 & 32.8 & 39.94 & 46.34 & 25.7 & 64.02 & 76.22 & 26.8 \\
    o1 & 2024-12-17 & 51.22 & 60.06 & 22.4 & 41.16 & 49.09 & 19.6 & 69.82 & 85.06 & 27.5 \\
    o3-mini & 2025-01-31 & 44.51 & 52.74 & 16.6 & 50.91 & 60.37 & 15.9 & 65.85 & 84.76 & 19.7     
    """
    lines = [x.strip() for x in raw_data.split("\\")]
    headers = [x.strip() for x in lines[0].split("&")]
    data_dict =  [{h: (l.split("&")[idx].strip()) if idx < len(l.split("&")) else 0 for idx, h in enumerate(headers)} for l in lines[1:]]
    
    for d in data_dict:
        try:
            d["debug_hidden"] = float(d["debug_hidden"])
        except:
            d["debug_hidden"] = 0
        try:
            d["debug_visible"] = float(d["debug_visible"])
        except:
            d["debug_visible"] = 0
        try:
            d["hidden"] = float(d["hidden"])
        except:
            d["hidden"] = 0
        try:
            d["visible"] = float(d["visible"])
        except:
            d["visible"] = 0
    
    return data_dict





# --- PLOTTING -------------------------------------------------------------

def main():
    cmap = cm.get_cmap('viridis')
    n_models_to_plot = 25
    data = get_data()
    #data = generate_synthetic_data(n_models=n_models_to_plot)

    # Ensure we have enough markers, cycle if not (though MARKERS list is sized for 25)
    if n_models_to_plot > len(MARKERS):
        print(f"Warning: Number of models ({n_models_to_plot}) exceeds number of unique markers ({len(MARKERS)}). Markers will be recycled.")

    fig, ax = plt.subplots(1, 1, figsize=(10, 7)) # Single scatter plot, slightly wider for legend

    # Plot data points
    for i, model in enumerate(data):
        point_color = STYLE_WITH_DEBUG['color']
        point_marker = MARKERS[i % len(MARKERS)] # Use unique marker for each model

        ax.scatter(
            model['debug_visible'] - model['debug_hidden'],
            model['visible'] - model['hidden'],
            marker=point_marker,
            color=cmap(random.uniform(0, 1)),
            s=100, # Marker size
            label=model['model_name'] # Label each point with its model name for the legend
        )

    # Add the y=x line (zero gap)
    limits = [0, 25] # Assuming rates are percentages
    # Plot the line and ensure it gets a label for the legend
    ax.plot(limits, limits, '--', alpha=0.7)

    # Set labels and title (Corrected axis labels for clarity)
    ax.set_xlabel("Success Rate Gap With Self-Debug (%)", fontsize=20)
    ax.set_ylabel("Success Rate Gap Without Self-Debug (%)", fontsize=15)
    ax.set_title("Success Rate Gap by Self-Debug Capability", fontsize=25, fontweight='bold')

    # Set axis limits to be equal and cover the rate range
    #ax.set_xlim(limits)
    #ax.set_ylim(limits)
    ax.set_aspect('equal', adjustable='box') # Ensure axes are scaled equally

    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.6)

    handles, labels = ax.get_legend_handles_labels()

    model_legend = ax.legend(handles, labels, title="Models",
                             loc='center left', bbox_to_anchor=(1.02, 0.5),
                             fontsize=15, ncol=1) # Adjust ncol
    #ax.add_artist(model_legend) # Required when adding multiple legends to the same axes

    ax.tick_params(axis='x', labelsize=20, direction='out')
    ax.tick_params(axis='y', labelsize=20, direction='out')

    fig.subplots_adjust(right=0.70 if n_models_to_plot <=10 else 0.60) # Reduce right boundary to make space for legends

    plt.savefig("visible_hidden_gap_debug_model_legend.pdf", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()