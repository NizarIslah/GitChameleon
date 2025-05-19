import matplotlib.pyplot as plt
import numpy as np

# --- Data (from your previous example) ---
model_names = ['GPT-4.1', 'Gemini 2.5 Pro', 'Claude 3.7 Sonnet', 'Llama 4 Maverick']
categories = ['rag', 'greedy', 'cot', 'self debug', 'average']
data = {
    'GPT-4.1':             [58.5, 48.5, 47.9, 67.1, 53.0],
    'Gemini 2.5 Pro':      [56.7, 50.0, 49.4, 64.0, 52.1],
    'Claude 3.7 Sonnet':   [56.1, 48.8, 45.1, 64.6, 50.7],
    'Llama 4 Maverick':    [45.1, 40.8, 46.6, 58.2, 47.7]
}

cat_order = [1, 2, 0, 3, 4]

# --- Plotting Parameters for ACL Single Column ---
# ACL single column width is typically around 3.3 to 3.5 inches.
ACL_COL_WIDTH_INCHES = 3.4 
# Adjust height to fit content, especially rotated x-labels and legend
FIGURE_HEIGHT_INCHES = 4.8 # Might need tuning (e.g., 4.5 to 5.2)

# Font sizes suitable for a small figure in a paper
TITLE_SIZE = 10.5
AXIS_LABEL_SIZE = 9.5
TICK_LABEL_SIZE = 8.5
LEGEND_SIZE = 9.0

# --- Colors and Emphasis for "average" ---
category_colors = {
    'rag': '#1f77b4',        # Muted Blue
    'greedy': '#ff7f0e',     # Muted Orange
    'cot': '#2ca02c',        # Muted Green
    'self debug': '#9467bd', # Muted Purple
    'average': '#d62728'     # Prominent Red for "average"
}
category_hatches = {
    'rag': '', 'greedy': '', 'cot': '', 'self debug': '',
    'average': '///'  # Hatch pattern to emphasize "average"
}
category_edgecolor = 'black'
# Linewidths for emphasis, adjusted for smaller bars
average_bar_linewidth = 1.0 # Slightly thicker edge for average bars
default_bar_linewidth = 0.6 # Thinner default edge for clarity at small size

# --- Bar Chart Setup ---
num_models = len(model_names)
num_categories = len(categories)

x_model_indices = np.arange(num_models)  # The label locations for models (0, 1, 2, 3)
bar_width = 0.13  # Adjusted width for 5 bars in a group (5 * 0.13 = 0.65 total bar group width)

# Create figure and axes
fig, ax = plt.subplots(figsize=(ACL_COL_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))

# --- Plot Bars for Each Category ---
for idx in range(len(categories)):
    i = cat_order[idx]
    category = categories[i]
    print(category)
    offset = (idx - num_categories / 2 + 0.5) * bar_width
    model_values_for_category = [data[model][i] for model in model_names]
    
    current_linewidth = average_bar_linewidth if category == 'average' else default_bar_linewidth
    
    rects = ax.bar(x_model_indices + offset, model_values_for_category, bar_width,
                   label=category.replace("_", " ").title(),
                   color=category_colors[category],
                   hatch=category_hatches[category],
                   edgecolor=category_edgecolor,
                   linewidth=current_linewidth)

# --- Axis Labels, Title, Ticks ---
ax.set_ylabel('Success Rate (%)', fontsize=AXIS_LABEL_SIZE, weight='normal') # Use normal weight for labels if bold is too much
ax.set_xlabel('Model', fontsize=AXIS_LABEL_SIZE, weight='normal', labelpad=10) # Add padding for rotated x-labels
#ax.set_title('Model Performance Comparison', fontsize=TITLE_SIZE, weight='bold', pad=10) # Shorter title
ax.set_xticks(x_model_indices)
# Rotate x-axis labels to prevent overlap with longer model names
ax.set_xticklabels(model_names, fontsize=TICK_LABEL_SIZE, rotation=20, ha="right", rotation_mode="anchor")
ax.tick_params(axis='y', labelsize=TICK_LABEL_SIZE)

# Set y-axis limits
max_val_overall = 0
for model in model_names:
    if data[model]:
        current_max = max(data[model])
        if current_max > max_val_overall: max_val_overall = current_max
y_axis_upper_limit = np.ceil(max_val_overall / 10.0) * 10 + (5 if max_val_overall % 10 > 5 or max_val_overall % 10 == 0 else 10)
y_axis_upper_limit = min(y_axis_upper_limit, 100) # Cap at 100
y_axis_upper_limit -= 5
ax.set_ylim([30, y_axis_upper_limit])

# Y-axis ticks at reasonable intervals (e.g., every 20%)
y_tick_step = 10
if y_axis_upper_limit <= 50: y_tick_step = 10 # More granular for smaller ranges
ax.set_yticks(np.arange(30, y_axis_upper_limit + 1, y_tick_step))

# --- Legend ---
handles, labels = ax.get_legend_handles_labels()
labels = ["Greedy Decoding", "CoT", "RAG", "Self-Debug", "Average"]
legend_ncol = 3 # Arrange 5 items in 2 rows (3 on first, 2 on second)
ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.35), 
          ncol=legend_ncol, fontsize=LEGEND_SIZE, handletextpad=0.5,          frameon=True,         edgecolor='black',           columnspacing=0.2,)

# --- Grid and Layout ---
ax.yaxis.grid(True, linestyle=':', alpha=0.6, color='gray') # Lighter grid lines
ax.set_axisbelow(True)

# Adjust layout: Use subplots_adjust for precise control of margins
# These values are fractions of the figure width/height
# May need tuning based on the final appearance and specific font metrics
plt.subplots_adjust(left=0.20, right=0.98, top=0.90, bottom=0.32)
# If using plt.tight_layout(), it might override subplots_adjust.
# Often, saving with bbox_inches='tight' is the best final step.

# Example of saving for ACL (vector format like PDF is preferred)
# plt.savefig("grouped_bar_chart_acl_singlecol.pdf", dpi=300, bbox_inches='tight')
# plt.savefig("grouped_bar_chart_acl_singlecol.png", dpi=300, bbox_inches='tight')
# print("Plot generated. Uncomment savefig lines to save.")
# The plotting tool will render the image.

fig.savefig("figure_main.pdf", dpi=300, bbox_inches="tight", pad_inches=0.1)