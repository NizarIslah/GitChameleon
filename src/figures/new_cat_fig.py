import matplotlib.pyplot as plt
import numpy as np

# --- Data Extraction / Estimation from the First Plot (for '2021' panel) ---
# Replace these estimated values with your actual data for all models and categories.
models = ['Gemini 2.5 Pro', 'Gemini 1.5 Pro', 'Claude 3.7 Sonnet', '0-1', 'GPT-4.5']

# Estimated Success Rates (mean values for 'Hidden' and 'Visible' bars)
# Format: [Hidden_mean, Visible_mean]
success_rates = {
    'Gemini 2.5 Pro': [0.65, 0.75],
    'Gemini 1.5 Pro': [0.55, 0.65],
    'Claude 3.7 Sonnet': [0.45, 0.58],
    '0-1': [0.40, 0.50],
    'GPT-4.5': [0.35, 0.45]
}

# Estimated Error Margins (assuming symmetrical error bars for simplicity)
# Format: [Hidden_error, Visible_error]
errors = {
    'Gemini 2.5 Pro': [0.02, 0.02],
    'Gemini 1.5 Pro': [0.02, 0.02],
    'Claude 3.7 Sonnet': [0.02, 0.02],
    '0-1': [0.02, 0.02],
    'GPT-4.5': [0.02, 0.02]
}

# Prepare data for plotting
hidden_means = [success_rates[model][0] for model in models]
visible_means = [success_rates[model][1] for model in models]
hidden_errors = [errors[model][0] for model in models]
visible_errors = [errors[model][1] for model in models]

# Parameters for plotting
n_models = len(models)
ind = np.arange(n_models)  # The y locations for the groups of horizontal bars
width = 0.35  # The height of the individual bars

# --- Styling from the Second Plot (`image_aa56ca.png`) ---
# Colors chosen to visually match the 'Greedy Decoding' (light purple)
# and '+ Self Debug' (darker purple) bars in the second image.
color_hidden = '#EBE0F4'  # Very light purple
color_visible = '#9370DB' # Medium purple (e.g., MediumPurple)

# Set global font properties for a clean, sans-serif look and ACL readability
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12  # Base font size for overall plot elements

# Create the figure and a single subplot (axis)
fig, ax = plt.subplots(figsize=(12, 7)) # Adjust figure size for better readability with larger fonts

# Plotting the horizontal grouped bars
# `barh` is used for horizontal bars. `ind - width/2` and `ind + width/2`
# position the bars for each model group.
rects1 = ax.barh(ind - width/2, hidden_means, width, xerr=hidden_errors,
                 color=color_hidden, label='Hidden', capsize=5, edgecolor='none') # `elinewidth` makes error bar lines thinner

rects2 = ax.barh(ind + width/2, visible_means, width, xerr=visible_errors,
                 color=color_visible, label='Visible', capsize=5, edgecolor='none')

# --- Apply Styling ---

# Grid lines: Add faint, dashed horizontal grid lines (on the x-axis for horizontal bars)
ax.xaxis.grid(True, linestyle='--', alpha=0.7, color='lightgray', linewidth=0.8)
ax.yaxis.grid(False) # Ensure no vertical grid lines

# Remove plot borders (spines)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False) # The x-axis line itself is removed for a cleaner look
ax.spines['left'].set_visible(False)  # The y-axis line is also removed

# Y-axis (Model Names):
ax.set_yticks(ind) # Set tick locations to align with bars
ax.set_yticklabels(models, fontsize=14, ha='right') # Set model names as labels, right-aligned with larger font
ax.tick_params(axis='y', length=0, pad=10) # Remove tick marks and add padding to labels

# X-axis (Success Rate):
ax.set_xlabel('Success Rate', fontsize=16, labelpad=15) # Set x-axis label with padding and larger font
ax.set_xlim(0, 1.0) # Set x-axis limits from 0 to 1.0 for success rate
ax.tick_params(axis='x', labelsize=12, pad=5) # Set tick label size and padding with larger font

# Title for the subplot
ax.set_title('Success Rate by Model (2021)', fontsize=18, pad=20) # Larger font for the title

# Legend: Positioned at the upper right, without a frame, and appropriate handle size
ax.legend(fontsize=14, frameon=False, loc='upper right', handlelength=1.5) # Larger font for legend

# Set background color to white for both the figure and the axes
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Adjust layout to prevent labels from overlapping and provide some margin around the plot
plt.tight_layout(rect=[0, 0, 1, 0.95]) # [left, bottom, right, top] normalized coordinates

# Display the plot
plt.show()