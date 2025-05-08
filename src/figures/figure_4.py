import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import itertools
from scipy.stats import spearmanr

# --- CONFIGURATION ----------------------------------------------------------
benchmarks_x = ["HumanEval", "MPDD++", "BigCodeBench"]
benchmark_y = "GitChameleon"

AVAILABLE_MARKERS = ['o', 's', '^', 'P', 'D', '*', 'X', '+', 'v', '<', '>', 'p', 'h', 'H']
AVAILABLE_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

# --- SYNTHETIC DATA GENERATION ----------------------------------------------
def generate_synthetic_data():
    return {
        "O-1":            {benchmark_y: 51.22, "HumanEval": 96.3, "MPDD++": 80.2, "BigCodeBench": 35.5},
        "GPT-4o":         {benchmark_y: 49.1,  "HumanEval": 92.7, "MPDD++": 72.2, "BigCodeBench": 30.8},
        "GPT-4o-Mini":    {benchmark_y: 37.2,  "HumanEval": 88.4, "MPDD++": 72.2, "BigCodeBench": 37.2},
        "Gemini 1.5-Pro": {benchmark_y: 45.1,  "HumanEval": 89.0, "MPDD++": 74.6, "BigCodeBench": 32.4},
        "Llama 3.1":      {benchmark_y: 30.2,  "HumanEval": 80.5, "MPDD++": 86.0, "BigCodeBench": 27.7},
        "Llama 3.3":      {benchmark_y: 36.3,  "HumanEval": 88.4, "MPDD++": 87.6, "BigCodeBench": 28.4},
    }

def assign_model_styles(model_names):
    marker_cycle = itertools.cycle(AVAILABLE_MARKERS)
    color_cycle = itertools.cycle(AVAILABLE_COLORS)
    model_styles = {}
    for name in model_names:
        model_styles[name] = (next(marker_cycle), next(color_cycle))
    return model_styles

# --- MAIN ------------------------------------------------------------------
data = generate_synthetic_data()
model_names = list(data.keys())
model_styles = assign_model_styles(model_names)

# Calculate Spearman correlations
print("Spearman Correlation Coefficients (ρ) between each benchmark and GitChameleon:")
for benchmark in benchmarks_x:
    x_vals = [data[m][benchmark] for m in model_names]
    y_vals = [data[m][benchmark_y] for m in model_names]
    rho, pval = spearmanr(x_vals, y_vals)
    print(f"  - {benchmark} vs {benchmark_y}: ρ = {rho:.3f}, p-value = {pval:.3f}")

# Plot with increased symbol sizes and bold styling
n_benchmarks = len(benchmarks_x)
fig, axs = plt.subplots(1, n_benchmarks, figsize=(6 * n_benchmarks, 6), sharey=True)
plt.tight_layout(rect=[0.05, 0.15, 0.95, 0.95])

scatter_size = 200  # Marker size
legend_marker_size = 12  # Legend marker size

for i, benchmark in enumerate(benchmarks_x):
    ax = axs[i]
    for model in model_names:
        x_val = data[model][benchmark]
        y_val = data[model][benchmark_y]
        marker, color = model_styles[model]
        ax.scatter(x_val, y_val, marker=marker, color=color, s=scatter_size)
    ax.set_xlabel(f"{benchmark} Success Rate (↑)", fontsize=15, fontweight='bold')
    if i == 0:
        ax.set_ylabel(f"{benchmark_y} Success Rate (↑)", fontsize=15, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)

# Unified legend formatted with bold title and labels
legend_handles = [
    mlines.Line2D([], [], color=style[1], marker=style[0], linestyle='None', markersize=legend_marker_size)
    for style in model_styles.values()
]
legend = fig.legend(
    handles=legend_handles,
    labels=model_names,
    title="Models",
    loc="lower center",
    ncol=3,
    frameon=True,
    framealpha=0.9,
    edgecolor='black',
    prop={'size': 10, 'weight': 'bold'},
    title_fontsize=12,
    bbox_to_anchor=(0.5, 0)
)
legend.get_title().set_fontweight('bold')

plt.savefig("figure1.pdf", dpi=300, bbox_inches="tight", transparent=True)
plt.show()
