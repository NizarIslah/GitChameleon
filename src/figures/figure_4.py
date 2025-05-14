import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import itertools
from scipy.stats import spearmanr, pearsonr

# --- CONFIGURATION ----------------------------------------------------------
benchmarks_x = ["SWE-Bench", "LiveCodeBench"]
y_bench      = "GitChameleon"

def generate_synthetic_data():
    return {
        "O1": {
            "GitChameleon": 51.2,
            "SWE-Bench":    41.0,
            "BigCodeBench": 35.5,
        },
        "o3-mini": {
            "GitChameleon": 44.51,
            "LiveCodeBench":77.7,
            "SWE-Bench":     49.3,
            "BigCodeBench":  35.1,
        },
        "GPT-4o": {
            "GitChameleon": 49.1,
            "LiveCodeBench":38.3,
            "BigCodeBench":  30.8,
            "SWE-Bench":     33.2,
        },
        "GPT-4o-Mini": {
            "GitChameleon": 37.2,
            "LiveCodeBench":35.5,
            "BigCodeBench":  25.3,
            "SWE-Bench":      8.7,
        },
        "GPT-4.1": {
            "GitChameleon": 48.5,
            "BigCodeBench": 32.8,
            "SWE-Bench":    54.6,
        },
        "GPT-4.1 Mini": {
            "GitChameleon": 44.2,
            "BigCodeBench": 31.8,
            "SWE-Bench":    23.6,
        },
        "GPT-4.5": {
            "GitChameleon": 40.1,
            "SWE-Bench":    38.0,
        },
        "Gemini 1.5-Pro": {
            "GitChameleon": 45.1,
            "BigCodeBench": 25.4,
        },
        "Gemini 2.5-Pro": {
            "GitChameleon": 50.0,
            "LiveCodeBench":81.5,
            "BigCodeBench":  33.1,
            "SWE-Bench":     63.8,
        },
        "Gemini 2.5-Flash": {
            "GitChameleon": 44.21,
            "LiveCodeBench":75.1,
        },
        "Claude 3.7 Sonnet": {
            "GitChameleon": 48.8,
            "LiveCodeBench":63.5,
            "BigCodeBench":  35.8,
            "SWE-Bench":     70.3,
        },
        "Claude 3.5 Sonnet": {
            "GitChameleon": 55.3,
            "LiveCodeBench":48.7,
        },
        "LLama 3.1": {
            "GitChameleon": 30.2,
            "BigCodeBench": 25.4,
        },
        "LLama 3.3": {
            "GitChameleon": 36.3,
            "BigCodeBench": 28.4,
        },
    }

data        = generate_synthetic_data()
model_names = list(data.keys())

# assign marker+color cycles
markers = itertools.cycle(['o','s','^','P','D','*','X','+','v','<','>','p','h','H'])
colors  = itertools.cycle([
    '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
    '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
    '#aec7e8','#ffbb78','#98df8a','#ff9896',
])
model_styles = {m:(next(markers), next(colors)) for m in model_names}

# Plot 1x2
# Significantly increased figure height to definitively provide space for the legend
fig, axs = plt.subplots(1, len(benchmarks_x), figsize=(10, 10), sharey=True) # Increased height to 10

for ax, x_bench in zip(axs, benchmarks_x):
    # collect valid points
    pts = [(data[m][x_bench], data[m][y_bench])
           for m in model_names
           if x_bench in data[m] and y_bench in data[m]]
    xs, ys = zip(*pts)
    rho_s, p_s = spearmanr(xs, ys)
    rho_p, p_p = pearsonr(xs, ys)
    print(f"{x_bench} vs {y_bench}: "
          f"Spearman ρ={rho_s:.3f} (p={p_s:.3f}), "
          f"Pearson r={rho_p:.3f} (p={p_p:.3f})")

    # Scatter
    for m in model_names:
        if x_bench not in data[m] or y_bench not in data[m]:
            continue
        mk, cl = model_styles[m]
        ax.scatter(data[m][x_bench],
                   data[m][y_bench],
                   marker=mk,
                   color=cl,
                   s=200,
                   label=m)
    ax.set_box_aspect(1) # Make plots square

    ax.set_xlabel(f"{x_bench} (↑)",
                  fontsize=25)
    if ax is axs[0]:
        ax.set_ylabel(f"{y_bench} (↑)",
                      fontsize=25)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='x', labelsize=25, direction='out')
    ax.tick_params(axis='y', labelsize=25, direction='out')


# Shared legend underneath
handles, labels = [], []
seen = set()
for m in model_names:
    if m in seen:
        continue
    mk, cl = model_styles[m]
    handles.append(mlines.Line2D([], [], marker=mk, color=cl,
                                 linestyle='None', markersize=12))
    labels.append(m)
    seen.add(m)

# Use tight_layout first to fit elements, then adjust legend placement
# bbox_to_anchor now uses coordinates relative to the figure,
# and it's placed far enough down due to the increased figure height.
fig.tight_layout()
fig.legend(handles, labels,
           title="Models",
           loc="lower center",
           bbox_to_anchor=(0.5, -0.15), # Adjusted y-coordinate, slightly below the bottom edge of the figure's default layout
           ncol=3,
           frameon=True,
           edgecolor='black',
           columnspacing=0.2,
           prop={'size':25},
           title_fontsize=25)\
   .get_title().set_fontweight('bold')

# Removed fig.subplots_adjust and rely on tight_layout with a larger figure size
# Using bbox_inches="tight" with pad_inches helps capture the legend
fig.savefig("figure_bench_corr.pdf", dpi=300, bbox_inches="tight", pad_inches=0.5)
# plt.show()