import json
import re
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import pandas as pd
from pathlib import Path
import textwrap
import matplotlib.patches as mpatches

def wrapped_labels(model_names, width=12):
    return [textwrap.fill(label, width=width) for label in model_names]


def load_jsonl(path):
    """Load a single JSON file."""
    with open(path) as f:
        data = [json.loads(line.strip()) for line in f]
    return data

def get_rag_results(path):
    """Get RAG results from a directory of JSON files."""
    rag_results = {}
    for file in path.glob("**/*.jsonl"):
        with open(file) as f:
            file_name = file.name.replace(".jsonl", "")
            x = load_jsonl(file)
            for y in x:
                y["filename"] = file_name
            rag_results[file_name] = x
    return rag_results

def get_rag_eval_results(path):
    # The results are in csv files
    eval_results = {}
    for file in path.glob("**/*.csv"):
        df = pd.read_csv(file)
        df["filename"] = file.stem.replace("_eval_results", "")
        eval_results[file.stem] = df
    return eval_results

def calculate_successes(gt, rag_results):
    successes = []
    for i, result in enumerate(rag_results):
        if gt[i]["answer"] == result["answer"]:
            successes.append(1)
        else:
            successes.append(0)

# Increase all font sizes by default
plt.rcParams.update({
    'font.size': 24,            # base font size
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
})

# --- User‑adjustable path ---
rag_path = Path("all_eval_data/RAG")
rag_eval_path = Path("all_eval_data/rag_results")


rag_results = get_rag_results(rag_path)
rag_eval_results = get_rag_eval_results(rag_eval_path)
#gt = load_jsonl(Path("dataset/final_fix_dataset.jsonl"))
rag_eval_combined = pd.concat(rag_eval_results)
rag_combined = pd.concat([pd.DataFrame(x) for x in rag_results.values()])
rag_combined["content"] = rag_combined.prompt.apply(lambda x: x["content"])
rag_combined["library"] = rag_combined["content"].apply(lambda x: x[x.index("<library>")+9:x.index("</library>")].strip().split("==")[0])
rag_combined = rag_combined[~rag_combined.example_id.isna()]
rag_combined.example_id = rag_combined.example_id.astype(int)
merged = rag_eval_combined.merge(rag_combined, on=["filename", "example_id"])
merged = merged[merged.filename.str.endswith("_k3")]
merged.filename = merged.filename.apply(lambda x: x.replace("rag_", "").replace("_k3", ""))
passed_by_model_library = merged.groupby(["filename", "library"]).passed.mean()

passed_by_model_library = passed_by_model_library.to_frame().reset_index().pivot(index="filename", columns=["library"])
passed_by_model_library.columns = passed_by_model_library.columns.get_level_values(1)
pass 

n_rows = 2
n_cols = 3
fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 12), sharey=True)
axs = axs.flatten()
bar_h = 0.35

models  = [
    ("gpt_41", "#4daf4a"),
    ("deepseek_v3", "#984ea3"),
    ("gemini25_pro", "#ff7f00"),
    ("claude_37_sonnet", "#e41a1c"),
    ("qwen3", "#377eb8")]
#libraries = passed_by_model_library.columns
model_names = ["GPT-4.1", "Deepseek V3", "Gemini 2.5 Pro", "Claude 3.7 Sonnet", "Qwen 3"]

libraries = ["torch", "numpy", "sympy", "scipy", "django", "flask", "falcon"]
libraries_case = ["Torch", "NumPy", "SymPy", "SciPy", "Django", "Flask", "Falcon"]



n_models = len(models)
n_libs = len(libraries)
y_pos = np.arange(n_models)






hid_dict = passed_by_model_library.to_dict()
#hid_mat = [] # models x libraries

for col_plot_idx, lib in enumerate(libraries):
    if col_plot_idx >= n_rows * n_cols:
        # Stop if we've filled all available subplot slots
        break

    ax = axs[col_plot_idx]
    for i, (label, color) in enumerate(models):
  
        ax.barh(y_pos[i] - bar_h/2, hid_dict[lib][label], height=bar_h,
                color=color, alpha=0.9,
                error_kw=dict(ecolor='black', lw=1, capsize=3))
    # if model has ddg or goose do not plot self debug
        # ax.barh(y_pos[i] + bar_h/2, sd_mat[i, col_plot_idx], height=bar_h,
        #         facecolor='white', edgecolor=color, hatch='///', linewidth=1.5,
        #         xerr=sd_errs[i, col_plot_idx], error_kw=dict(ecolor='black', lw=1, capsize=3))
    ax.set_title(libraries_case[col_plot_idx], fontsize=24, fontweight='bold', pad=10)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Success Rate", fontsize=20)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(wrapped_labels(model_names, 30), fontsize=20)
    ax.tick_params(axis='x', labelsize=20, direction='out')
    # ax.tick_params(axis='y', labelsize=20, direction='out')

num_plots_created = min(n_libs, n_rows * n_cols)
for i in range(num_plots_created, n_rows * n_cols):
    fig.delaxes(axs[i])
if num_plots_created > 0: # Ensure there was at least one library plotted
    last_plotted_ax = axs[num_plots_created - 1]
    last_plotted_ax.invert_yaxis() # This applies only to the last subplot
hidden_patch = mpatches.Patch(facecolor='gray', label='Greedy Decoding')
visible_patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='Self-Debug')
fig.legend(handles=[hidden_patch, visible_patch], loc="lower center", ncol=2,
            frameon=False, prop={'size': 20, 'weight': 'bold'},
            handlelength=4, bbox_to_anchor=(0.5, -0.05))
plt.tight_layout(rect=[0,0.05,1,1]) # rect=[left, bottom, right, top] adjusts the layout box
plt.subplots_adjust(                  
                wspace=0.2, # Increase horizontal space between subplots
                ) # Increase vertical space
plt.savefig("model_library_self_debug.pdf", dpi=300, bbox_inches="tight")
plt.show()
