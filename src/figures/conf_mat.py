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

def get_greedy_result_df(data_eval_path): 
    rag_eval_results = {}
    for f in data_eval_path.iterdir():
        if f.suffix != ".csv":
            continue
        rag_eval_results[f.stem] = pd.read_csv(f)
        rag_eval_results[f.stem]["filename"] = f.stem.replace("_eval_results", "")
    
    rag_eval_combined = pd.concat(rag_eval_results)
    rag_eval_combined = rag_eval_combined[["filename", "example_id", "passed"]]
    rag_eval_combined = rag_eval_combined[rag_eval_combined.filename.str.contains("agent_results") == False]    
    rag_eval_combined = rag_eval_combined[rag_eval_combined.filename.str.contains("_responses") == False]    
    rag_eval_combined = rag_eval_combined[rag_eval_combined.filename != "responses"]    
    rag_eval_combined = rag_eval_combined[rag_eval_combined.filename.str.endswith("t0_1") == False]    
    rag_eval_combined = rag_eval_combined[rag_eval_combined.filename.str.endswith("cot_1") == False]    
    rag_eval_combined = rag_eval_combined[rag_eval_combined.filename.str.contains("goose") == False]    
    rag_eval_combined = rag_eval_combined[rag_eval_combined.filename.str.contains("gemini-2.5-flash-preview-04-17_") == False]        
    rag_eval_combined = rag_eval_combined[rag_eval_combined.filename.str.contains("pro_1") == False]        
    rag_eval_combined = rag_eval_combined[rag_eval_combined.filename.str.contains("25_1") == False]        
    rag_eval_combined = rag_eval_combined[rag_eval_combined.filename.str.contains("flash_1") == False]        
    rag_eval_combined = rag_eval_combined[rag_eval_combined.filename.str.contains("o4_mini") == False]        
    rag_eval_combined = rag_eval_combined[rag_eval_combined.filename.str.contains("claude_code") == False]        
    rag_eval_combined["cot"] = rag_eval_combined.filename.str.contains("_cot")
    rag_eval_combined["greedy"] = rag_eval_combined.filename.str.contains("t0") | (rag_eval_combined.filename.str.contains("gemini") & ~rag_eval_combined.filename.str.contains("cot"))
    rag_eval_combined.filename = rag_eval_combined.filename.apply(
        lambda x: x.replace("responses_0.0_True_", "").replace("_655", "").replace("t0_", "").replace("_cot", "").replace("cot_", "").replace("_t0", ""))
    return rag_eval_combined

# Increase all font sizes by default
plt.rcParams.update({
    'font.size': 24,            # base font size
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
})

greedy_eval_path = Path("all_eval_data")


result_df = get_greedy_result_df(greedy_eval_path)
result_df = result_df[result_df.greedy == True]
result_df = result_df[result_df.filename.str.endswith("_1") == False]
result_df = result_df[result_df.filename.str.contains("o1") == False]
result_df = result_df[result_df.filename.str.contains("o3") == False]
result_df = result_df.sort_values(by=["filename", "example_id"])

models = result_df.filename.unique()
#model_names = ["o3-mini", "GPT-4.1-mini", "GPT-4o", "GPT-4.1-nano", "GPT-4o-mini", "o1", "GPT-4.5", "GPT-4.1"]

model_names = {
    'o3_mini': "o3-mini",
    'gemini-2.0-flash': "Gemini 2.0 Flash",
    'gemini-1.5-pro': "Gemini 1.5 Pro",
    'gpt_41_mini': "GPT 4.1-mini",
    'gpt_4o': "GPT-4o",
    'gpt_41_nano': "GPT 4.1-nano",    
    'gpt_4o_mini': "GPT-4o-mini",
    'gemini-2.0-flash': "Gemini 2.0 Flash",
    'o1': "o1",
    'gemini-2.5-pro-preview-03-25': "Gemini 2.5 Pro",
    'gpt_45': "GPT 4.5",
    'gpt_41': "GPT 4.1",   
    "claude_37_sonnet": "Claude 3.7 Sonnet" 
}

# You can print it to verify:
# for original, capitalized in properly_capitalized_models.items():
#     print(f"'{original}': \"{capitalized}\"")

conf_mat = np.zeros((len(models), len(models)))

for idx in range(len(models)):
    for idx2 in range(idx+1,len(models)):
        x1 = result_df[result_df.filename==models[idx]]
        x2 = result_df[result_df.filename==models[idx2]]        
        conf_mat[idx][idx2] = (x1.passed.values == x2.passed.values).mean()
pass


cm = conf_mat
labels = [model_names[name] for name in models]

# make a boolean mask for cells below the diagonal
mask = np.tril(np.ones_like(cm, dtype=bool), k=0)


# apply the mask
cm_masked = np.ma.masked_where(mask, cm)

# pick a colormap and set masked entries to white
cmap = plt.cm.plasma.copy()
cmap.set_bad(color='white')

fig, ax = plt.subplots(figsize=(12, 12))
im = ax.imshow(cm_masked, interpolation='nearest', cmap=cmap)

# colorbar (automatically skips masked cells)
cbar = plt.colorbar(im, ax=ax, shrink=0.85, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=16)

# annotate only the unmasked (upper‐triangle+diag) cells
thresh = cm.max() / 2
for i in range(len(models)):
    for j in range(len(models)):
        if i < j:
            color = 'white' if cm[i, j] > thresh else 'black'
            ax.text(j, i, f"{cm[i, j]:.2f}", 
                    ha='center', va='center', fontsize=16, color=color)

ax.set(xticks=np.arange(len(models)), yticks=np.arange(len(models)),
       xticklabels=labels, yticklabels=labels)

plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)
plt.tight_layout()
plt.savefig("model_conf_mat.pdf", dpi=300, bbox_inches="tight")
plt.show()
