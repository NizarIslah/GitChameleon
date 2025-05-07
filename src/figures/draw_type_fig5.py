import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- CONFIGURATION ----------------------------------------------------------

DATA_DIR    = "/Users/beike/Desktop/Workspace/GitChameleon/plot/all_eval_data"
MASTER_FILE = "/Users/beike/Desktop/Workspace/GitChameleon/plot/final_fix_dataset.jsonl"

eval_files = [
    "o1_t0_eval_results.jsonl",
    "gpt_45_t0_eval_results.jsonl",
    "t0_claude_37_sonnet_eval_results_1.jsonl",
    "responses_0.0_True_gemini-1.5-pro_655_eval_results_1.jsonl",
    "responses_0.0_True_gemini-2.5-pro-preview-03-25_655_eval_results_1.jsonl"
]

models = [
    ("GPT-4.5",           "#e41a1c"),
    ("O-1",               "#377eb8"),
    ("Claude 3.7 Sonnet", "#4daf4a"),
    ("Gemini 1.5 Pro",    "#984ea3"),
    ("Gemini 2.5 Pro",    "#ff7f00"),
]

assert len(models) == len(eval_files), "models[] must match eval_files[]"

# **Only** these four categories:
categories = [
    'Argument change',
    'Function Name',
    'Semantics',
    'New feature',
]

def categorize(change: str) -> str:
    s = change.strip().lower()
    if re.search(r'\b(argument|attribute|param)\b', s):
        return 'Argument change'
    elif re.search(r'\b(new|feature|introduc|dependency|additional)\b', s):
        return 'New feature'
    elif re.search(r'\b(name change|rename|function|func|method|class)\b', s):
        return 'Function Name'
    elif re.search(r'\b(semantic|behaviour|behavior|runtime|breaking|deprecate|deprecation|output|return)\b', s):
        return 'Semantics'
    else:
        return 'Other/Unmatched'

def load_change_map(path):
    m = {}
    with open(path) as f:
        for line in f:
            obj   = json.loads(line)
            ex_id = str(obj["example_id"])
            raw   = obj.get("type_of_change", "")
            m[ex_id] = categorize(raw)
    return m

def compute_rates_and_err(eval_path, change_map):
    # initialize only the four categories
    stats = {cat: {"hid":[0,0], "vis":[0,0]} for cat in categories}

    with open(eval_path) as f:
        for line in f:
            rec = json.loads(line)
            cid = change_map.get(str(rec["example_id"]), None)
            # skip anything not in our four categories
            if cid not in stats:
                continue

            hid = (rec.get("passed","False") == "True")
            vis = (rec.get("passed_manual","False") == "True")
            stats[cid]["hid"][0] += hid
            stats[cid]["hid"][1] += 1
            stats[cid]["vis"][0] += vis
            stats[cid]["vis"][1] += 1

    hid_rates, hid_errs, vis_rates, vis_errs = [], [], [], []
    for cat in categories:
        h_sum, h_tot = stats[cat]["hid"]
        v_sum, v_tot = stats[cat]["vis"]
        ph = h_sum / h_tot if h_tot else np.nan
        pv = v_sum / v_tot if v_tot else np.nan
        eh = np.sqrt(ph * (1-ph) / h_tot) if h_tot else 0
        ev = np.sqrt(pv * (1-pv) / v_tot) if v_tot else 0
        hid_rates.append(ph); hid_errs.append(eh)
        vis_rates.append(pv); vis_errs.append(ev)
    return hid_rates, hid_errs, vis_rates, vis_errs

def main():
    change_map = load_change_map(MASTER_FILE)
    n_models   = len(models)
    n_cats     = len(categories)

    hid_mat   = np.zeros((n_models, n_cats))
    hid_errs  = np.zeros_like(hid_mat)
    vis_mat   = np.zeros_like(hid_mat)
    vis_errs  = np.zeros_like(hid_mat)

    # fill in matrices
    for i,fname in enumerate(eval_files):
        path = os.path.join(DATA_DIR, fname)
        h_rates, h_err, v_rates, v_err = compute_rates_and_err(path, change_map)
        hid_mat[i,:]   = h_rates
        hid_errs[i,:]  = h_err
        vis_mat[i,:]   = v_rates
        vis_errs[i,:]  = v_err

    # --- PLOTTING -------------------------------------------------------------
    fig, axs = plt.subplots(1, n_cats, figsize=(6*n_cats, 8), sharey=True)
    fig.subplots_adjust(left=0.3)

    bar_h = 0.35
    y_pos = np.arange(n_models)

    for col, cat in enumerate(categories):
        ax = axs[col]
        for i,(label,color) in enumerate(models):
            # hidden: solid
            ax.barh(
                y_pos[i] - bar_h/2,
                hid_mat[i,col],
                height=bar_h,
                color=color,
                alpha=0.9,
                xerr=hid_errs[i,col],
                error_kw=dict(ecolor='black', lw=1, capsize=3)
            )
            # visible: hatch
            ax.barh(
                y_pos[i] + bar_h/2,
                vis_mat[i,col],
                height=bar_h,
                facecolor='white',
                edgecolor=color,
                hatch='///',
                linewidth=1.5,
                xerr=vis_errs[i,col],
                error_kw=dict(ecolor='black', lw=1, capsize=3)
            )
        ax.set_title(cat, fontsize=26, pad=10, fontweight='bold')
        ax.set_xlim(0, 0.8)
        ax.set_xlabel("Success Rate", fontsize=12, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [m[0] for m in models],
            fontsize=20,         # larger font size
            fontweight='bold'    # make it bold
        )

    # legend
    hidden_patch  = mpatches.Patch(facecolor='gray', label='Hidden')
    visible_patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='Visible')
    fig.legend(
        handles=[hidden_patch, visible_patch],
        loc="lower center",
        ncol=2,
        frameon=False,
        prop={'size': 20, 'weight': 'bold'},
        handlelength=4,
        handleheight=1.5,
        bbox_to_anchor=(0.5, -0.05)
    )

    plt.tight_layout(rect=[0,0.03,1,1])
    plt.savefig("model_change.pdf", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
