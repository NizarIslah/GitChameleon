import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- CONFIGURATION ----------------------------------------------------------

DATA_DIR = "/Users/beike/Desktop/Workspace/GitChameleon/plot/all_eval_data"
MASTER_FILE = "/Users/beike/Desktop/Workspace/GitChameleon/plot/final_fix_dataset.jsonl"
YEARS = [2021, 2022, 2023]

eval_files = [
    "o1_t0_eval_results.jsonl",
    "gpt_45_t0_eval_results.jsonl",
    "t0_claude_37_sonnet_eval_results_1.jsonl",
    "responses_0.0_True_gemini-1.5-pro_655_eval_results_1.jsonl",
    "responses_0.0_True_gemini-2.5-pro-preview-03-25_655_eval_results_1.jsonl"
]

models = [
    ("GPT-4.5", "#e41a1c"),
    ("O-1", "#377eb8"),
    ("Claude 3.7 Sonnet", "#4daf4a"),
    ("Gemini 1.5 Pro", "#984ea3"),
    ("Gemini 2.5 Pro", "#ff7f00"),
]

# models = [
#     ("O3‑Mini",     "#e41a1c"),
#     ("O-1",          "#377eb8"),
#     ("O4‑mini t0",     "#4daf4a"),
#     ("GPT‑4o‑Mini t0", "#984ea3"),
#     ("GPT‑4o‑t0",      "#ff7f00"),
#     ("GPT‑4.5‑t0",     "#ffff33"),
#     ("GPT‑4.1‑mini t0","#a65628"),
#     ("GPT‑4.1‑nano t0","#f781bf"),
#     ("GPT‑4.1‑t0",     "#999999"),
# ]

assert len(models) == len(eval_files), "models[] must match eval_files[]"

# ----------------------------------------------------------------------------

def load_release_years(path):
    m = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            ex_id = str(obj["example_id"])
            year = int(obj["release_date"].split("-")[0])
            m[ex_id] = year
    return m

def compute_rates_and_err(eval_path, year_map):
    stats = {yr: {"hid": [0,0], "vis": [0,0]} for yr in YEARS}
    with open(eval_path) as f:
        for line in f:
            rec = json.loads(line)
            yr = year_map.get(str(rec["example_id"]))
            if yr not in stats:
                continue
            hid = (rec.get("passed", "False") == "True")
            vis = (rec.get("passed_manual", "False") == "True")
            stats[yr]["hid"][0]  += hid
            stats[yr]["hid"][1]  += 1
            stats[yr]["vis"][0]  += vis
            stats[yr]["vis"][1]  += 1

    hid_rates, hid_errs, vis_rates, vis_errs = [], [], [], []
    for yr in YEARS:
        h_sum, h_tot = stats[yr]["hid"]
        v_sum, v_tot = stats[yr]["vis"]
        ph = h_sum / h_tot if h_tot else np.nan
        pv = v_sum / v_tot if v_tot else np.nan
        eh = np.sqrt(ph * (1-ph) / h_tot) if h_tot else 0
        ev = np.sqrt(pv * (1-pv) / v_tot) if v_tot else 0
        hid_rates.append(ph); hid_errs.append(eh)
        vis_rates.append(pv); vis_errs.append(ev)
    return hid_rates, hid_errs, vis_rates, vis_errs

def main():
    year_map = load_release_years(MASTER_FILE)

    n_models = len(models)
    hid_mat   = np.zeros((n_models, len(YEARS)))
    hid_errs  = np.zeros_like(hid_mat)
    vis_mat   = np.zeros_like(hid_mat)
    vis_errs  = np.zeros_like(hid_mat)

    for i, fname in enumerate(eval_files):
        path = os.path.join(DATA_DIR, fname)
        h_rates, h_err, v_rates, v_err = compute_rates_and_err(path, year_map)
        hid_mat[i, :]  = h_rates
        hid_errs[i, :] = h_err
        vis_mat[i, :]  = v_rates
        vis_errs[i, :] = v_err

    # --- PLOTTING -------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(24, 8), sharey=True)
    fig.subplots_adjust(left=0.3)

    bar_h = 0.35
    y_pos = np.arange(n_models)

    for col, ax in enumerate(axs):
        for i, (label, color) in enumerate(models):
            # hidden: solid fill
            ax.barh(
                y_pos[i] - bar_h/2,
                hid_mat[i, col],
                height=bar_h,
                color=color,
                alpha=0.9,
                xerr=hid_errs[i, col],
                error_kw=dict(ecolor='black', lw=1, capsize=3)
            )
            # visible: white fill + colored hatch outline
            ax.barh(
                y_pos[i] + bar_h/2,
                vis_mat[i, col],
                height=bar_h,
                facecolor='white',
                edgecolor=color,
                hatch='///',
                linewidth=1.5,
                xerr=vis_errs[i, col],
                error_kw=dict(ecolor='black', lw=1, capsize=3)
            )

        ax.set_title(f"{YEARS[col]}", fontsize=30, pad=12, fontweight='bold')
        ax.set_xlim(0, 0.8)
        ax.set_xlabel("Success Rate", fontsize=16, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [m[0] for m in models],
            fontsize=20,         # larger font size
            fontweight='bold'    # make it bold
        )
        
    # Legend for Hidden vs Visible
    hidden_patch = mpatches.Patch(facecolor='gray', edgecolor='none', label='Hidden')
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
    plt.savefig("model_date.pdf", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
