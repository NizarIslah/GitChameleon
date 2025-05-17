#!/usr/bin/env python3
import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap

# --- CONFIGURATION ----------------------------------------------------------
DATA_DIR    = "all_eval_data"
DEBUG_DATA_DIR = "self_debug_data"
MASTER_FILE = "dataset/final_fix_dataset.jsonl"

eval_files = [
    "t0_claude_37_sonnet_eval_results_1.jsonl",
    "responses_0.0_True_gemini-1.5-pro_655_eval_results_1.jsonl",
    "responses_0.0_True_gemini-2.5-pro-preview-03-25_655_eval_results_1.jsonl",
    "gpt_41_t0_eval_results_1.jsonl",
    "o1_t0_eval_results.jsonl",
    "agent_results_ddg_sb_claude_eval_results.jsonl",
    "goose_eval_results_1.jsonl",
]
self_debug_data_files = [
    "claude_37_sonnet_debug_answers_merged_correct/t0_claude_37_sonnet_eval_results.jsonl",
    "gemini_self_debug_answers_merged_correct/responses_0.0_True_gemini-1.5-pro_655_eval_results.jsonl",
    "gemini_self_debug_answers_merged_correct/responses_0.0_True_gemini-2.5-pro-preview-03-25_655_eval_results.jsonl",
    "gpt_debug_outputs_answers_merged_correct/gpt_41_t0_eval_results.jsonl",
    "gpt_debug_outputs_answers_merged_correct/o1_t0_eval_results.jsonl",
    "",
    "",
]

models = [
    ("Claude 3.7 Sonnet", "#4daf4a"),
    ("Gemini 1.5 Pro", "#984ea3"),
    ("Gemini 2.5 Pro", "#ff7f00"),
    ("GPT-4.1", "#e41a1c"),
    ("O-1", "#377eb8"),
    ("DDG‑SB Claude 3.5 ", "#f781bf"),
    ("Goose", "#a65628"),
]
assert len(models) == len(eval_files), "models[] must match eval_files[]"

def wrapped_labels(width=12):
    return [textwrap.fill(label, width=width) for label, _ in models]

# --- 1) RELEASE YEAR PLOTS -----------------------------------------------
YEARS = [2021, 2022, 2023]

def load_release_years(path):
    m = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            ex_id = str(obj["example_id"])
            year = int(obj["release_date"].split("-")[0])
            m[ex_id] = year
    return m

def compute_rates_and_err_year_self_debug(eval_path, self_debug_path, year_map):
    stats = {yr: {"hid": [0,0], "self_debug": [0,0]} for yr in YEARS}
    with open(eval_path) as f:
        for line in f:
            rec = json.loads(line)
            yr = year_map.get(str(rec["example_id"]))
            if yr not in stats:
                continue
            if isinstance(rec["passed"], str):
                # for models that do not have passed_manual
                hid = (rec.get("passed","False") == "True")
            else:
                # bool
                hid = rec.get("passed", False)
            stats[yr]["hid"][0] += hid
            stats[yr]["hid"][1] += 1
    if not os.path.isfile(self_debug_path):
        print(f"Self debug file not found: {self_debug_path}")
        print(stats[yr]["hid"])
        # set self debug to 0 if no self debug data
        for yr in YEARS:
            stats[yr]["self_debug"][0] = 0
            stats[yr]["self_debug"][1] = 0
    else:
        with open(self_debug_path) as f:
            for line in f:
                rec = json.loads(line)
                yr = year_map.get(str(rec["example_id"]))
                if yr not in stats:
                    continue
                sd = (rec.get("passed", "False") == "True")
                stats[yr]["self_debug"][0] += sd
                stats[yr]["self_debug"][1] += 1

    # hid and self debug rates and errors
    hid_rates, hid_errs, sdbg_rates, sdbg_errs = [], [], [], []
    for yr in YEARS:
        h_sum, h_tot = stats[yr]["hid"]
        sd_sum, sd_tot = stats[yr]["self_debug"]
        ph = h_sum / h_tot if h_tot else np.nan
        psd = sd_sum / sd_tot if sd_tot else np.nan
        eh = np.sqrt(ph * (1-ph) / h_tot) if h_tot else 0
        esd = np.sqrt(psd * (1-psd) / sd_tot) if sd_tot else 0
        hid_rates.append(ph); hid_errs.append(eh)
        sdbg_rates.append(psd); sdbg_errs.append(esd)
    return hid_rates, hid_errs, sdbg_rates, sdbg_errs

def plot_by_years_self_debug():
    year_map = load_release_years(MASTER_FILE)
    n_models = len(models)
    hid_mat = np.zeros((n_models, len(YEARS)))
    hid_errs = np.zeros_like(hid_mat)
    sd_mat = np.zeros_like(hid_mat)
    sd_errs = np.zeros_like(hid_mat)

    for i, fname in enumerate(eval_files):
        path = os.path.join(DATA_DIR, fname)
        self_debug_path = os.path.join(DEBUG_DATA_DIR, self_debug_data_files[i])
        h_rates, h_err, sd_rates, sd_err = compute_rates_and_err_year_self_debug(path, self_debug_path, year_map)
        hid_mat[i, :] = h_rates
        hid_errs[i, :] = h_err
        sd_mat[i, :] = sd_rates
        sd_errs[i, :] = sd_err

    fig, axs = plt.subplots(1, len(YEARS), figsize=(24, 8), sharey=True)
    fig.subplots_adjust(left=0.3)
    bar_h = 0.35
    y_pos = np.arange(n_models, dtype=float)
    # custom y_pos
    y_pos[-1] -= 0.25
    print(y_pos)


    for col, ax in enumerate(axs):
        for i, (label, color) in enumerate(models):
            if "Goose" in label or "DDG" in label:
                ax.barh(y_pos[i], hid_mat[i,col], height=bar_h,
                        color=color, alpha=0.9, xerr=hid_errs[i,col],
                        error_kw=dict(ecolor='black', lw=1, capsize=3))
            else:
                ax.barh(y_pos[i] - bar_h/2, hid_mat[i,col], height=bar_h,
                        color=color, alpha=0.9, xerr=hid_errs[i,col],
                        error_kw=dict(ecolor='black', lw=1, capsize=3))
            # if model has ddg or goose do not plot self debug
            if "DDG" in label or "Goose" in label:
                # print(f"Skipping self debug for {label} in {YEARS[col]}")
                continue
            ax.barh(y_pos[i] + bar_h/2, sd_mat[i, col], height=bar_h,
                    facecolor='white', edgecolor=color, hatch='///', linewidth=1.5,
                    xerr=sd_errs[i, col], error_kw=dict(ecolor='black', lw=1, capsize=3))
        ax.set_title(f"{YEARS[col]}", fontsize=25, pad=12, fontweight='bold')
        ax.set_xlim(0.15 if col == 0 else 0.0 , 1.0)
        ax.set_xlabel("Success Rate", fontsize=20)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(wrapped_labels(), fontsize=24)
        ax.tick_params(axis='x', labelsize=20, direction='out')
        ax.locator_params(axis='x', nbins=7)
    ax.invert_yaxis()
    hidden_patch = mpatches.Patch(facecolor='gray', label='Hidden')
    visible_patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='Self-Debug')
    fig.legend(handles=[hidden_patch, visible_patch], loc="lower center", ncol=2,
               frameon=False, prop={'size': 30, 'weight': 'bold'},
               handlelength=4, handleheight=1.5, bbox_to_anchor=(0.5, -0.10))
    plt.tight_layout(rect=[0,0.03,1,1])
    plt.subplots_adjust(
                    wspace=0.2, # Increase horizontal space between subplots
                    ) # Increase vertical space
    plt.savefig("model_date_self_debug.pdf", dpi=300, bbox_inches="tight")
    plt.show()


def compute_rates_and_err_year(eval_path, year_map):
    stats = {yr: {"hid": [0,0], "vis": [0,0]} for yr in YEARS}
    with open(eval_path) as f:
        for line in f:
            rec = json.loads(line)
            yr = year_map.get(str(rec["example_id"]))
            if yr not in stats:
                continue
            hid = (rec.get("passed", "False") == "True")
            vis = (rec.get("passed_manual", "False") == "True")
            stats[yr]["hid"][0] += hid
            stats[yr]["hid"][1] += 1
            stats[yr]["vis"][0] += vis
            stats[yr]["vis"][1] += 1

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


def plot_by_years():
    year_map = load_release_years(MASTER_FILE)
    n_models = len(models)
    hid_mat = np.zeros((n_models, len(YEARS)))
    hid_errs = np.zeros_like(hid_mat)
    vis_mat = np.zeros_like(hid_mat)
    vis_errs = np.zeros_like(hid_mat)

    for i, fname in enumerate(eval_files):
        path = os.path.join(DATA_DIR, fname)
        h_rates, h_err, v_rates, v_err = compute_rates_and_err_year(path, year_map)
        hid_mat[i, :] = h_rates
        hid_errs[i, :] = h_err
        vis_mat[i, :] = v_rates
        vis_errs[i, :] = v_err

    fig, axs = plt.subplots(1, len(YEARS), figsize=(24, 8), sharey=True)
    fig.subplots_adjust(left=0.3)
    bar_h = 0.35
    y_pos = np.arange(n_models)

    for col, ax in enumerate(axs):
        for i, (label, color) in enumerate(models):
            ax.barh(y_pos[i] - bar_h/2, hid_mat[i, col], height=bar_h,
                    color=color, alpha=0.9, xerr=hid_errs[i, col],
                    error_kw=dict(ecolor='black', lw=1, capsize=3))
            ax.barh(y_pos[i] + bar_h/2, vis_mat[i, col], height=bar_h,
                    facecolor='white', edgecolor=color, hatch='///', linewidth=1.5,
                    xerr=vis_errs[i, col], error_kw=dict(ecolor='black', lw=1, capsize=3))
        ax.set_title(f"{YEARS[col]}", fontsize=25, pad=12, fontweight='bold')
        ax.set_xlim(0, 0.8)
        ax.set_xlabel("Success Rate", fontsize=20)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(wrapped_labels(), fontsize=24)
        ax.tick_params(axis='x', labelsize=20, direction='out')
        ax.locator_params(axis='x', nbins=7)
    ax.invert_yaxis()
    hidden_patch = mpatches.Patch(facecolor='gray', label='Hidden')
    visible_patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='Visible')
    fig.legend(handles=[hidden_patch, visible_patch], loc="lower center", ncol=2,
               frameon=False, prop={'size': 30, 'weight': 'bold'},
               handlelength=4, handleheight=1.5, bbox_to_anchor=(0.5, -0.10))

    plt.tight_layout(rect=[0,0.03,1,1])
    plt.subplots_adjust(                  
                    wspace=0.2, # Increase horizontal space between subplots
                    ) # Increase vertical space
    plt.savefig("model_date.pdf", dpi=300, bbox_inches="tight")
    plt.show()

# --- 2) CHANGE CATEGORY PLOTS --------------------------------------------
categories = [
    'Argument Change',
    'Function Name',
    'Semantics',
    'New Feature',
]

def categorize(change: str) -> str:
    s = change.strip().lower()
    # 1) Argument or Attribute change
    if re.search(r'\b(argument|attribute|param)\b', s):
        return 'Argument Change'

    # 2) New feature or additional dependency-based change
    elif re.search(r'\b(new|feature|introduc|dependency|additional)\b', s):
        return 'New Feature'

    # 3) Function Name change
    elif re.search(r'\b(name change|rename|function|func|method|class)\b', s):
        return 'Function Name'

    # 4) Semantics or Function Behavior change
    elif re.search(
        r'\b(semantic|behaviour|behavior|runtime|breaking|deprecate|deprecation|output|return)\b',
        s
    ):
        return 'Semantics'
    else:
        return 'Other/Unmatched'


def load_change_map(path):
    m = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            ex_id = str(obj["example_id"])
            raw   = obj.get("type_of_change", "")
            cat = categorize(raw)
            if cat in categories:
                m[ex_id] = cat
    return m

###
def compute_rates_and_err_change_self_debug(eval_path, self_debug_path, change_map):
    stats = {cat: {"hid":[0,0], "self_debug":[0,0]} for cat in categories}
    with open(eval_path) as f:
        for line in f:
            rec = json.loads(line)
            cid = change_map.get(str(rec["example_id"]))
            if cid is None:
                continue
            if isinstance(rec["passed"], str):
                # for models that do not have passed_manual
                hid = (rec.get("passed","False") == "True")
            else:
                # bool
                hid = rec.get("passed", False)
            stats[cid]["hid"][0] += hid
            stats[cid]["hid"][1] += 1
    if not os.path.isfile(self_debug_path):
        # set self debug to 0 if no self debug data
        for cat in categories:
            stats[cat]["self_debug"][0] = 0
            stats[cat]["self_debug"][1] = 0
    else:
        with open(self_debug_path) as f:
            for line in f:
                rec = json.loads(line)
                cid = change_map.get(str(rec["example_id"]))
                if cid is None:
                    continue
                sd = (rec.get("passed","False") == "True")
                stats[cid]["self_debug"][0] += sd
                stats[cid]["self_debug"][1] += 1

    # hid and self debug rates and errors
    hid_rates, hid_errs, sdbg_rates, sdbg_errs = [], [], [], []
    for cat in categories:
        h_sum, h_tot = stats[cat]["hid"]
        sd_sum, sd_tot = stats[cat]["self_debug"]
        ph = h_sum / h_tot if h_tot else np.nan
        psd = sd_sum / sd_tot if sd_tot else np.nan
        eh = np.sqrt(ph * (1-ph) / h_tot) if h_tot else 0
        esd = np.sqrt(psd * (1-psd) / sd_tot) if sd_tot else 0
        hid_rates.append(ph); hid_errs.append(eh)
        sdbg_rates.append(psd); sdbg_errs.append(esd)
    return hid_rates, hid_errs, sdbg_rates, sdbg_errs

###
def plot_by_change_categories_self_debug():
    change_map = load_change_map(MASTER_FILE)
    n_models = len(models)
    n_cats   = len(categories)
    hid_mat = np.zeros((n_models, n_cats))
    hid_errs = np.zeros_like(hid_mat)
    sd_mat = np.zeros_like(hid_mat)
    sd_errs = np.zeros_like(hid_mat)

    for i, fname in enumerate(eval_files):            
        path = os.path.join(DATA_DIR, fname)
        self_debug_path = os.path.join(DEBUG_DATA_DIR, self_debug_data_files[i])
        h_rates, h_err, sd_rates, sd_err = compute_rates_and_err_change_self_debug(path, self_debug_path, change_map)
        hid_mat[i,:]  = h_rates
        hid_errs[i,:] = h_err
        sd_mat[i,:]   = sd_rates
        sd_errs[i,:]  = sd_err

    fig, axs = plt.subplots(1, n_cats, figsize=(6*n_cats, 8), sharey=True)
    fig.subplots_adjust(left=0.3)
    bar_h = 0.35
    y_pos = np.arange(n_models, dtype=float)
    # custom y_pos
    y_pos[-1] -= 0.25
    print(y_pos)


    for col, cat in enumerate(categories):
        ax = axs[col]
        for i, (label, color) in enumerate(models):
            if "Goose" in label or "DDG" in label:
                ax.barh(y_pos[i], hid_mat[i,col], height=bar_h,
                        color=color, alpha=0.9, xerr=hid_errs[i,col],
                        error_kw=dict(ecolor='black', lw=1, capsize=3))
            else:
                ax.barh(y_pos[i] - bar_h/2, hid_mat[i,col], height=bar_h,
                        color=color, alpha=0.9, xerr=hid_errs[i,col],
                        error_kw=dict(ecolor='black', lw=1, capsize=3))
            # if model has ddg or goose do not plot self debug
            if "DDG" in label or "Goose" in label:
                # print(f"Skipping self debug for {label} in {cat}")
                continue
            ax.barh(y_pos[i] + bar_h/2, sd_mat[i,col], height=bar_h,
                    facecolor='white', edgecolor=color, hatch='///', linewidth=1.5,
                    xerr=sd_errs[i,col], error_kw=dict(ecolor='black', lw=1, capsize=3))
        ax.set_title(cat, fontsize=30, pad=10, fontweight='bold')
        ax.set_xlim(0.15, 0.9)
        ax.set_xlabel("Success Rate", fontsize=25)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(wrapped_labels(), fontsize=20)
        #ax.set_xticks([0, 0.
        ax.tick_params(axis='x', labelsize=20, direction='out')
        ax.locator_params(axis='x', nbins=7)

        ticks  = ax.get_yticks()

 
    ax.invert_yaxis()

    hidden_patch = mpatches.Patch(facecolor='gray', label='Greedy Decoding')
    visible_patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='Self-Debug')
    fig.legend(handles=[hidden_patch, visible_patch], loc="lower center", ncol=2,
               frameon=False, prop={'size': 30},
               handlelength=4, handleheight=1.5, bbox_to_anchor=(0.5, -0.10))

    #plt.tight_layout(rect=[0,0.03,1,1])    
    plt.tight_layout() 
    plt.subplots_adjust(                  
                    wspace=0.2, # Increase horizontal space between subplots
                    ) # Increase vertical space
    #ax.set_xticklabels(error_categories, rotation=45, ha='right', fontsize=10)
    plt.savefig("model_change_self_debug.pdf", dpi=300, bbox_inches="tight")
    plt.show()

def compute_rates_and_err_change(eval_path, change_map):
    stats = {cat: {"hid":[0,0], "vis":[0,0]} for cat in categories}
    with open(eval_path) as f:
        for line in f:
            rec = json.loads(line)
            cid = change_map.get(str(rec["example_id"]))
            if cid is None:
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


def plot_by_change_categories():
    change_map = load_change_map(MASTER_FILE)
    n_models = len(models)
    n_cats   = len(categories)
    hid_mat = np.zeros((n_models, n_cats))
    hid_errs = np.zeros_like(hid_mat)
    vis_mat = np.zeros_like(hid_mat)
    vis_errs = np.zeros_like(hid_mat)

    for i, fname in enumerate(eval_files):
        path = os.path.join(DATA_DIR, fname)
        h_rates, h_err, v_rates, v_err = compute_rates_and_err_change(path, change_map)
        hid_mat[i,:]  = h_rates
        hid_errs[i,:] = h_err
        vis_mat[i,:]  = v_rates
        vis_errs[i,:] = v_err

    fig, axs = plt.subplots(1, n_cats, figsize=(6*n_cats, 8), sharey=True)
    fig.subplots_adjust(left=0.3)
    bar_h = 0.35
    y_pos = np.arange(n_models)

    for col, cat in enumerate(categories):
        ax = axs[col]
        for i, (label, color) in enumerate(models):
            ax.barh(y_pos[i] - bar_h/2, hid_mat[i,col], height=bar_h,
                    color=color, alpha=0.9, xerr=hid_errs[i,col],
                    error_kw=dict(ecolor='black', lw=1, capsize=3))
            ax.barh(y_pos[i] + bar_h/2, vis_mat[i,col], height=bar_h,
                    facecolor='white', edgecolor=color, hatch='///', linewidth=1.5,
                    xerr=vis_errs[i,col], error_kw=dict(ecolor='black', lw=1, capsize=3))
        ax.set_title(cat, fontsize=30, pad=10, fontweight='bold')
        ax.set_xlim(0, 0.8)
        ax.set_xlabel("Success Rate", fontsize=25)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(wrapped_labels(), fontsize=20)
        #ax.set_xticks([0, 0.2, 0,4, 0.6, 0.8], fontsize=30)
        ax.tick_params(axis='x', labelsize=20, direction='out')
        ax.locator_params(axis='x', nbins=7)
    ax.invert_yaxis()

    hidden_patch = mpatches.Patch(facecolor='gray', label='Hidden')
    visible_patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='Visible')
    fig.legend(handles=[hidden_patch, visible_patch], loc="lower center", ncol=2,
               frameon=False, prop={'size': 30},
               handlelength=4, handleheight=1.5, bbox_to_anchor=(0.5, -0.10))

    #plt.tight_layout(rect=[0,0.03,1,1])    
    plt.tight_layout() 
    plt.subplots_adjust(                  
                    wspace=0.2, # Increase horizontal space between subplots
                    ) # Increase vertical space
    #ax.set_xticklabels(error_categories, rotation=45, ha='right', fontsize=10)
    plt.savefig("model_change.pdf", dpi=300, bbox_inches="tight")
    plt.show()

# --- 3) LIBRARY-SPECIFIC PLOTS --------------------------------------------
libraries = ["torch", "numpy", "sympy", "scipy", "django", "flask", "falcon"]
libraries_case = ["Torch", "NumPy", "SymPy", "SciPy", "Django", "Flask", "Falcon"]

def load_library_map(path):
    lib_map = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            lib = obj.get("library", "").lower()
            if lib in libraries:
                lib_map[str(obj["example_id"])] = lib
    return lib_map


def compute_rates_and_err_lib_self_debug(eval_path, self_debug_path, lib_map):
    stats = {lib: {"hid":[0,0], "self_debug":[0,0]} for lib in libraries}
    with open(eval_path) as f:
        for line in f:
            rec = json.loads(line)
            lib = lib_map.get(str(rec["example_id"]))
            if lib is None:
                continue
            if isinstance(rec["passed"], str):
                # for models that do not have passed_manual
                hid = (rec.get("passed","False") == "True")
            else:
                # bool
                hid = rec.get("passed", False)
            stats[lib]["hid"][0] += hid
            stats[lib]["hid"][1] += 1
    if not os.path.isfile(self_debug_path):
        # set self debug to 0 if no self debug data
        for lib in libraries:
            stats[lib]["self_debug"][0] = 0
            stats[lib]["self_debug"][1] = 0
    else:
        with open(self_debug_path) as f:
            for line in f:
                rec = json.loads(line)
                lib = lib_map.get(str(rec["example_id"]))
                if lib is None:
                    continue
                sd = (rec.get("passed","False") == "True")
                stats[lib]["self_debug"][0] += sd
                stats[lib]["self_debug"][1] += 1

    # hid and self debug rates and errors
    hid_rates, hid_errs, sdbg_rates, sdbg_errs = [], [], [], []
    for lib in libraries:
        h_sum, h_tot = stats[lib]["hid"]
        sd_sum, sd_tot = stats[lib]["self_debug"]
        ph = h_sum / h_tot if h_tot else np.nan
        psd = sd_sum / sd_tot if sd_tot else np.nan
        eh = np.sqrt(ph * (1-ph) / h_tot) if h_tot else 0
        esd = np.sqrt(psd * (1-psd) / sd_tot) if sd_tot else 0
        hid_rates.append(ph); hid_errs.append(eh)
        sdbg_rates.append(psd); sdbg_errs.append(esd)
    return hid_rates, hid_errs, sdbg_rates, sdbg_errs

def plot_by_libraries_self_debug():
    lib_map = load_library_map(MASTER_FILE)
    n_models = len(models)
    n_libs   = len(libraries)
    hid_mat = np.zeros((n_models, n_libs))
    hid_errs = np.zeros_like(hid_mat)
    sd_mat = np.zeros_like(hid_mat)
    sd_errs = np.zeros_like(hid_mat)

    for i, fname in enumerate(eval_files):
        path = os.path.join(DATA_DIR, fname)
        self_debug_path = os.path.join(DEBUG_DATA_DIR, self_debug_data_files[i])
        h_rates, h_err, sd_rates, sd_err = compute_rates_and_err_lib_self_debug(path, self_debug_path, lib_map)
        hid_mat[i,:]  = h_rates
        hid_errs[i,:] = h_err
        sd_mat[i,:]   = sd_rates
        sd_errs[i,:]  = sd_err

    n_rows = 2
    n_cols = 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 12), sharey=True)
    axs = axs.flatten()
    # fig.subplots_adjust(left=0.3)
    bar_h = 0.35
    y_pos = np.arange(n_models)

    for col_plot_idx, lib in enumerate(libraries):
        if col_plot_idx >= n_rows * n_cols:
            # Stop if we've filled all available subplot slots
            break

        ax = axs[col_plot_idx]
        for i, (label, color) in enumerate(models):
            if "Goose" in label or "DDG" in label:
                ax.barh(y_pos[i], hid_mat[i, col_plot_idx], height=bar_h,
                        color=color, alpha=0.9, xerr=hid_errs[i, col_plot_idx],
                        error_kw=dict(ecolor='black', lw=1, capsize=3))
            else:
                ax.barh(y_pos[i] - bar_h/2, hid_mat[i, col_plot_idx], height=bar_h,
                        color=color, alpha=0.9, xerr=hid_errs[i, col_plot_idx],
                        error_kw=dict(ecolor='black', lw=1, capsize=3))
            # if model has ddg or goose do not plot self debug
            if "DDG" in label or "Goose" in label:
                # print(f"Skipping self debug for {label} in {lib}")
                continue
            ax.barh(y_pos[i] + bar_h/2, sd_mat[i, col_plot_idx], height=bar_h,
                    facecolor='white', edgecolor=color, hatch='///', linewidth=1.5,
                    xerr=sd_errs[i, col_plot_idx], error_kw=dict(ecolor='black', lw=1, capsize=3))
        ax.set_title(libraries_case[col_plot_idx], fontsize=24, fontweight='bold', pad=10)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Success Rate", fontsize=20)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(wrapped_labels(30), fontsize=20)
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

def compute_rates_and_err_lib(eval_path, lib_map):
    stats = {lib: {"hid":[0,0], "vis":[0,0]} for lib in libraries}
    with open(eval_path) as f:
        for line in f:
            rec = json.loads(line)
            lib = lib_map.get(str(rec["example_id"]))
            if lib is None:
                continue
            hid = (rec.get("passed","False") == "True")
            vis = (rec.get("passed_manual","False") == "True")
            stats[lib]["hid"][0] += hid
            stats[lib]["hid"][1] += 1
            stats[lib]["vis"][0] += vis
            stats[lib]["vis"][1] += 1

    hid_rates, hid_errs, vis_rates, vis_errs = [], [], [], []
    for lib in libraries:
        h_sum, h_tot = stats[lib]["hid"]
        v_sum, v_tot = stats[lib]["vis"]
        ph = h_sum / h_tot if h_tot else np.nan
        pv = v_sum / v_tot if v_tot else np.nan
        eh = np.sqrt(ph * (1-ph) / h_tot) if h_tot else 0
        ev = np.sqrt(pv * (1-pv) / v_tot) if v_tot else 0
        hid_rates.append(ph); hid_errs.append(eh)
        vis_rates.append(pv); vis_errs.append(ev)
    return hid_rates, hid_errs, vis_rates, vis_errs


def plot_by_libraries():
    lib_map = load_library_map(MASTER_FILE)
    n_libs   = len(libraries) # Total number of libraries available
    n_models = len(models)

    hid_mat   = np.zeros((n_models, n_libs))
    hid_errs  = np.zeros_like(hid_mat)
    vis_mat   = np.zeros_like(hid_mat)
    vis_errs  = np.zeros_like(hid_mat)

    for i, fname in enumerate(eval_files):
        path = os.path.join(DATA_DIR, fname)
        h_rates, h_err, v_rates, v_err = compute_rates_and_err_lib(path, lib_map)
        hid_mat[i,:]   = h_rates
        hid_errs[i,:]  = h_err
        vis_mat[i,:]   = v_rates
        vis_errs[i,:]  = v_err

    n_rows = 2
    n_cols = 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 12), sharey=True)
    
    axs = axs.flatten()
    #fig.subplots_adjust(left=0.3) 
    bar_h = 0.35
    y_pos = np.arange(n_models)

    for col_plot_idx, lib in enumerate(libraries):
        if col_plot_idx >= n_rows * n_cols:
            # Stop if we've filled all available subplot slots
            break

        ax = axs[col_plot_idx] # Use the flattened index for the current subplot

        for i, (label, color) in enumerate(models):
            # Use col_plot_idx to get the correct data column for the current library
            ax.barh(y_pos[i] - bar_h/2, hid_mat[i, col_plot_idx], height=bar_h,
                    color=color, alpha=0.9, xerr=hid_errs[i, col_plot_idx],
                    error_kw=dict(ecolor='black', lw=1, capsize=3))
            ax.barh(y_pos[i] + bar_h/2, vis_mat[i, col_plot_idx], height=bar_h,
                    facecolor='white', edgecolor=color, hatch='///', linewidth=1.5,
                    xerr=vis_errs[i, col_plot_idx], error_kw=dict(ecolor='black', lw=1, capsize=3))
        
        ax.set_title(libraries_case[col_plot_idx], fontsize=24, fontweight='bold', pad=10)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Success Rate", fontsize=20)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(wrapped_labels(30), fontsize=20)
        ax.tick_params(axis='x', labelsize=20, direction='out')
        #ax.tick_params(axis='y', labelsize=20, direction='out')
    
    num_plots_created = min(n_libs, n_rows * n_cols)
    for i in range(num_plots_created, n_rows * n_cols):
        fig.delaxes(axs[i])

    if num_plots_created > 0: # Ensure there was at least one library plotted
        last_plotted_ax = axs[num_plots_created - 1]
        last_plotted_ax.invert_yaxis() # This applies only to the last subplot

    hidden_patch = mpatches.Patch(facecolor='gray', label='Hidden')
    visible_patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='Visible')
    fig.legend(handles=[hidden_patch, visible_patch], loc="lower center", ncol=2,
               frameon=False, prop={'size': 20, 'weight': 'bold'},
               handlelength=4, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=[0,0.05,1,1]) # rect=[left, bottom, right, top] adjusts the layout box
    plt.subplots_adjust(                  
                    wspace=0.2, # Increase horizontal space between subplots
                    ) # Increase vertical space
    plt.savefig("model_library.pdf", dpi=300, bbox_inches="tight")
    plt.show()

# --- MAIN ------------------------------------------------------------------
def main():
    plot_by_years()
    #plot_by_change_categories()
    # plot_by_libraries()
    plot_by_change_categories_self_debug()
    plot_by_libraries_self_debug()
    plot_by_years_self_debug()

if __name__ == "__main__":
    main()
