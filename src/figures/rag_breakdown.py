import json
import re
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import pandas as pd
from pathlib import Path

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
passed_by_library = merged.groupby("library").passed.mean()
pass 

