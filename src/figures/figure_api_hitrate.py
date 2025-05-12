#!/usr/bin/env python3
import pandas as pd
from scipy.stats import spearmanr, pearsonr

# 1) Paste your data here, now including the hidden‐test columns:
rows = [
    # Model, greedy_hidden, greedy_api, cot_hidden, cot_api, self_hidden, self_api
    ("Llama 3.1 Instruct Turbo", 30.18, 38.2, 36.59, 34.3, 48.48, 43.1),
    ("Llama 3.3 Instruct Turbo", 36.28, 34.8, 37.50, 35.6, 44.82, 38.8),
    ("Llama 4 Maverick",        40.85, 44.9, 46.65, 40.1, 58.23, 46.5),
    ("Claude 3.7 Sonnet",       48.78, 45.1, 45.12, 42.2, 64.63, 51.2),
    ("Gemini 1.5 Pro",          45.12, 44.6, 43.29, 42.2, 62.50, 51.5),
    ("Gemini 2.0 Flash",        44.21, 42.5, 35.98, 40.3, 58.54, 46.3),
    ("Gemini 2.5 Pro",          50.00, 45.8, 49.39, 46.3, 64.02, 48.6),
    ("Gemini 2.5 Flash",        38.11, 44.6, 30.79, 48.8, 61.28, 47.3),
    ("GPT-4.1",                 48.48, 45.3, 47.87, 43.6, 67.07, 50.6),
    ("GPT-4.1-mini",            44.21, 43.0, 24.09, 39.1, 60.67, 46.0),
    ("GPT-4.1-nano",            33.84, 40.8, 11.89, 30.8, 40.85, 44.0),
    ("GPT-4o",                  49.09, 44.6, 50.30, 41.3, 60.76, 47.6),
    ("GPT-4o-mini",             37.20, 36.9, 35.98, 36.0, 46.95, 40.9),
    ("GPT-4.5",                 40.85, 49.2, 39.94, 46.5, 64.02, 51.2),
    ("o1",                      51.22, 40.9, 41.16, 40.2, 69.82, 47.1),
    ("o3-mini",                 44.51, 39.4, 50.91, 39.8, 65.85, 43.9),
    # add more rows if desired...
]

df = pd.DataFrame(rows, columns=[
    "Model",
    "greedy_hidden", "greedy_api",
    "cot_hidden",    "cot_api",
    "self_hidden",   "self_api",
])

# 2) Compute correlations between hidden success rate and API hit rate
for setting, hid_col, api_col in [
    ("Greedy",     "greedy_hidden", "greedy_api"),
    ("CoT",        "cot_hidden",    "cot_api"),
    ("Self-Debug", "self_hidden",   "self_api"),
]:
    x = df[hid_col]
    y = df[api_col]
    spear_r, spear_p = spearmanr(x, y)
    pear_r, pear_p = pearsonr(x, y)
    print(f"--- {setting} ---")
    # print(f"Spearman ⍴ = {spear_r:.3f}, p = {spear_p:.3f}")
    print(f"Pearson  r = {pear_r:.3f}, p = {pear_p:.3f}\n")
