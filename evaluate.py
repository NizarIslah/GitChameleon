import wandb
import json
import argparse
import pandas as pd
import sys
import numpy as np
import py_compile
import re
import os
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from copy import deepcopy
from transformers import AutoTokenizer
from collections import defaultdict
from src.sanitize import sanitize
from src.eval_code import evaluate_model, load_outputs_from_json, prepare_eval_df
from configs import get_evaluate_args


if __name__ == "__main__":
    options = get_evaluate_args()
    # if instruct in model name
    if (
        "instruct" in options.model_name.lower()
        or "codestral" in options.model_name.lower()
        or "openai" in options.model_name.lower()
    ):
        options.instruct = True
    # start a new wandb run to track this script
    config = deepcopy(vars(options))
    print(config)

    instruct_str = "_new_instruct_no_ans" if options.instruct else ""
    run_name = (
        options.model_name.split("/")[-1]
        + "_"
        + options.dataset_path.split("/")[-1].split(".")[0]
        + "_top_p"
        + str(options.top_p)
        + instruct_str
    )
    if options.enable_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project=options.wandb_project,
            entity=options.wandb_entity,
            # track hyperparameters and run metadata
            config=config,
            name=run_name,
        )

    if options.dataset_path.endswith(".jsonl"):
        # if jsonl
        df = pd.read_json(options.dataset_path, lines=True)
    else:
        # if csv
        df = pd.read_csv(options.dataset_path, encoding="latin1")

    save_dir = (
        options.output_path + "/" + options.model_name.split("/")[0]
        if "/" in options.model_name
        else options.output_path
    )
    os.makedirs(save_dir, exist_ok=True)
    print("dir to save results:", save_dir)

    output_df = load_outputs_from_json(options)
    assert output_df is not None, "No outputs to evaluate. Exiting..."
    df = prepare_eval_df(options, df, output_df)
    print(df.head())
    print("---Evaluation---")
    cot_str="" if not options.cot else "_cot"
    eval_save_file = (
        options.model_name.split("/")[-1]
        + "_n"
        + str(options.n_generate)
        + "_k"
        + str(options.k)
        + "_T="
        + str(options.temperature)
        + "_seed"
        + str(options.seed)
        + cot_str
        + "_eval.csv"
    )

    if options.resume and os.path.exists(save_dir + "/" + eval_save_file):
        df_updated = pd.read_csv(save_dir + "/" + eval_save_file)
        print(
            "Loaded partial results from: ",
            eval_save_file,
            f"Rows done: {df_updated.shape[0]}/{len(df)}",
        )
    else:
        df_updated = None

    eval_df = evaluate_model(
        options,
        df,
        save_dir + "/" + eval_save_file,
        df_updated=df_updated,
        bs=options.batch_size,
    )
    eval_df.to_csv(save_dir + "/" + eval_save_file, index=False)
    print("Saved results to: ", eval_save_file)
    print(
        f"final_pass @ {options.k}: ",
        eval_df[f"pass_at_{options.k}"].mean(),
    )
    print(
        f"final_compile @ {options.k}: ",
        eval_df[
            f"compile_at_{options.k}"
        ].mean(),
    )
