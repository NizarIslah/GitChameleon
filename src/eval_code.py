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
import pdb

def load_outputs_from_json(options):
    if options.cot:
        # greedy, 1 file, "solution"
        model_name = options.model_name.split('/')[-1]
        outputs = defaultdict(list)
        with open(options.cot_output_path, 'r') as f:
            for line_idx, line in enumerate(f):
                resp = json.loads(line)
                for k in range(options.n_generate):
                    outputs[f"output_{k}"].append(sanitize(resp["solution"]))

    else:
        assert os.path.exists(options.json_out_file), "Json out file does not exist."
        if '/' in options.model_name:
            model_name = options.model_name.split('/')[-1]
        else:
            model_name = options.model_name
        outputs = defaultdict(list)
        possible_keys = ["task_id", "seed"]
        def get_task(resp):
            for key in possible_keys:
                if key in list(resp.keys()):
                    return key, resp[key]
            return None
        with open(options.json_out_file, 'r') as f:
            k = 0
            cur_task_id = 0
            for line_idx, line in enumerate(f):
                resp = json.loads(line)
                # task_id in json format
                task_key, task_id = get_task(resp)
                if task_key == 'task_id':
                    if task_id != cur_task_id:
                        # pdb.set_trace()
                        options.n_generate = k
                        try:
                            assert options.k <= options.n_generate
                        except Exception as e:
                            print("value of --k should be <= to number of samples generated.")
                            exit(1)
                        k = 0
                        cur_task_id = task_id
                    try:
                        solution = sanitize(resp["solution"])
                    except Exception as e:
                        print("Error: ", e)
                        solution = resp["solution"]
                elif task_key == 'seed':   # gemini format: "output", "seed"
                    k = task_id
                    if k > 1000:
                        k = 0 # hardcoded
                    try:
                        solution = sanitize(resp["output"])
                    except Exception as e:
                        print("Error: ", e)
                        solution = resp["output"]
                else:
                    raise ValueError("task_id not found in json output.")
                outputs[f"output_{k}"].append(solution)
                if task_key == 'task_id':
                    k += 1

    if outputs is not None:
        output_df = pd.DataFrame(outputs)
        if options.cot:
            output_df = output_df.map(lambda x: extract_code_cot(x) if x is not None else x)
    else:
        output_df = None
    print(output_df.head()['output_0'])
    return output_df

def check_empty_outputs(options, df):
    # check that outputs are not empty
    for k in range(options.n_generate):
        if any(df[f"output_{k}"].isna()):
            print(
                f"Empty outputs for output_{k}"
            )
        # drop rows with empty outputs
        df = df[~df[f"output_{k}"].isna()]
    if df.shape[0] == 0:
        print("No outputs to evaluate. Exiting...")
        exit(1)
    else:
        print("Non empty rows: ", df.shape[0])
    df.reset_index(drop=True, inplace=True)
    return df

def prepare_eval_df(options, df, output_df):
    id_end = len(output_df) if options.id_end == -1 else options.id_end
    # df = df.iloc[options.id_start:id_end]
    # if options.test:
    #     print("Running in test mode")
    #     # take 10 random samples
    #     n = 10
    #     df = df.sample(n, random_state=options.seed)
    # df.reset_index(drop=True, inplace=True)
    output_df = output_df.iloc[options.id_start:id_end]
    output_df.reset_index(drop=True, inplace=True)
    if options.library != "":
        df = df[df["library"] == options.library]
        output_df = output_df[df["library"] == options.library]

    df = pd.merge(df, output_df, left_index=True, right_index=True)

    df = add_ranking_index(
        df, options.model_name.split("/")[-1], options.n_generate
    )
    # check empty outputs
    df = check_empty_outputs(options, df)
    try:
        # repo_dir = os.path.dirname(os.path.dirname(options.dataset_path))
        # df_env = pd.read_csv(f"{repo_dir}/updated_libraries.csv")
        df_env = pd.read_csv(options.dataset_env_path)
        # id_end = len(df_env) if options.id_end == -1 else options.id_end
        # df_env = df_env.iloc[options.id_start:id_end]
        # df_env.reset_index(drop=True, inplace=True)
        df["env_id"] = df_env["env_id"]
    except Exception as e:
        print("Error: ", e)
        exit(1)
    # print(df.head())
    # exit(0)
    return df

def get_ranks(model_name, row):
    """
    returns the indices of the best mean_logp, sum_logp and a random index out of the k generated outputs
    """
    return [
        row[f"best_mean_logp_index"],
        row[f"best_sum_logp_index"],
        row[f"random_index"],
    ]


def add_ranking_index(df, model_name, n, regen=False):
    # Generate column names
    regen_str = "regen_" if regen else ""
    outputs_cols = [f"{regen_str}output_{i}" for i in range(n)]
    mean_logp_cols = [f"{regen_str}mean_logp_{i}" for i in range(n)]
    sum_logp_cols = [f"{regen_str}sum_logp_{i}" for i in range(n)]

    # Filter the DataFrame to include only the necessary columns
    try:
        df_filtered = df[outputs_cols + mean_logp_cols + sum_logp_cols]
    except Exception as e:
        # print("Not all columns found in the dataframe, skipping...", e)
        df[f"best_mean_logp_index"] = np.random.randint(0, n, size=len(df))
        df[f"best_sum_logp_index"] = np.random.randint(0, n, size=len(df))
        df[f"random_index"] = np.random.randint(0, n, size=len(df))
        return df

    # Calculate the indices of the best mean_logp and sum_logp
    df[f"best_mean_logp_index"] = (
        df_filtered[mean_logp_cols]
        .idxmax(axis=1)
        .apply(lambda x: int(x.split("_")[-1]))
    )
    df[f"best_sum_logp_index"] = (
        df_filtered[sum_logp_cols].idxmax(axis=1).apply(lambda x: int(x.split("_")[-1]))
    )
    # make the values ints
    df[f"best_mean_logp_index"] = df[
        f"best_mean_logp_index"
    ].astype(int)
    df[f"best_sum_logp_index"] = df[
        f"best_sum_logp_index"
    ].astype(int)
    # Add a random output column
    df[f"random_index"] = np.random.randint(0, n, size=len(df))
    return df


def pass_at_k(model_test_results, k=10):
    """
    Function to check if model passed any

    Args:
    model_test_results: List of test results (k) from the model, 0 means success

    Returns:
    a list of size k which at each index contains 1 if that sample passed the test, 0 otherwise
    """
    if k < len(model_test_results):
        model_test_results = model_test_results[:k]
    return [int(result == 0) for result in model_test_results]


def corrected_pass_at_k(n, c, k=10):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


import subprocess
import os


def has_triple_quotes(string):
    return "'''" in string or '"""' in string


def get_python_executable(base_path, venv_name):
    """
    base_path: str, path to the base directory.
    venv_name: str, name of the virtual environment.
    return: str, path to the Python executable in the virtual environment.
    """
    venv_path = os.path.join(base_path, "gcham_venv" + venv_name)
    bin_path = os.path.join(venv_path, "bin")
    python_executable = os.path.join(bin_path, "python")
    return python_executable


def remove_patterns(starting_code, model_output):
    """
    Function to remove patterns from a text
    text: str, text to remove patterns from
    patterns: list of str, patterns to remove
    return: str, text with patterns removed
    """
    pattern = r"(?m)^.*assert.*$\n?"
    # Use re.sub() to remove the 'assert' line
    cleaned_code = re.sub(pattern, "", model_output)
    # gradio cleaning - remove lines with .launch and parentheses
    pattern = r"(?m)^\s*.*\.launch\([^\n]*\).*\s*$"
    cleaned_code = re.sub(pattern, "", cleaned_code)
    # remove lines with test_launch
    # pattern = r'(?m)^\s*.*\.test_launch\([^\n]*\).*\s*$'
    # cleaned_code = re.sub(pattern, '', cleaned_code)
    # matplotlib - remove .show and parentheses
    pattern = r"(?m)^\s*.*\.show\([^\n]*\).*\s*$"
    cleaned_code = re.sub(pattern, "", cleaned_code)
    # remove assistant may be on the first line
    pattern = r"(?m)^.*assistant.*$\n?"
    pattern2 = r"(?m)^.*Assistant.*$\n?"
    cleaned_code = re.sub(pattern, "", cleaned_code)
    cleaned_code = re.sub(pattern2, "", cleaned_code)
    # "2"
    pattern = r"(?m)^.*2.*$\n?"
    cleaned_code = re.sub(pattern, "", cleaned_code)
    # <|placeholder6|> in the code
    pattern = r"(?m)^.*<\|placeholder\d+\|>.*$\n?"
    cleaned_code = re.sub(pattern, "", cleaned_code)
    # [inst] and [starter_code] lower or capitalized
    pattern = r"(?m)^.*\[inst\].*$\n?"
    pattern2 = r"(?m)^.*\[starter_code\].*$\n?"
    cleaned_code = re.sub(pattern, "", cleaned_code)
    cleaned_code = re.sub(pattern2, "", cleaned_code)
    # [ PYTHON ] and [/ PYTHON]
    pattern = r"(?m)^.*\[ PYTHON \].*$\n?"
    pattern2 = r"(?m)^.*\[/ PYTHON\].*$\n?"
    cleaned_code = re.sub(pattern, "", cleaned_code)
    cleaned_code = re.sub(pattern2, "", cleaned_code)
    # remove answer may be on the first line
    pattern = r"(?m)^.*answer.*$\n?"
    cleaned_code = re.sub(pattern, "", cleaned_code)
    # remove main block
    cleaned_code = remove_main_block(cleaned_code)

    model_output = cleaned_code
    return model_output


def remove_main_block(script: str) -> str:
    # Find the start of the if __name__ == '__main__': block
    main_start = script.find("if __name__ ==")

    # If the block is found, return the script up to that point
    if main_start != -1:
        return script[:main_start]

    # If the block isn't found, return the script unchanged
    return script


def concat_testcase(
    starting_code,
    model_output,
    test,
    instruct,
    add_back_starter=True,
    verbose_mode=False,
):
    # TODO: check if no parsing is needed
    # try:
        # regex filter out starting_code from output exact string matching
        # if instruct:  for base as well
        # model_output = remove_patterns(starting_code, model_output)
    # except Exception as e:
    #     print("Error: ", e)
    #     print(type(model_output))
    if verbose_mode:
        print("This is model code to run before assert: ", model_output)
        print("------------------------------------")

    if add_back_starter:
        return starting_code + model_output + "\n" + test
    # if instruct, then we need to just concatenate the model output and test
    try:
        final_code = str(model_output) + "\n" + str(test)
    except Exception as e:
        print("Error: ", e)
        print("model_output: ", model_output)
        final_code = test
    return final_code


# Function to run a Python script and return the result
def run_script(python_executable, py_file="temp.py"):
    if py_file is None:
        return 0, 0, "", ""

    parsed_code = ""
    try:
        with open(py_file, "r") as file:
            parsed_code = file.read()
    except Exception as e:
        print(py_file, type(py_file))
        print("Error at py_file open:", e)

    error_log = ""
    try:
        # Try to compile the temporary file
        py_compile.compile(py_file, doraise=True)
        compile_code = 0  # Compilation successful
    except py_compile.PyCompileError as e:
        compile_code = 1  # Compilation failed due to a syntax error
        error_log = str(e)
    if compile_code == 0:
        # Run the Python script within the virtual environment
        command = [python_executable, py_file]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            exit_code = result.returncode
            error_log = result.stderr
            # print("error_log: ", error_log)
            # if "ModuleNotFoundError" in error_log:
            #     print("ModuleNotFoundError")
            # else:
            #     print("No ModuleNotFoundError")
        except subprocess.TimeoutExpired as e:
            print(e)
            exit_code = 1
            error_log = "TimeoutError"
    else:
        exit_code = 1  # since Compilation failed, the script will not run
    try:
        os.remove(py_file)
    except Exception as e:
        print(e)
    return 1 - exit_code, 1 - compile_code, parsed_code, error_log  # 1 = pass, 0 = fail


def extract_code_cot(text):

    if "[/THOUGHT]" in text:
        try:
            text = text.split("[/THOUGHT]")[1]
        except Exception as e:
            print("Error: ", e)
    if "[SOLUTION]" in text:
        if text.count("[SOLUTION]") > text.count("[/SOLUTION]"):
            # add closing tag if odd number of delimiters
            text += "[/SOLUTION]"
        try:
            text = re.search(r"\[SOLUTION\](.*?)\[/SOLUTION\]", text, re.DOTALL).group(
                1
            )
        except Exception as e:
            print("Error: ", e)
    if "[PYTHON]" in text:
        if text.count("[PYTHON]") > text.count("[/PYTHON]"):
            # add closing tag if odd number of delimiters
            text += "[/PYTHON]"
        try:
            code = re.search(r"\[PYTHON\](.*?)\[/PYTHON\]", text, re.DOTALL).group(1)
        except Exception as e:
            print("Error: ", e)
            print("text: ", text)
            code = text
        return code
    # odd number of backticks
    if text.count("```") % 2 == 1:
        text += "```"
    code_blocks = re.findall(r"```(?:\w+\n)?(.*?)```", text, re.DOTALL)
    code = "\n".join(code_blocks)
    if not code_blocks:
        # Fallback in case there are no code blocks found
        code = text.strip()
    return code


def extract_columns(df, row, columns):
    """
    Function to extract columns from a dataframe
    df: pd.DataFrame, dataframe to extract columns from
    row: int, row index
    columns: list of str, columns to extract
    return: tuple of str, extracted columns
    """
    return (df[column][row] for column in columns)


def extract_columns(row, columns):
    """
    Function to extract columns from a dataframe
    df: pd.DataFrame, dataframe to extract columns from
    row: int, row index
    columns: list of str, columns to extract
    return: tuple of str, extracted columns
    """
    return (row[column] for column in columns)


def write_py_file(code, py_file):
    if has_triple_quotes(code):
        with open(py_file, "w") as file:
            file.write(code)
    elif "\n" in code:
        # If the code has multiple lines, write it to a temporary file
        code = code.replace("\\n", "\n")
        code_lines = code.split("\n")
        # write line by line
        with open(py_file, "w") as file:
            file.writelines([f"\n{line}" for line in code_lines])
    else:
        with open(py_file, "w") as file:
            file.write(code)  # Run the script as is


def make_py_file(
    starting_code, model_out, test, instruct, py_file="temp.py", verbose_mode=False
):
    if model_out is None or pd.isna(model_out) or model_out == "":
        return None
    py_file_wo_starter = py_file.replace(".py", "_wo_starter.py")

    code_wo_starter = concat_testcase(
        starting_code,
        model_out,
        test,
        instruct,
        add_back_starter=False,
        verbose_mode=verbose_mode,
    )

    """
    code: str, Python code to run.
    return: python file.
    """
    write_py_file(code_wo_starter, py_file_wo_starter)
    return py_file_wo_starter


def eval_sample_k(
    base_path, model_name, row, n, k, idx, seed, temperature, options, regen=False
):
    """
    Function to evaluate the model outputs at k
    base_path: str, path to the base directory.
    model_name: name of the model
    row: row of the dataframe, example to evaluate
    n: int, number of generations from the model
    k: int, number of k to evaluate
    return: results_dict, containing eval results for each model and for each of the k then sample ranking heuristics (sum_logp, mean_logp, random)
    """
    starting_code, test, venv_name = row["starting_code"], row["test"], row["env_id"]
    py_exec = get_python_executable(base_path, venv_name)
    regen_str = "regen_" if regen else ""
    try:
        assert py_exec is not None
        assert os.path.exists(py_exec)
    except Exception as e:
        print(f"Error: venv not found, skipping sample {idx}...", e)
        return None, None, None, None, None

    # concat k's + sample ranking heuristics
    outputs_cols = [f"{regen_str}output_{i}" for i in range(n)] + [
        f"{regen_str}output_{rank}"
        for rank in get_ranks(model_name, row)
    ]  # [k:] is ranking heuristics
    model_outputs = extract_columns(row, outputs_cols)

    tmp_path = f"{options.scratch}/tmp_files/{model_name}/{seed}/{temperature}"
    # if not os.path.exists(tmp_path):
    os.makedirs(tmp_path, exist_ok=True)
    py_files = [
        make_py_file(
            starting_code,
            model_out,
            test,
            options.instruct,
            py_file=os.path.join(
                tmp_path, f"temp_{idx}_{k}.py"
            ),
            verbose_mode=options.verbose_mode,
        )
        for k, model_out in enumerate(model_outputs)
    ]

    passes, compiles, parsed_codes, error_logs = zip(
        *[run_script(py_exec, py_file) for py_file in py_files]
    )

    return passes, compiles, parsed_codes, error_logs, outputs_cols


def make_result_df(results, options, regen=False):
    regen_str = "regen_" if regen else ""
    model_name = options.model_name.split("/")[-1]
    n = options.n_generate
    k = options.k
    results_dict = {}
    passes, compiles, parsed_codes, error_logs, outputs_cols = results

    if passes is None:
        # make empty df with one row of nothing
        col_names = [
            f"{regen_str}{key}"
            for key in [
                "best_mean_logp_pass",
                "best_mean_logp_compile",
                "best_sum_logp_pass",
                "best_sum_logp_compile",
                "random_pass",
                "random_compile",
            ]
        ]
        col_names += [f"{regen_str}pass_at_{k_}" for k_ in range(1, k + 1)]
        col_names += [
            f"{regen_str}compile_at_{k_}" for k_ in range(1, k + 1)
        ]
        empty_df = pd.DataFrame({col: [None] for col in col_names})
        return empty_df

    # Define the dictionary for storing results
    results_dict[model_name] = {}

    pass_count = sum(passes[:n])
    compile_count = sum(compiles[:n])
    for i in range(n):
        if i < k:
            results_dict[model_name].update(
                {
                    f"{regen_str}pass_at_{i+1}": corrected_pass_at_k(
                        n, c=pass_count, k=i + 1
                    ),
                    f"{regen_str}compile_at_{i+1}": corrected_pass_at_k(
                        n, c=compile_count, k=i + 1
                    ),
                }
            )
        results_dict[model_name].update(
            {
                f"{outputs_cols[i]}_pass": passes[i],
                f"{outputs_cols[i]}_compile": compiles[i],
                f"{regen_str}parsed_code_{i}": parsed_codes[i],
                f"{regen_str}error_log_{i}": error_logs[i],
            }
        )
    # add sample ranking heuristics
    results_dict[model_name].update(
        {
            f"{regen_str}best_mean_logp_pass": passes[n],
            f"{regen_str}best_mean_logp_compile": compiles[n],
            f"{regen_str}best_sum_logp_pass": passes[n + 1],
            f"{regen_str}best_sum_logp_compile": compiles[n + 1],
            f"{regen_str}random_pass": passes[n + 2],
            f"{regen_str}random_compile": compiles[n + 2],
        }
    )
    # add model_name prefix
    # results_dict[model_name] = {f'{key}': value for key, value in results_dict[model_name].items()}
    df_result = pd.concat(
        [pd.DataFrame(result, index=[0]) for result in results_dict.values()], axis=1
    )  # now we have for one sample, all models results in cols
    return df_result


def sample_eval_parallel(
    base_path, model_name, idxs, df_with_outputs, n, k, n_jobs, options, regen=False
):
    """
    Evaluate the model outputs in parallel for a batch of samples
    base_path: str, path to the base directory.
    model_name: name of the model
    idxs: list, list of indices for the samples
    rows: list, list of rows for the samples
    n: int, number of generations from the model
    k: pass @ k evaluation
    """
    start, end = list(idxs)[0], list(idxs)[-1] + 1
    rows = df_with_outputs.iloc[start:end].iterrows()
    batch_results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(eval_sample_k)(
            base_path,
            model_name,
            row,
            n,
            k,
            idx,
            options.seed,
            options.temperature,
            options,
            regen=regen,
        )
        for idx, (_, row) in zip(idxs, rows)
    )  #

    results_dfs = [make_result_df(results, options) for results in batch_results]
    # results is list of dfs. make a df concat by rows, ignore index
    df_results = pd.concat(
        results_dfs, axis=0, ignore_index=True
    )  # now we have for all samples in minibatch, all models results in cols
    return df_results


def evaluate_model(
    options, df_with_outputs, eval_path_csv, bs=8, regen=False, df_updated=None
):
    """
    Evaluate the model on the test cases in the dataframe
    options: argparse.Namespace, options for the evaluation
    df_with_outputs: pd.DataFrame, dataframe with the model outputs
    return: pd.DataFrame, dataframe with the evaluation results
    """
    model_name = options.model_name.split("/")[-1]
    base_path = options.base_path
    empty_count = 0
    # print(df_with_outputs.columns)

    if df_updated is not None:
        start = len(df_updated)
    else:
        start = 0

    for i in tqdm(range(start, len(df_with_outputs), bs)):
        end = min(i + bs, len(df_with_outputs))
        # assert len(list(rows)) == end-i, "Row length does not match."
        idxs = range(i, end)
        # evaluate the model outputs
        batch_df_results = sample_eval_parallel(
            base_path,
            model_name,
            idxs,
            df_with_outputs,
            options.n_generate,
            options.k,
            options.n_jobs,
            options,
            regen=regen,
        )
        # update main df with the results (adding the cols to right indices)
        if df_updated is None:
            df_updated = batch_df_results
        else:
            df_updated = pd.concat(
                [df_updated, batch_df_results], axis=0, ignore_index=True
            )

        if options.enable_wandb:
            wandb.log(
                {
                    col: df_updated[col].mean()
                    for col in df_updated.columns
                    if "pass" in col or "compile" in col
                }
            )
            # log samples processed
            wandb.log({"samples_processed": df_updated.dropna().shape[0]})
        df_updated.to_csv(eval_path_csv, index=False)

        if options.debug_mode:
            print("Done first iteration, exiting debug successfully.")
            exit(0)
    # concat df_with_outputs and df_updated col axis
    df_with_outputs = pd.concat([df_with_outputs, df_updated], axis=1)
    df_with_outputs.to_csv(eval_path_csv, index=False)
    df_with_outputs["model_name"] = model_name
    if options.enable_wandb:
        wandb.log({"eval_df": wandb.Table(data=df_with_outputs)})
    print("Evaluation complete!")
    return df_with_outputs
