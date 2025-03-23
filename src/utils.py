import gzip
import json
import os
from os import PathLike
from typing import Dict, Iterable
import pandas as pd
from tqdm import tqdm


def write_jsonl(
    filename: str, data: Iterable[Dict], append: bool = False, drop_builtin: bool = True
):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = "ab"
    else:
        mode = "wb"
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                for x in data:
                    if drop_builtin:
                        x = {k: v for k, v in x.items() if not k.startswith("_")}
                    gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with open(filename, mode) as fp:
            for x in data:
                if drop_builtin:
                    x = {k: v for k, v in x.items() if not k.startswith("_")}
                fp.write((json.dumps(x) + "\n").encode("utf-8"))


import gzip
import json
from typing import Iterable, Dict

def stream_jsonl(filename: str, seed: int) -> Iterable[Dict]:
    """
    Parses each JSONL line from the given file and yields it as a dictionary.
    If a line is empty (only whitespace), yields a dictionary with {"output": "", "seed": seed}.
    
    Parameters:
      filename (str): The path to the JSONL file (can be gzip compressed if ends with '.gz').
      seed (int): The seed value to include if an empty line is encountered.
    
    Returns:
      Iterable[Dict]: An iterable of dictionaries parsed from the file.
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    stripped_line = line.strip()
                    if stripped_line == "":
                        yield {"output": "", "seed": seed}
                    else:
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                stripped_line = line.strip()
                if json.loads(stripped_line) == "":
                    print("empty line")
                    yield {"output": "", "seed": seed}
                else:
                    yield json.loads(line)



def write_directory(directory: PathLike, data: Iterable[Dict]):
    os.makedirs(directory, exist_ok=True)
    counters = {}
    for sample in data:
        assert "solution" in sample, "Samples must come with `solution` field!"
        task_id = sample["task_id"].replace("/", "_")
        task_dir = os.path.join(directory, task_id)
        os.makedirs(task_dir, exist_ok=True)
        if task_id not in counters:
            counters[task_id] = 0
        sample_id = counters[task_id]
        with open(os.path.join(task_dir, f"{sample_id}.py"), "w") as f:
            f.write(sample["solution"])
        counters[task_id] += 1


def to_raw(string):
    return string.encode("unicode-escape").decode().replace("\\\\", "\\")

COT_PROMPT_TEMPLATE = """\
You are to solve this in python using {}-{}.
First, let's think step by step. Then, provide a self-contained Python script that solves the following problem in a markdown code block.
{}
"""

PROMPT_TEMPLATE = """\
You are to solve this in python using {}-{}. Provide a self-contained Python script that solves the following problem in a markdown code block.
{}
"""

library_specific_instructions = {
    "gradio" : "Do not launch a gradio interface."
}

def get_prompt(example, instruct=False, cot=False):
    if instruct:
        if cot:
            prompt_template = COT_PROMPT_TEMPLATE
        else:
            prompt_template = PROMPT_TEMPLATE
        prompt  = prompt_template.format(example['library'], example['version'], example['problem'])
        if example['library'] in library_specific_instructions.keys():
            prompt += library_specific_instructions[example['library']] 
        prompt += """\
Please start with the following markdown code block:
```
{}
```
""".format(example["starting_code"].replace('\\n', '\n'))
        return prompt
    else:
        raise ValueError("Not Implemeneted")

def get_prompt_feedback(example, generated_code, error_log):
    prompt_template = PROMPT_TEMPLATE
    prompt  = prompt_template.format(example['library'], example['version'], example['problem'])
    if example['library'] in library_specific_instructions.keys():
        prompt += library_specific_instructions[example['library']] 
    # add output here
    prompt += """\
Your solution was:
```
{}
```
""".format(generated_code.replace('\\n', '\n'))
    prompt += """\
Your solution had the following error:
```
{}
```
""".format(error_log.replace('\\n', '\n'))

    prompt += """\
Please start with the following markdown code block:
```
{}
```
""".format(example["starting_code"].replace('\\n', '\n'))
    return prompt


def generate_prompt(model_name, example, df_idx, sample_idx):
    base_model_name = model_name.split('/')[-1]
    parsed_code = example[f'parsed_code_{sample_idx}']
    error_log = example[f'error_log_{sample_idx}']
    # task_id is index of the example
    task_id = df_idx
    # print(type(parsed_code), type(error_log))
    if type(error_log) == float:
        error_log = ""
    if type(parsed_code) == float:
        parsed_code = ""
    prompt = get_prompt_feedback(example, parsed_code, error_log)
    prompt = {"task_id": task_id, "sample_id": sample_idx,"prompt": prompt}
    return prompt

def generate_prompt_with_error_log(model_name, n_generate, eval_df, indices):
    prompts = []
    df_indices = [i//n_generate for i in indices]
    sample_indices = [i%n_generate for i in indices]
    for df_idx, sample_idx in zip(df_indices, sample_indices):
        example = eval_df.iloc[df_idx]
        prompt = generate_prompt(model_name, example, df_idx, sample_idx)
        prompts.append(prompt)
    return prompts

def save_feedback_prompts_jsonl(model_name, n_generate, eval_df_path, jsonl_save_path):
    """
    Regenerates error log prompts based on evaluation data.

    This function:
      - Loads evaluation data from either a WandB table file or a CSV.
      - Filters the evaluation DataFrame based on the model name.
      - Sets up the tokenizer (unless prompt_saving is enabled).
      - Processes the evaluation data in batches to generate error log prompts.
      - If prompt_saving is enabled, saves the prompts to a JSONL file and exits;
        otherwise, returns the generated prompts.

    Parameters:
        options: An object with attributes including:
            - error_log_regenerate (bool)
            - wandb_table_file (str)
            - model_name (str)
            - seed (any)
            - temperature (any)
            - prompt_saving (bool)
            - batch_size (int)
            - n_generate (int)
            - model_url (str)
            - token (str)
        save_dir (str): Directory where the evaluation CSV is located.
        eval_save_file (str): Filename for the evaluation CSV (used if wandb_table_file is empty).

    Returns:
        If prompt_saving is False, returns a list of generated prompts.
    """
    eval_df = pd.read_csv(eval_df_path)
    print(eval_df.head())

    bs = 16
    n_samples = len(eval_df)
    total_gen = n_samples * n_generate
    print("n_samples * n_generate:", total_gen, "n_mb:", total_gen // bs + 1)
    all_prompts = []
    # Process evaluation data in batches.
    for i in tqdm(range(0, total_gen, bs)):
        stop = min(total_gen, i + bs)
        prompts = generate_prompt_with_error_log(model_name, n_generate, eval_df, range(i, stop))
        all_prompts.extend(prompts)

    with open(jsonl_save_path, 'w') as f:
        for prompt in all_prompts:
            f.write(json.dumps(prompt) + '\n')
    print(f"Saved error log prompts to {jsonl_save_path}")

    return all_prompts


def get_prompt_doc(example, instruct=False):
    if instruct:
        prompt_template = PROMPT_TEMPLATE
        prompt  = prompt_template.format(example['library'], example['version'], example['problem'])
        if example['library'] in library_specific_instructions.keys():
            prompt += library_specific_instructions[example['library']] 
        prompt += """\
Here is the documentation for the function to be used:
```
{}
```

""".format(example['docs'])
        prompt += """\
Please start with the following markdown code block:
```
{}
```
""".format(example["starting_code"].replace('\\n', '\n'))
    else:
        raise ValueError("Not Implemeneted")
    return prompt

def load_dataset(data_path):
    df = pd.read_csv(data_path)
    # convert to dataset with row index as key, row as dict value
    dataset = {index: row.to_dict() for index, row in df.iterrows()}
    return dataset

import pandas as pd

def move_rows_to_position(df, idx1, idx2, idx3):
    """
    Moves the block of rows from idx1 to idx2 (inclusive) so that in the final DataFrame,
    the block starts at index position idx2.
    
    Note: This assumes that the DataFrame is 0-indexed and that idx1 <= idx2.
    """
    # Extract the block of rows (using iloc ensures we're working by position)
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1
    block = df.iloc[idx1:idx2]
    # Remove these rows from the DataFrame
    df_remaining = df.drop(df.index[idx1:idx2])
    
    # Determine the insertion position in df_remaining:
    # We want the final DataFrame (after concatenation and reindexing) to have the block's first row at position idx2.
    #
    # Since we've removed rows from the original DataFrame, the remaining rows now are in a new order.
    # Here, we simply choose the insertion position as the minimum between idx2 and the number of remaining rows.
    insertion_index = idx1 + idx3-idx2
    
    # Insert the block into the remaining DataFrame at the computed insertion index.
    df_new = pd.concat([
        df_remaining.iloc[:insertion_index],
        block,
        df_remaining.iloc[insertion_index:]
    ]).reset_index(drop=True)
    
    return df_new

PROMPT_TEMPLATE="""\
1. Required Library:
<library>
{}=={} with python {}
</library>

2. Coding Problem:
<coding_problem>
{}
</coding_problem>

3. Starter Code:
<starter_code>
{}
</starter_code>
```

After writing your solution, please review it to ensure all requirements are met and the code is correct and efficient."""
    
# # Example usage:
if __name__ == "__main__":
    import pandas as pd
    import json

    samples = []
    with open("/home/mila/n/nizar.islah/scratch/GitChameleon/GitChameleon/dataset/all_samples_final_merged.jsonl", 'r') as f:
        idx=0
        for line in f:
            sample = json.loads(line)
            lib = sample['library']
            version = sample['version']
            problem = sample['problem']
            starter_code = sample['starting_code']
            if idx < 293:
                python_version = "3.10"
            else:
                python_version = "3.7"
            idx+=1
            prompt = PROMPT_TEMPLATE.format(lib,version,python_version,problem,starter_code)
            samples.append({"role": "user", "content": prompt})
    # write to a jsonl
    with open("dataset/all_samples_final_new.jsonl", 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    print("all samples written to jsonl")

    # samples = pd.read_csv("dataset/all_samples_final.csv")
    # idxs = (50, 150, 220)
    # prompts=[]
    # for idx in idxs:
    #     sample = samples.iloc[idx]
    #     lib = sample['library']
    #     version = sample['version']
    #     problem = sample['problem']
    #     starter_code = sample['starting_code']
    #     prompt = PROMPT_TEMPLATE.format(lib,version,problem,starter_code,lib,version,lib,version)
    #     prompts.append({"role": "user", "content": prompt})
    # # write to a jsonl
    # with open("dataset/test_new_prompts.jsonl", 'w') as f:
    #     for prompt in prompts:
    #         f.write(json.dumps(prompt) + '\n')


    # # combine csvs
    # import csv
    # input_csvs = [
    #     "src/gitchameleon samples to verify - Brice (1).csv",
    #     "src/gitchameleon samples to verify - zihan (1).csv",
    #     "src/gitchameleon samples to verify - Muawiz (2).csv",
    # ]
    # output_csv = "dataset/samples_collab_final.csv"
    # # they may have different columns
    # combined = []
    # columns = set()
    # for input_csv in input_csvs:
    #     with open(input_csv, newline='', encoding='utf-8') as csvfile:
    #         reader = csv.DictReader(csvfile)
    #         for row in reader:
    #             combined.append(row)
    #             columns.update(row.keys())
    # columns = list(columns)
    # with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=columns)
    #     writer.writeheader()
    #     for row in combined:
    #         writer.writerow(row)


    # df2=pd.read_csv(output_csv)
    # df1=pd.read_csv("/home/mila/n/nizar.islah/GitChameleon/dataset/combined_dataset.csv")
    # # merge with different columns some same.
    # all_columns = set(df1.columns).union(set(df2.columns))
    # df1 = df1.reindex(columns=all_columns)
    # df2 = df2.reindex(columns=all_columns)
    # df = pd.concat([df1, df2], ignore_index=True)
    # df=df.drop([110])
    # df=df.reset_index(drop=True)
    # df.to_csv("dataset/all_samples_final.csv", index=False)


    import csv
    ## feedback prompt saving ##
    # model_names = [
    #     "gemini15",
    #     "gemini-2.0-flash",
    #     "codegemma-7b-it",
    #     "gpt4o",
    #     "gpt4o_2",
    #     "gpt_mini",
    #     "Qwen2-7B-Instruct",
    #     "Llama-3.2-3B-Instruct",
    #     "CodeLlama-7b-Instruct-hf",
    #     "CodeLlama-13b-Instruct-hf",
    #     "CodeLlama-34b-Instruct-hf",
    #     "CodeQwen1.5-7B-Chat",
    #     "Codestral-22B-v0.1"
    # ]
    # for model_name in model_names:
    #     n_generate = 1
    #     temp = 0.0
    #     seed = 0
    #     eval_df_path = f"/home/mila/n/nizar.islah/GitChameleon/results/{model_name}/{model_name}_n{n_generate}_k1_T={temp}_seed{seed}_eval.csv"
    #     jsonl_save_path = f"/home/mila/n/nizar.islah/GitChameleon/results/feedback_prompts/{model_name}_n{n_generate}_k1_T={temp}_feedback_prompts.jsonl"
    #     save_feedback_prompts_jsonl(model_name, n_generate, eval_df_path, jsonl_save_path)

    #####  csv to jsonl saving #######0#

    # import csv
    # # File paths
    # input_csv = "dataset/all_samples_final.csv"
    # output_jsonl = "dataset/all_samples_final.jsonl"

    # # Read CSV and write JSONL
    # with open(input_csv, newline='', encoding='utf-8') as csvfile, open(output_jsonl, 'w', encoding='utf-8') as jsonlfile:
    #     reader = csv.DictReader(csvfile)
    #     for i,row in enumerate(reader):
    #         # if i==110:
    #         #     continue
    #         prompt = get_prompt(row, instruct=True, cot=False)
    #         json_obj = {"role": "user", "content": prompt}
    #         jsonlfile.write(json.dumps(json_obj) + "\n")

    ### gpt formatting to jsonl ###
    # import pickle
    # temp=0.8
    # seeds = 100
    # file_template = '/home/mila/n/nizar.islah/nizar.islah/GitChameleon/results/gpt_42_08/responses_{}.pkl'
    # n_generate = seeds//5
    # for seed in range(0, seeds, n_generate):
    #     out_jsonl_path = f'/home/mila/n/nizar.islah/nizar.islah/GitChameleon/results/gpt_42_08/responses_{temp}_{seed // n_generate}.jsonl'
    #     print(out_jsonl_path)
    #     data = []
    #     for i in range(n_generate):
    #         file_path = file_template.format(seed+i)
    #         assert os.path.isfile(file_path)
    #         if 'pkl' in file_path:
    #             data.extend([{'output': x, 'seed': seed+i} for x in pickle.load(open(file_path, 'rb'))])
    #         else: # jsonl
    #             data.extend(stream_jsonl(file_path, seed // n_generate))
    #     for i,d in enumerate(data):
    #         if not type(d) == dict:
    #             print(i, d, type(d))
    #     write_jsonl(out_jsonl_path, data)



#     # Read the CSV file into a DataFrame
#     df = pd.read_csv("/home/mila/n/nizar.islah/GitChameleon/dataset/updated_libraries.csv")
    
#     # Suppose you have a function find_lib that returns the first row position where the "library" column contains a given string.
#     def find_lib(df, library):
#         for idx, row in df.iterrows():
#             if library in str(row.get('library', '')):
#                 return idx
#         return None

#     idx3 = find_lib(df, 'librosa')
#     idx2 = find_lib(df, 'sympy')
#     idx1 = find_lib(df, 'django')
#     print(idx1, idx2, idx3)
#     # print(df[idx1:idx2])
#     # print(df[idx2:idx3])
#     # print(df[idx3:])

#     if idx1 is None or idx2 is None:
#         print("One or both specified libraries were not found in the DataFrame.")
#     else:
#         df_new = move_rows_to_position(df, idx1, idx2, idx3)
#         df_new.to_csv("/home/mila/n/nizar.islah/GitChameleon/dataset/env_ids_reordered.csv", index=False)
#         print(df_new[idx2:idx2+(idx3-idx1)])
#         print(df_new[idx2+(idx3-idx1):idx3])
#         print(df_new[idx3:])
#         print("CSV file has been reordered and saved.")
#         # only new samlpes
#         df_only_new = df_new[idx2:]
#         df_only_new.to_csv("/home/mila/n/nizar.islah/GitChameleon/dataset/env_ids_only_new.csv", index=False)
#         print("CSV file has been reordered and saved.")
