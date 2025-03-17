import gzip
import json
import os
from os import PathLike
from typing import Dict, Iterable

import pandas as pd


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


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
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

# # Example usage:
# if __name__ == "__main__":
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
