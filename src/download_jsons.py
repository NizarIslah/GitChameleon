import wandb
import json
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from collections import defaultdict
import os
import json


def extract_json_from_wandb(filter_key, model_name, seed, temp, n_generate, cot=False):
    # Initialize a W&B API instance
    api = wandb.Api()

    # Define the project and entity (username or team name)
    entity = "cl4code"  # Replace with your entity
    project = "GitChameleon_new"  # Replace with your project name

    # Define the date after which to filter runs (use the format YYYY-MM-DD)
    filter_date = "2025-03-15"  # Example date

    # Convert the filter_date to datetime for comparison
    filter_date_time = datetime.strptime(filter_date, "%Y-%m-%d")
    print(f"Filtering runs created after {filter_date_time}")

    if "/fast/dmisra/hf_tmp/" not in model_name and filter_key == filter_key:
        model_name = f"/fast/dmisra/hf_tmp/{model_name}"

    filters = {
        "generate": {
            "config.n_samples": n_generate,
            "State": "finished",
            "config.model": model_name,
            "config.bs": 200,
            "config.temperature": temp,
            "config.backend": "vllm",
            "config.cot": cot,
            "config.prefetched": True,
        },
        "regenerate": {
            "config.n_samples": 1,
            "State": "finished",
            "config.model": model_name,
            "config.bs": 200,
            "config.temperature": temp,
            "config.backend": "vllm",
            "config.cot": cot,
            "config.prefetched": True,
            "config.feedback": True,
        },
    }
    # Fetch all runs from the project
    filtered_runs = api.runs(path=f"{entity}/{project}", filters=filters[filter_key])

    # Filter runs by creation date (created_at)
    # Function to parse timestamps with or without milliseconds
    def parse_created_at(created_at):
        try:
            # Try to parse the timestamp without milliseconds
            return datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            # If it fails, parse with milliseconds
            return datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%fZ")

    filtered_runs = [
        run
        for run in filtered_runs
        if parse_created_at(run.created_at) > filter_date_time
    ]
    assert len(filtered_runs) > 0, "No runs found. Please check the filters."
    # sort by name
    filtered_runs = sorted(filtered_runs, key=lambda x: x.name)
    print(
        f"Total & Unique runs satisfying the criteria: {len(filtered_runs)}, {len(set([run.name for run in filtered_runs]))}"
    )
    print(f"Created at: {[run.created_at for run in filtered_runs]}")

    found = False
    file_found = None
    for run in filtered_runs:
        # print(run.summary)
        for file in run.files():
            if (
                "outputs"
                and "jsonl" in file.name
                and ("evals" in file.name or filter_key != "evaluate")
            ):
                # print(file.name)
                # file.download(replace=True)
                # get df from it
                file_found = file
                found = True
                # data_json = json.load(open(file.name))
                # cols = data_json['columns']
                # data = data_json['data']
                # table = pd.DataFrame(data, columns=cols)
                # if len(table) >= len(largest_table):
                #     largest_table = table
                # found=True
                # # delete file
                # os.remove(file.name)
                # print(f"Found table.json for run {run.name}. Rows: {len(table)}")
                # break
    if not found:
        print(f"No file found")
        exit(1)
    return file_found


def concatenate_jsonl_files(input_dir, model_name, temperature, output_file):
    # List to hold concatenated data
    concatenated_data = []

    # Loop through files in the input directory
    for filename in os.listdir(input_dir):
        # print(f"Checking file: {filename}")
        # Check if the filename contains the model name and a seed number
        if (
            model_name in filename
            and str(temperature) in filename
            and filename.endswith(".jsonl")
        ):
            # Open and read each matching jsonl file
            file_path = os.path.join(input_dir, filename)
            print(f"Processing file: {file_path}")

            with open(file_path, "r") as jsonl_file:
                for line in jsonl_file:
                    # Each line in a jsonl file is a separate JSON object
                    json_object = json.loads(line)
                    concatenated_data.append(json_object)

    # Write concatenated data to the output file
    with open(output_file, "w") as output_jsonl_file:
        for item in concatenated_data:
            output_jsonl_file.write(json.dumps(item) + "\n")

    print(f"Concatenated {len(concatenated_data)} records into {output_file}")


def extract_every_n_lines_with_offset(input_path, output_path, n, offset):
    """
    Extracts every n-th line from the input JSONL file starting at the given offset,
    and writes those lines to the output JSONL file.

    Parameters:
    input_path (str): Path to the input JSONL file.
    output_path (str): Path to the output JSONL file.
    n (int): Step size (every n-th line will be taken).
    offset (int): The starting line index offset (zero-indexed).
    """
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for i, line in enumerate(infile):
            if i >= offset and (i - offset) % n == 0:
                outfile.write(line)


if __name__ == "__main__":
    # json_path="/home/mila/n/nizar.islah/GitChameleon/hf_tmp/CodeLlama-34b-Instruct-hf_outputs_0_0.8_100.jsonl"
    # model_name = "CodeLlama-34b-Instruct-hf"
    # outputs = defaultdict(list)
    # seed = 0
    # temperature = 0.8
    # n_samples=100
    # n_datapts=116
    # with open(json_path, 'r') as f:
    #     k = 0
    #     cur_task_id = 0
    #     # get num lines in file
    #     n_lines = sum(1 for line in f)
    #     print(n_lines)
    # with open(json_path, 'r') as f:
    #     # start from the last n_samples*n_datapts lines
    #     start_idx = n_lines - n_samples * n_datapts
    #     print(start_idx)
    #     for line_idx, line in enumerate(f):
    #         if line_idx < start_idx:
    #             continue
    #         # skip lines by seed (5 seeds). TODO: un-hardcode
    #         if (line_idx - start_idx) % 5 != seed and temperature > 0:
    #             continue
    #         resp = json.loads(line)
    #         task_id = resp["task_id"]
    #         if task_id != cur_task_id:
    #             k = 0
    #             cur_task_id = task_id
    #         solution = resp["solution"]
    #         outputs[f"{model_name}_output_{k}"].append(solution)
    #         k += 1
    # key = f"{model_name}_output_0"
    # key2 = f"{model_name}_output_1"
    # print(len(outputs[key]))
    # print(len(outputs[key2]))
    # outputs = pd.DataFrame(outputs)

    model_names = [
        # "DeepSeek-Coder-V2-Lite-Instruct",
        # "Qwen2.5-0.5B-Instruct",
        # "Yi-Coder-9B-Chat",
        # "Yi-Coder-1.5B-Chat",
        "CodeLlama-7b-Instruct-hf",
        "CodeLlama-13b-Instruct-hf",
        "CodeLlama-34b-Instruct-hf",
        # "Llama-3.1-8B-Instruct",
        # "Llama-3.1-70B-Instruct",
        "Qwen2-7B-Instruct",
        # "Qwen2.5-Coder-1.5B-Instruct",
        # "Qwen2.5-Coder-7B-Instruct",
        # "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct",
        "Codestral-22B-v0.1",
        # "Yi-1.5-6B-Chat",
        # "Yi-1.5-9B-Chat",
        # "Yi-1.5-34B-Chat"
        # "codegemma-7b-it",
        # "stable-code-instruct-3b",
        # "starcoder2-15b-instruct-v0.1",
        # "Qwen2-72B-Instruct",
        # "granite-3b-code-instruct-2k",
        # "granite-3b-code-instruct-128k",
        # "granite-8b-code-instruct-4k",
        # "granite-8b-code-instruct-128k",
        # "granite-20b-code-instruct-8k",
        # "granite-34b-code-instruct-8k",
        # "Phi-3.5-mini-instruct",
        # "Phi-3.5-MoE-instruct",
        "CodeQwen1.5-7B-Chat",
        # "Nxcode-CQ-7B-orpo"
    ]
    # model_names = ['granite-8b-code-instruct-4k', 'Qwen2.5-Coder-1.5B-Instruct', 'CodeLlama-34b-Instruct-hf',
    # 'Llama-3.1-70B-Instruct', 'stable-code-instruct-3b', 'Qwen2.5-Coder-7B-Instruct', 'Qwen2-7B-Instruct',
    # 'granite-3b-code-instruct-2k', 'Yi-1.5-34B-Chat', 'granite-34b-code-instruct-8k', 'Nxcode-CQ-7B-orpo',
    # 'granite-20b-code-instruct-8k', 'starcoder2-15b-instruct-v0.1', 'Llama-3.2-3B-Instruct',
    # 'granite-3b-code-instruct-128k', 'Llama-3.2-1B-Instruct', 'CodeQwen1.5-7B-Chat', 'Phi-3.5-mini-instruct',
    # 'Phi-3.5-MoE-instruct', 'granite-8b-code-instruct-128k', 'Qwen2-72B-Instruct']
    cot = False
    filter_key = "generate"
    if filter_key == "regenerate":
        seeds = range(5)
    else:
        seeds = [0]

    for seed in seeds:
        for model in tqdm(model_names):
            for cfg in [
                (seed, 0.0, 1)
            ]:  # , (seed, 0.3, 25)]: #, (seed, 0.8, 100)]:  # seed, temp, n_sample (seed, 0.3, 25),
                try:
                    json_file = extract_json_from_wandb(
                        filter_key, model, cfg[0], cfg[1], cfg[2], cot=cot
                    )
                except:
                    model_alt = model.replace("Instruct", "instruct")
                    try:
                        json_file = extract_json_from_wandb(
                            filter_key, model_alt, cfg[0], cfg[1], cfg[2], cot=cot
                        )
                    except Exception as e:
                        print(f"Failed for {model} with {cfg}")
                        print(e)
                        continue
                print(json_file)
                json_file.download(replace=True)
                print("Downloaded json file")
                # read a few lines
                with open(json_file.name, "r") as f:
                    for i in range(1):
                        print(f.readline())
                # save under new file name with cfg appended
                if cot:
                    new_name = json_file.name.replace(
                        ".jsonl", f"_{cfg[0]}_{cfg[1]}_{cfg[2]}_cot.jsonl"
                    )
                else:
                    new_name = json_file.name.replace(
                        ".jsonl", f"_{cfg[0]}_{cfg[1]}_{cfg[2]}.jsonl"
                    )
                if filter_key == "regenerate":
                    new_name = new_name.replace(".jsonl", "_regen.jsonl")
                import shutil

                shutil.move(json_file.name, new_name)
                print(f"Renamed file to {new_name}")

            # for i in range(5):
            #     input_file = new_name
            #     output_file = new_name.replace(".jsonl", f"_{i}.jsonl")
            #     n = 5
            #     offset = i
            #     extract_every_n_lines_with_offset(input_file, output_file, n, offset)
            #     print("Done")

    import sys

    # break
    # # # testing extract json from wandb
    # # json_file = extract_json_from_wandb(filter_key, "/fast/dmisra/hf_tmp/CodeLlama-7b-instruct-hf", 0, 0.3, 25, cot=False)
    # # json_file = extract_json_from_wandb(filter_key, "/fast/dmisra/hf_tmp/CodeLlama-7b-Instruct-hf", 0, 0.8, 100, cot=False)
    # # json_file = extract_json_from_wandb(filter_key, "/fast/dmisra/hf_tmp/CodeLlama-7b-Instruct-hf", 0, 0.0, 1, cot=False)
    # # print(json_file)
    # # json_file.download(replace=True)
    # # print("Downloaded json file")
    # # # read a few lines
    # # with open(json_file.name, 'r') as f:
    # #     for i in range(10):
    # #         print(f.readline())
    # exit(0)

    # # # Concatenate jsonl files
    # # Example usage:
    # input_directory = "/home/mila/n/nizar.islah/GitChameleon/gc_new_feedback/"  # Replace with the actual directory path
    # output_directory = "/home/mila/n/nizar.islah/GitChameleon/gc_new_feedback_concat/"  # Replace with the actual directory path
    # os.makedirs(output_directory, exist_ok=True)
    # for model_name in model_names:
    #     for temperature in (0.0, 0.3, 0.8):
    #         output_jsonl = f"{model_name}_{temperature}_error_feedback_prompts.jsonl"  # Path for output file
    #         concatenate_jsonl_files(input_directory, model_name, temperature, output_jsonl)
