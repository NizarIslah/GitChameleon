#!/usr/bin/env python3
import json
import os
import subprocess
from tqdm import tqdm
import argparse


python_versions = {
    "3.7": "/root/.pyenv/versions/3.7.17/bin/python",
    "3.9": "/root/.pyenv/versions/3.9.19/bin/python",
    "3.10": "/root/.pyenv/versions/3.10.14/bin/python",
}


def get_python_path(python_version):
    return python_versions.get(python_version)


def main():
    """
    For each line in the provided JSONL file (where each line is a JSON object),
    read the fields, convert them into the argument list for verify_dataset.sh,
    and run the shell script.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Process a JSONL file and verify dataset."
    )
    parser.add_argument("jsonl_file", help="Path to the JSONL file to process")
    args = parser.parse_args()

    # 1) Read all JSONL lines into a list of dicts
    data = []
    with open(args.jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line:
                continue
            # Parse JSON
            data.append(json.loads(line))

    # 2) Iterate over each record (JSON object)
    for idx, record in tqdm(
        enumerate(data), total=len(data), desc="Processing JSON lines"
    ):
        # Pull each field from the record, defaulting to "-" if missing
        library = record.get("library", "-")
        version = record.get("version", "-")
        func_name = record.get("name_of_class_or_func", "-")
        change = record.get("type_of_change", "-")
        problem = record.get("problem", "-")
        code = record.get("starting_code", "-")
        solution = record.get("solution", "-")
        test = record.get("test", "-")
        dep = record.get("additional_dependencies", "")
        python_version = record.get("python_version", "3.10")
        python_executable = get_python_path(python_version)
        # Replace literal "\n" with actual newlines in potentially multiline fields
        # (This helps pass multiline code to the shell script correctly)
        code = code.replace("\\n", os.linesep)
        solution = solution.replace("\\n", os.linesep)
        test = test.replace("\\n", os.linesep)

        # 3) Build the argument list in the order your verify_dataset.sh expects
        args = [
            library,  # $1
            version,  # $2
            func_name,  # $3
            change,  # $4
            problem,  # $5
            code + solution + os.linesep + test,  # $6
            dep,  # $7
            python_executable,  # $8
        ]
        # 4) Call verify_dataset.sh with these arguments
        result = subprocess.run(
            ["bash", "verify_dataset.sh", *args], capture_output=True, text=True
        )

        # 5) Examine the output and return code
        stdout_lower = result.stdout.lower() if result.stdout else ""
        if "this was the exit code: 0" in stdout_lower:
            print(f"[{idx}] SUCCESS for: {func_name}")
        elif "this was the exit code: 1" in stdout_lower:
            print(result.stdout)
            print(result.stderr)
            print(f"[{idx}] FAILURE for: {func_name}")
        else:
            print(f"[{idx}] UNKNOWN ERROR for: {func_name}")


if __name__ == "__main__":
    main()
