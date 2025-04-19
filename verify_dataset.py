#!/usr/bin/env python3
import json
import os
import subprocess
from tqdm import tqdm
import argparse
from src.eval_sample import eval_sample


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Process a JSONL file and verify dataset."
    )
    parser.add_argument("jsonl_file", help="Path to the JSONL file to process")
    parser.add_argument(
        "env_dir", help="Path to the dir where the environnement have been created"
    )
    parser.add_argument(
        "test_dir", help="Path to the dir where the test files are stored"
    )
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
        example_id = int(record.get("example_id", ""))
        code = record.get("starting_code", "-")
        solution = record.get("solution", "-")
        env_path = os.path.join(args.env_dir, f"gcham_venv_{example_id}")

        test_folder_path = os.path.join(
            args.test_dir, f"test_early_sample_{example_id}/"
        )

        # Get the first file starting with "test" in the test folder
        test_files = [
            f
            for f in os.listdir(test_folder_path)
            if f.startswith("test") and f.endswith(".py")
        ]
        assert (
            len(test_files) == 1
        ), f"Expected exactly one test file in {test_folder_path}, but found {len(test_files)}"
        test_file_path = os.path.join(test_folder_path, test_files[0])
        code_dict = {}
        # Read the test file content
        with open(test_file_path, "r") as test_file:
            test_file_content = test_file.read()
        # Update the code_dict with the test file content
        code_dict["test_file"] = test_file_content
        code_dict["codes"] = {"solution_code": {"code": code + solution}}
        print(eval_sample(example_id, env_path, code_dict))


if __name__ == "__main__":
    main()
