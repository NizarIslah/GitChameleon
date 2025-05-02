#!/usr/bin/env python3
import json
import os
import argparse
from tqdm import tqdm
import pandas as pd
from src.eval_sample import eval_sample


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Process a JSONL file, run eval_sample on each, and save results."
    )
    parser.add_argument("jsonl_file", help="Path to the JSONL file to process")
    parser.add_argument(
        "env_dir", help="Path to the dir where the environments have been created"
    )
    parser.add_argument(
        "test_dir", help="Path to the dir where the test files are stored"
    )
    args = parser.parse_args()

    # 1) Read all JSONL lines into a list of dicts
    data = []
    with open(args.jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))

    # Prepare a list to collect per-example results
    results = []

    # 2) Iterate over each record (JSON object)
    for idx, record in tqdm(
        enumerate(data), total=len(data), desc="Processing JSON lines"
    ):
        example_id = record.get("example_id")
        try:
            example_id = int(example_id)
            code = record.get("starting_code", "")
            solution = record.get("solution", "")
            env_path = os.path.join(args.env_dir, f"gcham_venv_{example_id}")

            # Locate the matching test file
            test_file_path = os.path.join(
                args.test_dir, f"test_sample_{example_id}.py"
            )
            with open(test_file_path, "r") as tf:
                test_file_content = tf.read()

            # Build the code_dict and run evaluation
            code_dict = {
                "test_file": test_file_content,
                "codes": {"solution_code": {"code": code + solution}},
            }
            eval_res = eval_sample(example_id, env_path, code_dict, coverage=False)["codes"]["solution_code"]

            # Append row for this example
            results.append({
                "example_id": example_id,
                "code_id": "solution_code",
                "output": eval_res.get("output", "").strip(),
                "passed": eval_res.get("pass", False),
                "compiled": eval_res.get("compile", True),
                "coverage": eval_res.get("coverage", -1),
            })

        except Exception as e:
            # On error, still record a failure row
            results.append({
                "example_id": example_id,
                "code_id": "solution_code",
                "output": f"Error: {e}",
                "passed": False,
                "compiled": False,
                "coverage": 0,
            })
            print(f"[!] Error processing record {idx} (example_id={example_id}): {e}")
            continue

    # 3) Build DataFrame and save CSV
    df = pd.DataFrame(results)
    output_csv = os.path.splitext(args.jsonl_file)[0] + "_verification_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"[✓] Saved results DataFrame to {output_csv}")


if __name__ == "__main__":
    main()