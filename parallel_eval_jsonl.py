#!/usr/bin/env python3
import json
import os
import argparse
import re
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.eval_sample import eval_sample

def extract_code(text: str) -> str:
    """Parse raw string into python code"""
    try:
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
    except Exception as e:
        try:
            match = re.search(r"```(.*?)```", rf'{text}', re.DOTALL) # anthropic
        except Exception as e:
            print("Error: ", e)
            match = None
    return match.group(1) if match else None


def get_solution(record):
    solution = record.get("answer", "")
    if solution == "":
        solution = record.get("solution", "")
    if solution == "":
        solution = record.get("output", "")
    if solution == "":
        raise ValueError("No solution found in record")
    return extract_code(solution)
    
def get_example_id(record):
    id = record.get("example_id", "")
    if id == "":
        id = record.get("sample_idx", "")
    if id == "":
        raise ValueError("No example_id found in record")
    return id


def process_record(idx, record, starting_codes, env_dir, test_dir):
    """
    Process one JSON record: run eval_sample() and return a dict
    with example_id, code_id, output, passed, compiled, and idx.
    """
    example_id = get_example_id(record)
    try:
        example_id = int(example_id)
        code = starting_codes[example_id]
        solution = get_solution(record)
        env_path = os.path.join(env_dir, f"gcham_venv_{example_id}")

        test_file_path = os.path.join(test_dir, f"test_sample_{example_id}.py")
        with open(test_file_path, "r") as tf:
            test_file_content = tf.read()

        code_dict = {
            "test_file": test_file_content,
            "codes": {"solution_code": {"code": solution}},
        }
        eval_res = eval_sample(example_id, env_path, code_dict)["codes"]["solution_code"]

        return {
            "idx": idx,
            "example_id": example_id,
            "code_id": "solution_code",
            "output": eval_res.get("output", "").strip(),
            "passed": eval_res.get("pass", False),
            "compiled": eval_res.get("compile", True),
        }

    except Exception as e:
        return {
            "idx": idx,
            "example_id": example_id,
            "code_id": "solution_code",
            "output": f"Error: {e}",
            "passed": False,
            "compiled": False,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Process a JSONL file in parallel with eval_sample and save results."
    )
    parser.add_argument("data_file", help="Path to the dataset JSONL file to process")
    parser.add_argument("jsonl_file", help="Path to the model outputs JSONL file to process")
    parser.add_argument("env_dir", help="Path to the dir where environments live")
    parser.add_argument("test_dir", help="Path to the dir where test files are stored")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of threads to use (default: CPU count)",
    )
    args = parser.parse_args()

   # Load JSONL records
    starting_codes = {}
    with open(args.data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                starting_codes[int(data["example_id"])] = data["starting_code"]

    # Load JSONL records
    outputs = []
    with open(args.jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                outputs.append(json.loads(line))

    results = []
    # Kick off parallel tasks
    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        futures = [
            exe.submit(process_record, idx, rec, starting_codes, args.env_dir, args.test_dir)
            for idx, rec in enumerate(outputs)
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            results.append(fut.result())

    # Sort back into original order
    results.sort(key=lambda row: row["idx"])
    # Build DataFrame, drop the helper idx column
    df = pd.DataFrame(results).drop(columns=["idx"])

    # Save CSV
    output_csv = os.path.splitext(args.jsonl_file)[0] + "_eval_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"[✓] Saved results to {output_csv}")
    # fraction passed
    passed = df["passed"].sum()
    total = len(df)
    print(f"[✓] {passed}/{total} tests passed ({passed/total:.2%})")
    compiled = df["compiled"].sum()
    print(f"[✓] {compiled}/{total} tests compiled ({compiled/total:.2%})")


if __name__ == "__main__":
    main()
