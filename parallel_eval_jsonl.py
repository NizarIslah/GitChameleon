#!/usr/bin/env python3
import json
import os
import argparse
import re
from tqdm import tqdm
import pandas as pd
import subprocess
import tempfile
import py_compile
import wandb
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.eval_sample import eval_sample

def run_script(env_path, py_file="temp.py"):
    python_executable = os.path.join(env_path, "bin", "python")
    if py_file is None:
        return False, False, "", ""

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
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            exit_code = result.returncode
            error_log = result.stderr
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
    result = {
        "compiled_manual": bool(1 - compile_code),
        "passed_manual": bool(1 - exit_code),
        "output_manual": error_log,
    }
    return result  # 1 = pass, 0 = fail

def extract_code(text: str) -> str:
    """Parse raw string into python code"""
    try:
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
    except Exception as e:
        try:
            match = re.search(r"```(.*?)```", rf"{text}", re.DOTALL)  # anthropic
        except Exception as e:
            print("Error: ", e)
            match = None
    return match.group(1) if match else text


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


def process_record(idx, record, starting_codes, manual_tests, env_dir, test_dir):
    """
    Process one JSON record: run eval_sample() and return a dict
    with example_id, code_id, output, passed, compiled, and idx.
    """
    example_id = get_example_id(record)
    try:
        example_id = int(example_id)
        code = starting_codes[example_id]
        manual_test = manual_tests[example_id]
        solution = get_solution(record)
        env_path = os.path.join(env_dir, f"gcham_venv_{example_id}")

        test_file_path = os.path.join(test_dir, f"test_sample_{example_id}.py")
        with open(test_file_path, "r") as tf:
            test_file_content = tf.read()

        code_dict = {
            "test_file": test_file_content,
            "codes": {"solution_code": {"code": solution}},
        }
        eval_res = eval_sample(example_id, env_path, code_dict)["codes"][
            "solution_code"
        ]
        res = {
            "idx": idx,
            "example_id": example_id,
            "code_id": "solution_code",
            "output": eval_res.get("output", "").strip(),
            "passed": eval_res.get("pass", False),
            "compiled": eval_res.get("compile", True),
        }
    except Exception as e:
        print(f"Error processing record (hidden) {idx}: {e}")
        res = {
            "idx": idx,
            "example_id": example_id,
            "code_id": "solution_code",
            "output": f"Error: {e}",
            "passed": False,
            "compiled": False,
        }
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            test_code = solution + '\n' + manual_test
            test_file = os.path.join(temp_dir, f"manual_test_sample_{example_id}.py")
            with open(test_file, "w") as f:
                f.write(test_code)
            eval_res_manual = run_script(env_path, test_file)
        res.update(
            {
                "output_manual": eval_res_manual.get("output_manual", "").strip(),
                "passed_manual": eval_res_manual.get("passed_manual", False),
                "compiled_manual": eval_res_manual.get("compiled_manual", True),
            }
        )

    except Exception as e:
        print(f"Error processing record (visible) {idx}: {e}")
        res.update({
            "output_manual": f"Error: {e}",
            "passed_manual": False,
            "compiled_manual": False,
        })
    return res


def main():
    parser = argparse.ArgumentParser(
        description="Process a JSONL file in parallel with eval_sample and save results."
    )
    parser.add_argument("data_file", help="Path to the dataset JSONL file to process")
    parser.add_argument(
        "jsonl_file", help="Path to the model outputs JSONL file to process"
    )
    parser.add_argument("env_dir", help="Path to the dir where environments live")
    parser.add_argument("test_dir", help="Path to the dir where test files are stored")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of threads to use (default: CPU count)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log results to Weights & Biases (wandb)",
    )
    args = parser.parse_args()

    if args.wandb:
        run = wandb.init(
            project="GC_Evals_EMNLP",
            entity="cl4code",
            name=os.path.basename(args.jsonl_file),
            config={"jsonl_file": args.jsonl_file},
        )

    # Load JSONL records
    starting_codes = {}
    manual_tests = {}
    with open(args.data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                starting_codes[int(data["example_id"])] = data["starting_code"]
                manual_tests[int(data["example_id"])] = data["test"]

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
            exe.submit(process_record, idx, rec, starting_codes, manual_tests, args.env_dir, args.test_dir)
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

    # log to wandb
    if args.wandb:
        run.log({"eval_results": wandb.Table(dataframe=df)})
        # log as an artifact
        artifact = wandb.Artifact(
            name=os.path.basename(output_csv),
            type="evaluation",
            description="Evaluation results of the model outputs",
        )
        artifact.add_file(output_csv)
        run.log_artifact(artifact)

    # fraction passed
    passed = df["passed"].sum()
    total = len(df)
    print(f"[✓] {passed}/{total} tests passed (hidden) ({passed/total:.2%})")
    compiled = df["compiled"].sum()
    print(f"[✓] {compiled}/{total} tests compiled (hidden) ({compiled/total:.2%})")

    # fraction passed manual
    passed_manual = df["passed_manual"].sum()
    total_manual = len(df)
    print(f"[✓] {passed_manual}/{total_manual} tests passed (visible) ({passed_manual/total_manual:.2%})")
    compiled_manual = df["compiled_manual"].sum()
    print(f"[✓] {compiled_manual}/{total_manual} tests compiled (visible) ({compiled_manual/total_manual:.2%})")

    if args.wandb:
        run.log({
            "pass_at_1_hidden": passed / total,
            "pass_at_1_visible": passed_manual / total_manual,
            "compiled_at_1_hidden": compiled / total,
            "compiled_at_1_visible": compiled_manual / total_manual,
        })
        run.finish()


if __name__ == "__main__":
    main()
