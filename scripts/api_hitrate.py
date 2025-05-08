#!/usr/bin/env python3
import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Set
import re
import ast

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

def extract_api_calls_with_aliases(code_snippet):
    """
    Extract API calls (function calls) from a Python code snippet, considering aliases and direct imports.

    Args:
        code_snippet (str): The Python code snippet as a string.

    Returns:
        set: A set of normalized API calls used in the code snippet.
    """
    class APICallVisitor(ast.NodeVisitor):
        def __init__(self):
            self.api_calls = set()
            self.aliases = {}

        def visit_Import(self, node):
            for alias in node.names:
                self.aliases[alias.asname or alias.name] = alias.name

        def visit_ImportFrom(self, node):
            module = node.module
            for alias in node.names:
                full_name = f"{module}.{alias.name}" if module else alias.name
                self.aliases[alias.asname or alias.name] = full_name

        def get_full_attr(self, node):
            if isinstance(node, ast.Name):
                return self.aliases.get(node.id, node.id)
            elif isinstance(node, ast.Attribute):
                return f"{self.get_full_attr(node.value)}.{node.attr}"
            return ""

        def visit_Call(self, node):
            full_name = self.get_full_attr(node.func)
            if full_name:
                self.api_calls.add(full_name)
            self.generic_visit(node)

    tree = ast.parse(code_snippet)
    visitor = APICallVisitor()
    visitor.visit(tree)
    return visitor.api_calls

def compare_api_calls(code_snippet1, code_snippet2):
    """
    Compare API calls between two code snippets.

    Args:
        code_snippet1 (str): The first Python code snippet.
        code_snippet2 (str): The second Python code snippet.

    Returns:
        bool: True if the two code snippets use the same API calls, False otherwise.
    """
    calls1 = extract_api_calls_with_aliases(code_snippet1)
    calls2 = extract_api_calls_with_aliases(code_snippet2)
    print("Ground Truth API Calls:", calls1)
    print("Generated Code API Calls:", calls2)
    return calls1.issubset(calls2)

def process_file(input_path: Path, output_path: Path, sol_field: str):
    recs_with_hitrate = []
    total = 0
    # Read and augment
    with input_path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] {input_path.name} parse error: {e}", file=sys.stderr)
                continue

            total += 1
            if not rec.get("solution_api_call", False):
                continue

            try:
                solution_code = rec.get(sol_field, "")
                solution_code = extract_code(solution_code)
                sol_api_calls = extract_api_calls_with_aliases(solution_code)
                api_calls = set(rec.get("api_calls", []))
            except Exception as e:
                # print(f"[WARN] {input_path.name} line {total}: error processing record: {e}", file=sys.stderr)
                continue

            api_hit = 1 if sol_api_calls.issubset(api_calls) else 0

            rec["api_hit"] = api_hit
            recs_with_hitrate.append(rec)

    # Write out
    with output_path.open("w", encoding="utf-8") as outfile:
        for rec in recs_with_hitrate:
            outfile.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Summary
    hits = sum(r["api_hit"] for r in recs_with_hitrate)
    count = len(recs_with_hitrate)
    pct = hits / count * 100 if count else 0.0
    # print(f"[INFO] {input_path.name}: wrote {count} records, api_hit {hits}/{count} \t")
    print(f"{str(input_path.name).split('_eval')[0]} \t {pct:.1f}%", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description="For each JSONL in input directory, add 'api_hit' to records and write to output directory"
    )
    parser.add_argument("input_dir",  type=Path,
                        help="Directory containing input .jsonl files")
    parser.add_argument("output_dir", type=Path,
                        help="Directory to write augmented .jsonl files")
    parser.add_argument("--sol-field", default="answer",
                        help="Field name for solution code (default: answer)")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"[ERROR] Input directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for input_jsonl in sorted(args.input_dir.glob("*.jsonl")):
        output_jsonl = args.output_dir / input_jsonl.name
        process_file(input_jsonl, output_jsonl, args.sol_field)

if __name__ == "__main__":
    main()