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


def extract_api_calls_with_aliases(code_snippet: str) -> set[str]:
    """
    Extract API calls (function calls) from a Python code snippet, considering
    direct imports, aliases, and attribute chains.

    Args:
        code_snippet: Python source as a string.

    Returns:
        A set of fully qualified call names (e.g. 'torch.from_numpy', 'scipy.special.i0').
    """
    class APICallVisitor(ast.NodeVisitor):
        def __init__(self):
            self.api_calls: set[str] = set()
            self.aliases: dict[str, str] = {}

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                name = alias.name
                asname = alias.asname or name.split('.')[0]
                self.aliases[asname] = name

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            module = node.module or ''
            for alias in node.names:
                fullname = f"{module}.{alias.name}" if module else alias.name
                asname = alias.asname or alias.name
                self.aliases[asname] = fullname

        def get_full_attr(self, node: ast.AST) -> str:
            if isinstance(node, ast.Name):
                return self.aliases.get(node.id, node.id)
            if isinstance(node, ast.Attribute):
                prefix = self.get_full_attr(node.value)
                return f"{prefix}.{node.attr}" if prefix else node.attr
            return ''

        def visit_Call(self, node: ast.Call) -> None:
            # Handle direct-name calls via alias lookup
            if isinstance(node.func, ast.Name):
                name = self.aliases.get(node.func.id, node.func.id)
                self.api_calls.add(name)
            else:
                full_name = self.get_full_attr(node.func)
                if full_name:
                    self.api_calls.add(full_name)
            # Recurse to catch nested calls or calls in args
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
            # if total == 33:
            #     print("ground truth code:", rec.get("solution", ""))
            #     print("solution code:", solution_code)
            #     print("solution api calls:", sol_api_calls)
            #     print("gt api calls:", api_calls)
            #     print("------------------------------------------------")
            api_hit = 1 if all(call in sol_api_calls for call in api_calls) else 0

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