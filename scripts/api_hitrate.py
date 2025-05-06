#!/usr/bin/env python3
import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Set

import ast

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

def main():
    parser = argparse.ArgumentParser(
        description="Augment each record in a JSONL with a boolean 'solution_api_call' field."
    )
    parser.add_argument("input_jsonl", type=Path,
                        help="Path to the input JSONL (must contain solution field)")
    parser.add_argument("output_jsonl", type=Path,
                        help="Path to write the augmented JSONL")
    parser.add_argument("--sol-field", default="answer",
                        help="Field name for solution code (default: solution)")
    args = parser.parse_args()

    if not args.input_jsonl.is_file():
        print(f"[ERROR] Input file not found: {args.input_jsonl}", file=sys.stderr)
        sys.exit(1)

    recs_with_hitrate = []

    with args.input_jsonl.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSONL line parse error: {e}", file=sys.stderr)
                continue

            if not rec["solution_api_call"]:
                continue

            # Extract the solution code
            try:
                solution_code = rec.get(args.sol_field)
                sol_api_calls = extract_api_calls_with_aliases(solution_code)
                api_calls = rec.get("api_calls", [])
                api_hit = 1 if sol_api_calls.issubset(api_calls) else 0
            except Exception as e:
                print(f"[ERROR] Error processing record: {e}", file=sys.stderr)
                continue
            rec["api_hit"] = api_hit
            recs_with_hitrate.append(rec)

    # Write the augmented record to the output JSONL
    with args.output_jsonl.open("a", encoding="utf-8") as outfile:
        for rec in recs_with_hitrate:
            outfile.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # Print summary of processed records
    print(f"[INFO] Wrote {len(recs_with_hitrate)} records with API hit rate to {args.output_jsonl}")
    print(f"[INFO] API hitrate: {sum(rec['api_hit'] for rec in recs_with_hitrate)} / {len(recs_with_hitrate)} ({sum(rec['api_hit'] for rec in recs_with_hitrate) / len(recs_with_hitrate):.2%})", file=sys.stderr)


if __name__ == "__main__":
    main()
