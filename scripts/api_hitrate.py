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
    parser.add_argument("--sol-field", default="solution",
                        help="Field name for solution code (default: solution)")
    args = parser.parse_args()

    if not args.input_jsonl.is_file():
        print(f"[ERROR] Input file not found: {args.input_jsonl}", file=sys.stderr)
        sys.exit(1)

    total = 0
    sol_hits = 0
    with args.input_jsonl.open("r", encoding="utf-8") as inf, \
         args.output_jsonl.open("w", encoding="utf-8") as outf:
        for lineno, line in enumerate(inf, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] line {lineno}: invalid JSON: {e}", file=sys.stderr)
                continue

            sol = rec.get(args.sol_field, "")
            starting_code = rec.get("starting_code", "")
            combined = starting_code + sol
            calls = extract_api_calls_with_aliases(combined)
            rec["solution_api_call"] = any([call in sol for call in calls])
            total += 1
            if rec["solution_api_call"]:
                sol_hits += 1

            outf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[INFO] Wrote augmented JSONL to {args.output_jsonl}", file=sys.stderr)
    print(f"[INFO] With solution API calls: {sol_hits} / {total} ({sol_hits / total:.2%})", file=sys.stderr)

if __name__ == "__main__":
    main()
