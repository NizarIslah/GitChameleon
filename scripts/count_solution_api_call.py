#!/usr/bin/env python3
import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Set

import ast

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

import re
import textwrap
from typing import Set, Dict

def extract_solution_calls(
    starting_code: str,
    solution_code: str
) -> Set[str]:
    """
    Extract all API calls in solution_code, resolving aliases declared
    in both starting_code and solution_code, *without* ever AST-parsing
    those possibly-broken snippets.
    """

    # 1) Normalize indentation
    start = textwrap.dedent(starting_code)
    sol   = textwrap.dedent(solution_code)

    # 2) Pull out all valid import lines from both snippets
    import_lines = re.findall(
        r'^\s*(?:'
        r'import\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)(?:\s+as\s+(\w+))?'
        r'|from\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s+import\s+([A-Za-z_]\w*)(?:\s+as\s+(\w+))?'
        r')\s*$',
        start + "\n" + sol,
        flags=re.MULTILINE
    )

    # 3) Build alias_map via regex groups
    alias_map: Dict[str,str] = {}
    for imp_mod, imp_as, from_mod, from_name, from_as in import_lines:
        if imp_mod:
            alias = imp_as or imp_mod.split('.', 1)[0]
            alias_map[alias] = imp_mod
        else:
            alias = from_as or from_name
            alias_map[alias] = f"{from_mod}.{from_name}"

    # 4) Find all NAME(.NAME)* calls in solution_code
    raw_calls = re.findall(r'\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(', sol)

    # 5) Remap heads through alias_map to full module paths
    final_calls: Set[str] = set()
    for entry in raw_calls:
        parts = entry.split('.')
        head, *rest = parts
        full_head = alias_map.get(head, head)
        fq = full_head + ('.' + '.'.join(rest) if rest else '')
        final_calls.add(fq)

    return final_calls





# --- Example ---
if __name__ == "__main__":
    starting = """
    import numpy as np
    from scipy.special import gammaln as scipy_gammaln
    """

    solution = """
    output = torch.from_numpy(scipy_gammaln(input_tensor.numpy()))
    """

    print(extract_solution_calls(starting, solution))
    # -> {'torch.from_numpy', 'scipy.special.gammaln', 'input_tensor.numpy'}



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
            sol_calls = extract_solution_calls(starting_code, sol)

            # if lineno < 30:
            #     print("solution code:", sol)
            #     print("combined code:", combined)
            #     print("all api calls:", calls)
            #     print("solution api calls:", sol_calls)
            #     print("------------------------------------------------")

            rec["solution_api_call"] = sol_calls != []
            rec["api_calls"] = list(sol_calls)
            total += 1
            if rec["solution_api_call"]:
                sol_hits += 1

            outf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[INFO] Wrote augmented JSONL to {args.output_jsonl}", file=sys.stderr)
    print(f"[INFO] With solution API calls: {sol_hits} / {total} ({sol_hits / total:.2%})", file=sys.stderr)

if __name__ == "__main__":
    main()
