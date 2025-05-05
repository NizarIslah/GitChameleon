#!/usr/bin/env python3
import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Set

def extract_api_calls_with_aliases(code: str) -> Set[str]:
    """
    Extract API calls (module.func or func) from the given code via AST.
    """
    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.calls = set()
            self.aliases = {}

        def visit_Import(self, node):
            for alias in node.names:
                name = alias.asname or alias.name
                self.aliases[name] = alias.name

        def visit_ImportFrom(self, node):
            mod = node.module or ""
            for alias in node.names:
                asn = alias.asname or alias.name
                fullname = f"{mod}.{alias.name}" if mod else alias.name
                self.aliases[asn] = fullname

        def visit_Call(self, node):
            fn = node.func
            if isinstance(fn, ast.Attribute):
                val = fn.value
                if isinstance(val, ast.Name):
                    mod = self.aliases.get(val.id, val.id)
                    self.calls.add(f"{mod}.{fn.attr}")
            elif isinstance(fn, ast.Name):
                self.calls.add(self.aliases.get(fn.id, fn.id))
            self.generic_visit(node)

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    visitor = Visitor()
    visitor.visit(tree)
    return visitor.calls

def main():
    parser = argparse.ArgumentParser(
        description="For each record in a JSONL, detect if any API call appears in the solution portion."
    )
    parser.add_argument("input_jsonl", type=Path,
                        help="Path to input JSONL (must contain example_id, starting_code, solution fields)")
    parser.add_argument("output_jsonl", type=Path,
                        help="Path to write output JSONL (with example_id and solution_api_call bool)")
    parser.add_argument("--id-field", default="example_id",
                        help="JSON key for example ID (default: example_id)")
    parser.add_argument("--start-field", default="starting_code",
                        help="JSON key for starter code (default: starting_code)")
    parser.add_argument("--sol-field", default="solution",
                        help="JSON key for solution code (default: solution)")
    args = parser.parse_args()

    if not args.input_jsonl.is_file():
        print(f"[ERROR] Input file not found: {args.input_jsonl}", file=sys.stderr)
        sys.exit(1)

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

            eid = rec.get(args.id_field)
            if eid is None:
                print(f"[WARN] line {lineno}: missing '{args.id_field}'", file=sys.stderr)
                continue

            start = rec.get(args.start_field, "")
            sol   = rec.get(args.sol_field, "")

            combined = f"{start}\n{sol}"
            calls = extract_api_calls_with_aliases(combined)

            # check if any call string appears in the solution text
            solution_api_call = any(call in sol for call in calls)

            out_rec = {
                args.id_field: eid,
                "solution_api_call": solution_api_call
            }
            outf.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print(f"Wrote detection results to {args.output_jsonl}", file=sys.stderr)

if __name__ == "__main__":
    main()
