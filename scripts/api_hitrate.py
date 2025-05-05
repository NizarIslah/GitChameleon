#!/usr/bin/env python3
import ast
import argparse
import json
import sys
from pathlib import Path
from typing import Set, Dict
import re
import os
import py_compile

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

def extract_api_calls_with_aliases(code_snippet: str) -> Set[str]:
    """
    Extract API calls (function calls) from a Python code snippet,
    considering aliases and direct imports.
    """
    class APICallVisitor(ast.NodeVisitor):
        def __init__(self):
            self.api_calls = set()
            self.aliases = {}

        def visit_Import(self, node):
            for alias in node.names:
                name = alias.asname or alias.name
                self.aliases[name] = alias.name

        def visit_ImportFrom(self, node):
            module = node.module or ""
            for alias in node.names:
                asname = alias.asname or alias.name
                fullname = f"{module}.{alias.name}" if module else alias.name
                self.aliases[asname] = fullname

        def visit_Call(self, node):
            # Handle module.func() or alias.func()
            if isinstance(node.func, ast.Attribute):
                val = node.func.value
                mod = None
                if isinstance(val, ast.Name):
                    mod = self.aliases.get(val.id, val.id)
                elif isinstance(val, ast.Attribute):
                    # chained imports: reconstruct e.g. pkg.mod
                    names = []
                    cur = val
                    while isinstance(cur, ast.Attribute):
                        names.append(cur.attr)
                        cur = cur.value
                    if isinstance(cur, ast.Name):
                        base = self.aliases.get(cur.id, cur.id)
                        names.append(base)
                        names.reverse()
                        mod = ".".join(names)
                if mod:
                    self.api_calls.add(f"{mod}.{node.func.attr}")
            # Handle direct func()
            elif isinstance(node.func, ast.Name):
                fn = node.func.id
                self.api_calls.add(self.aliases.get(fn, fn))
            self.generic_visit(node)

    tree = ast.parse(code_snippet)
    visitor = APICallVisitor()
    visitor.visit(tree)
    return visitor.api_calls

def load_jsonl(path: Path, code_key: str) -> Dict[str, str]:
    """
    Load a JSONL mapping example_id -> code (from the given code_key field).
    """
    d = {}
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            obj = json.loads(ln)
            eid = obj["example_id"]
            if code_key not in obj:
                print(f"[WARN] {path.name}: no '{code_key}' field for example_id={eid}", file=sys.stderr)
            d[eid] = obj.get(code_key, "")
    return d

def main():
    p = argparse.ArgumentParser(
        description="Compare API-call sets between ground truth (solution) and predictions (answer)."
    )
    p.add_argument("dataset",  type=Path,
                   help="Ground-truth JSONL (must have 'example_id' and 'solution' fields)")
    p.add_argument("--pred","-p", type=Path, required=True,
                   help="Prediction JSONL (must have 'example_id' and 'answer' fields)")
    p.add_argument("--out", "-o", type=Path, required=True,
                   help="Output JSONL with comparison results")
    args = p.parse_args()

    # Load ground truth from 'solution', predictions from 'answer'
    gt_map   = load_jsonl(args.dataset,   code_key="solution")
    starter_code_map = load_jsonl(args.dataset, code_key="starting_code")
    gt_code_map = {eid: starter_code_map[eid] + gt_code for eid, gt_code in gt_map.items()}
    pred_map = load_jsonl(args.pred, code_key="answer")

    with Path(args.out / args.pred.name).open("w", encoding="utf-8") as outf:
        for eid, gt_code in gt_code_map.items():
            if eid not in pred_map:
                print(f"[WARN] No prediction for example_id={eid}", file=sys.stderr)
                continue
            pred_code = extract_code(pred_map[eid])

            try:
                # check it compiles
                # write to a tempfile
                with open(f"{eid}.py", "w") as f:
                    f.write(pred_code)
                py_compile.compile(f"{eid}.py", cfile=f"{eid}.pyc", doraise=True)
                os.remove(f"{eid}.pyc")
                os.remove(f"{eid}.py")
            except py_compile.PyCompileError as e:
                print(f"[WARN] Compilation error for example_id={eid}: {e.msg}", file=sys.stderr)
                continue

            calls_gt   = sorted(extract_api_calls_with_aliases(gt_code))
            calls_pred = sorted(extract_api_calls_with_aliases(pred_code))
            same       = set(calls_gt) == set(calls_pred)
            print(f"example_id={eid}: {calls_gt} == {calls_pred} -> {same}")

            rec = {
                "example_id": eid,
                "same_api_calls": same,
                "api_calls_gt": calls_gt,
                "api_calls_pred": calls_pred
            }
            outf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Comparison written to {Path(args.out / args.pred.name)}")

    # get average of the API call hitrate
    with open(args.out / args.pred.name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        total_calls = 0
        total_matches = 0
        for line in lines:
            rec = json.loads(line)
            if rec["same_api_calls"]:
                total_matches += len(rec["api_calls_pred"])
            total_calls += len(rec["api_calls_gt"])
        hitrate = total_matches / total_calls if total_calls > 0 else 0
        print(f"API call hitrate: {hitrate:.2%}")

if __name__ == "__main__":
    main()
