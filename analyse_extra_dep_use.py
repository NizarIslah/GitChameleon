#!/usr/bin/env python3
"""count_extra_dep_usage.py — **v5**
Analyse extra‑dependency declarations versus usage **for every model‑output
JSONL** found in a directory (or for a single file). Results are gathered into
a `pandas` *DataFrame* that can optionally be saved to CSV.

New in v5
---------
* **No heuristic fallbacks** – `code_uses_pkg` now relies *solely* on the AST
  analysis.  If the snippet cannot be parsed, the function simply returns
  `False`; there is no regex back‑off.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple
import inspect

import pandas as pd  # type: ignore

# ---------------------------------------------------------------------------
# Regex & constants
# ---------------------------------------------------------------------------

CODE_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def extract_code(block: str) -> str:
    """Return the Python snippet inside the first markdown code fence."""
    m = CODE_FENCE_RE.search(block or "")
    return m.group(1) if m else block  # fall back to raw text if no fence


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# ---------------------------------------------------------------------------
# AST‑based dependency analysis helpers
# ---------------------------------------------------------------------------


def _collect_pkg_aliases(tree: ast.AST, pkg: str) -> Set[str]:
    """Return all root identifiers that refer to *pkg* (itself or aliased)."""
    roots: Set[str] = {pkg}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == pkg:
                    roots.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] == pkg:
                for alias in node.names:
                    roots.add(alias.asname or alias.name)
    return roots


def code_uses_pkg(code: str, pkg: str) -> bool:
    """Return *True* if `code` contains a call or attribute access whose *root*
    identifier resolves to *pkg* or any of its aliases.  The snippet is
    `textwrap.dedent`‑ed before parsing to tolerate uneven indentation.
    If the snippet fails to parse, the function returns *False* (no regex
    fallback)."""
    try:
        tree = ast.parse(textwrap.dedent(code))
    except:
        # If the code cannot be parsed, we assume it doesn't use the package
        return False
    roots = _collect_pkg_aliases(tree, pkg)

    class CallCollector(ast.NodeVisitor):
        def __init__(self):
            self.calls = []
            self._current = []
            self._in_call = False

        def visit_Call(self, node):
            self._current = []
            self._in_call = True
            self.generic_visit(node)

        def visit_Attribute(self, node):
            if self._in_call:
                self._current.append(node.attr)
            self.generic_visit(node)

        def visit_Name(self, node):
            if self._in_call:
                self._current.append(node.id)
                self.calls.append(".".join(self._current[::-1]))
                # Reset the state
                self._current = []
                self._in_call = False
            self.generic_visit(node)

    # Get the source code of the function as a string
    cc = CallCollector()
    cc.visit(tree)
    for call in cc.calls:
        base = call.split(".")[0]
        if base in roots:
            return True
    return False


# ---------------------------------------------------------------------------
# Ground‑truth (solution) usage annotations
# ---------------------------------------------------------------------------


def annotate_solution_usage(fix_objects: List[dict]) -> dict:
    extras_declared = extras_used = samples_w_extras = samples_w_usage = 0

    for obj in fix_objects:
        sol_code = extract_code(obj.get("solution", ""))
        pkgs = [
            dep.split("==", 1)[0].strip()
            for dep in obj.get("extra_dependencies", []) or []
        ]
        if pkgs:
            samples_w_extras += 1
        used_any = False
        for pkg in pkgs:
            extras_declared += 1
            if code_uses_pkg(sol_code, pkg):
                extras_used += 1
                used_any = True
        if used_any:
            samples_w_usage += 1
        obj["_solution_uses"] = used_any

    return {
        "samples_w_extras": samples_w_extras,
        "samples_w_usage": samples_w_usage,
        "extras_declared": extras_declared,
        "extras_used": extras_used,
    }


# ---------------------------------------------------------------------------
# Per‑file analysis
# ---------------------------------------------------------------------------


def build_answer_lookup(orig_path: Path) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for obj in load_jsonl(orig_path):
        eid = obj.get("example_id")
        if eid is not None:
            lookup[eid] = extract_code(obj.get("answer", ""))
    return lookup


def analyse_file(
    relevant_fix_objects: List[dict], orig_path: Path, verbose: bool = False
) -> Tuple[str, int, int, int, int]:
    ans_lookup = build_answer_lookup(orig_path)
    declared = used = swx = swu = 0
    for fix_obj in relevant_fix_objects:
        code = ans_lookup.get(fix_obj.get("example_id"), "")
        pkgs = [
            dep.split("==", 1)[0].strip()
            for dep in (fix_obj.get("extra_dependencies", []) or [])
        ]
        swx += 1
        used_any = False
        for pkg in pkgs:
            declared += 1
            if code_uses_pkg(code, pkg):
                used += 1
                used_any = True
                if verbose:
                    print(f"USED   {orig_path.name}: {pkg}")
            elif verbose:
                print(f"UNUSED {orig_path.name}: {pkg}")
        if not used_any:
            swu += 1
    return orig_path.name, declared, used, swx, swu


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Analyse extra‑dependency usage across model outputs."
    )
    p.add_argument(
        "--fix",
        type=Path,
        required=True,
        help="JSONL with `extra_dependencies` & `solution` fields.",
    )
    p.add_argument(
        "--orig",
        type=Path,
        required=True,
        help="Single JSONL file *or* directory of files.",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        help="If provided, write the DataFrame to this CSV file.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if not args.fix.is_file():
        sys.exit(f"Fix dataset '{args.fix}' not found.")

    fix_objects: List[dict] = list(load_jsonl(args.fix))

    # Analyse ground‑truth solutions
    soln_stats = annotate_solution_usage(fix_objects)
    print("\n=== Ground‑truth (solution) stats ===")
    print(
        (
            "Samples w/ extras: {samples_w_extras}\n"
            "Samples where extras USED: {samples_w_usage} ({p:.1f}%)\n"
            "Extras declared / used: {extras_declared} / {extras_used} ({q:.1f}%)"
        ).format(
            p=100
            * soln_stats["samples_w_usage"]
            / max(1, soln_stats["samples_w_extras"]),
            q=100 * soln_stats["extras_used"] / max(1, soln_stats["extras_declared"]),
            **soln_stats,
        )
    )

    # Keep only examples whose solution truly uses extras
    relevant_fix_objects = [o for o in fix_objects if o.get("_solution_uses")]
    if not relevant_fix_objects:
        sys.exit("No examples where the solution actually uses its extra dependencies.")

    orig_paths = (
        sorted(args.orig.glob("*.jsonl")) if args.orig.is_dir() else [args.orig]
    )
    if not orig_paths:
        sys.exit(f"No JSONL files to process in '{args.orig}'.")

    rows = [
        analyse_file(relevant_fix_objects, path, args.verbose) for path in orig_paths
    ]

    df = pd.DataFrame(
        rows,
        columns=["file", "declared", "used", "samples_w_extras", "samples_wout_usage"],
    )
    df["ref_%"] = (
        (100 * df["used"] / df["declared"]).round(1).where(df["declared"] != 0)
    )

    print("\n=== Model metrics (on solution‑relevant examples) ===")
    print(df.to_markdown(index=False))

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\nSaved metrics to {args.out_csv}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--fix",
                "dataset/final_fix_dataset.jsonl",
                "--orig",
                "all_eval_data",
                "--out-csv",
                "extra_dep_use.csv",  # optional
            ]
        )
    main()
