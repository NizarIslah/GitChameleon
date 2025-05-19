#!/usr/bin/env python3
"""count_extra_dep_usage.py
Given two JSONL files—(i) a *fix* file that lists `extra_dependencies`
and (ii) the *original* file containing the solution `answer` code—this
script computes four metrics:

1. **Total declared extra dependencies** (package entries across all samples).
2. **Total actually referenced dependencies** in the generated code.
3. **Number of samples that declare ≥1 extra dependency.**
4. **Number of those samples whose code uses *none* of its declared
   extras.**

A dependency is considered *used* if the solution code contains a token
`pkg_name.` (simple regex search).

Default paths (override with flags):
  * Fix file     : dataset/final_fix_dataset.jsonl
  * Original file: gt4o_t0.jsonl

Usage examples
--------------
```bash
python count_extra_dep_usage.py              # uses defaults
python count_extra_dep_usage.py --fix my_fix.jsonl --orig answers.jsonl -v
```
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Dict, Iterable, List

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CODE_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_code(answer: str) -> str:
    """Return the Python snippet inside the first markdown code fence."""
    m = CODE_FENCE_RE.search(answer)
    return m.group(1) if m else ""


def load_jsonl(path: Path) -> Iterable[dict]:
    """Yield JSON objects from a .jsonl file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_answer_lookup(orig_path: Path) -> Dict[str, str]:
    """Map example_id → raw code string extracted from `answer`."""
    lookup: Dict[str, str] = {}
    for obj in load_jsonl(orig_path):
        eid = obj.get("example_id")
        if eid is not None:
            lookup[eid] = extract_code(obj.get("answer", ""))
    return lookup


def code_uses_pkg(code: str, pkg: str) -> bool:
    """Return *True* if `code` references `pkg.`."""
    return bool(re.search(rf"\b{re.escape(pkg)}\.", code))


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse declared vs used extra deps")
    parser.add_argument(
        "--fix",
        type=Path,
        help="Path to JSONL fix dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--orig",
        type=Path,
        help="Path to original JSONL with answer code (default: %(default)s)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print per‑example details"
    )
    args = parser.parse_args()

    answer_lookup = build_answer_lookup(args.orig)

    total_declared = 0
    total_used = 0
    samples_with_extras = 0
    samples_without_usage = 0

    for fix_obj in load_jsonl(args.fix):
        eid = fix_obj.get("example_id")
        code = answer_lookup.get(eid, "")

        extras: List[str] = fix_obj.get("extra_dependencies", []) or []
        pkgs = [dep.split("==", 1)[0].strip() for dep in extras]

        if pkgs:
            samples_with_extras += 1

        used_any = False
        for pkg in pkgs:
            total_declared += 1
            if code_uses_pkg(code, pkg):
                total_used += 1
                used_any = True
                if args.verbose:
                    print(f"USED   {eid}: {pkg}")
            else:
                if args.verbose:
                    print(f"UNUSED {eid}: {pkg}")

        if pkgs and not used_any:
            samples_without_usage += 1

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------
    print("Total declared extra dependencies :", total_declared)
    print("Total actually referenced         :", total_used)
    if total_declared:
        print(
            f"Reference rate                    : {100*total_used/total_declared:.1f}%"
        )
    print()
    print("Num samples with defined extras   :", samples_with_extras)
    print("Num samples whose code uses none of its extras :", samples_without_usage)


if __name__ == "__main__":
    if len(sys.argv) == 1:  # No arguments provided
        sys.argv.extend(
            [
                "--fix",
                "dataset/final_fix_dataset.jsonl",
                "--orig",
                "wandb_results/gpt_4o_t0.jsonl",
            ]
        )
    main()
