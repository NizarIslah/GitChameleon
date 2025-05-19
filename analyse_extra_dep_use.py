#!/usr/bin/env python3
"""count_extra_dep_usage.py
Analyse extra‑dependency declarations vs. usage **for every model‑output JSONL**
found in a directory (or for a single file).  Results are assembled into a
`pandas` DataFrame that can optionally be saved to CSV.

Metrics per file
----------------
* **Declared** – total declared extra‑dependency entries.
* **Used** – number of those references that appear in the code.
* **Ref %** – 100·Used/Declared (or ``NaN`` when none declared).
* **Samples w/ Extras** – count of samples that declare at least one extra.
* **Samples w/out Usage** – among the above, those whose code uses **none** of
  its declared extras.

Usage
-----
```bash
python count_extra_dep_usage.py \
  --fix dataset/final_fix_dataset.jsonl \
  --orig all_eval_data/ \
  --out-csv metrics.csv        # optional
```
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd  # type: ignore

# ---------------------------------------------------------------------------
# Regex & constants
# ---------------------------------------------------------------------------

CODE_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def extract_code(answer: str) -> str:
    """Return the Python snippet inside the first markdown code fence."""
    m = CODE_FENCE_RE.search(answer)
    return m.group(1) if m else ""


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_answer_lookup(orig_path: Path) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for obj in load_jsonl(orig_path):
        eid = obj.get("example_id")
        if eid is not None:
            lookup[eid] = extract_code(obj.get("answer", ""))
    return lookup


def code_uses_pkg(code: str, pkg: str) -> bool:
    return bool(re.search(rf"\b{re.escape(pkg)}\.", code))


# ---------------------------------------------------------------------------
# Core logic – single‑file analysis
# ---------------------------------------------------------------------------


def analyse_file(
    fix_objects: List[dict], orig_path: Path, verbose: bool = False
) -> Tuple[str, int, int, int, int]:
    ans_lookup = build_answer_lookup(orig_path)
    declared = used = swx = swu = 0

    for fix_obj in fix_objects:
        code = ans_lookup.get(fix_obj.get("example_id"), "")
        pkgs = [
            dep.split("==", 1)[0].strip()
            for dep in (fix_obj.get("extra_dependencies", []) or [])
        ]

        if pkgs:
            swx += 1
        used_any = False

        for pkg in pkgs:
            declared += 1
            if code_uses_pkg(code, pkg):
                used += 1
                used_any = True
                if verbose:
                    print(f"USED   {orig_path.name}: {pkg}")
            else:
                if verbose:
                    print(f"UNUSED {orig_path.name}: {pkg}")

        if pkgs and not used_any:
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
        "--fix", type=Path, required=True, help="JSONL with `extra_dependencies` field."
    )
    p.add_argument(
        "--orig",
        type=Path,
        required=True,
        help="Single JSONL file **or** directory of files.",
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

    fix_objects = list(load_jsonl(args.fix))

    orig_paths = (
        sorted(args.orig.glob("*.jsonl")) if args.orig.is_dir() else [args.orig]
    )
    if not orig_paths:
        sys.exit(f"No JSONL files to process in '{args.orig}'.")

    rows = [analyse_file(fix_objects, path, args.verbose) for path in orig_paths]

    df = pd.DataFrame(
        rows,
        columns=["file", "declared", "used", "samples_w_extras", "samples_wout_usage"],
    )
    df["ref_%"] = (
        (100 * df["used"] / df["declared"]).round(1).where(df["declared"] != 0)
    )

    # Pretty print to console
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
