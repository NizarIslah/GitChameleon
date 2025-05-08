#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="For each JSONL in a directory, find the corresponding CSV (<basename>_eval_results.csv), "
                    "extract one or more keys from the JSONL records, join on example_id, and write out the updated CSV."
    )
    parser.add_argument("dir", type=Path,
                        help="Directory containing JSONL and CSV files")
    parser.add_argument("--keys", nargs="+", required=True,
                        help="One or more JSON keys to extract and add as new columns in the CSVs")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="If provided, write updated CSVs here; otherwise overwrite originals")
    parser.add_argument("--skip-cot", action="store_true",
                        help="Skip JSONL files that contain 'cot' in their name (default: False)")
    args = parser.parse_args()

    base_dir = args.dir
    keys = args.keys
    out_dir = args.out_dir or base_dir
    if args.out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for jsonl_path in sorted(base_dir.glob("*.jsonl")):
        # optional skip logic
        if args.skip_cot and "cot" in jsonl_path.name:
            print(f"[INFO] Skipping {jsonl_path.name} (contains 'cot')", file=sys.stderr)
            continue

        base = jsonl_path.stem
        csv_name = f"{base}_eval_results.csv"
        csv_path = base_dir / csv_name
        if not csv_path.exists():
            print(f"[WARN] No CSV found for {jsonl_path.name} → expected {csv_name}", file=sys.stderr)
            continue

        # Load JSONL into dict: example_id -> record
        record_map = {}
        with jsonl_path.open("r", encoding="utf-8") as jf:
            for ln in jf:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except json.JSONDecodeError as e:
                    print(f"[WARN] {jsonl_path.name} line parse error: {e}", file=sys.stderr)
                    continue
                eid = rec.get("example_id")
                if eid is None:
                    print(f"[WARN] {jsonl_path.name} missing example_id in record", file=sys.stderr)
                    continue
                record_map[str(eid)] = rec  # ensure string keys

        # Load CSV
        df = pd.read_csv(csv_path)
        if "example_id" not in df.columns:
            print(f"[ERROR] {csv_name} missing 'example_id' column, skipping", file=sys.stderr)
            continue

        # Ensure example_id is string for mapping
        df["example_id"] = df["example_id"].astype(str)

        # For each requested key, add a column
        for key in keys:
            # map df key to answer if df key is in (output, solution)
            key_ = "answer" if key in ("output", "solution") else key
            # extract the key from the JSONL record_map
            df[key_] = df["example_id"].map(
                lambda eid: record_map.get(eid, {}).get(key, None)
            )
            # print(df[key].head(5))
            # warn if any values are missing
            missing = df.loc[df[key_].isna(), "example_id"].tolist()
            if missing:
                print(f"[WARN] {csv_name}: no '{key_}' for example_ids {missing}", file=sys.stderr)
                # optionally fill with empty string instead of NaN:
                df[key_] = df[key_].fillna("")

        # Write out
        out_path = out_dir / csv_name
        df.to_csv(out_path, index=False)
        print(f"[INFO] Wrote updated CSV to {out_path}")

if __name__ == "__main__":
    main()
