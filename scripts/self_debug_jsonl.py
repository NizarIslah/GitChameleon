#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path
import csv
import sys

# increase CSV field size limit to the max the platform allows
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        # on some platforms sys.maxsize is too large—scale it down
        max_int = int(max_int / 10)

def load_gt_jsonl(gt_path):
    """
    Load ground-truth JSONL into a dict mapping example_id -> record
    """
    gt_map = {}
    with gt_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] GT JSONL line {lineno}: invalid JSON: {e}", file=sys.stderr)
                continue
            eid = rec.get("example_id")
            if eid is None:
                print(f"[WARN] GT JSONL line {lineno}: missing 'example_id'", file=sys.stderr)
                continue
            if eid in gt_map:
                print(f"[WARN] duplicate GT example_id={eid} at line {lineno}, overwriting", file=sys.stderr)
            gt_map[eid] = rec
    return gt_map

def process_csv_file(csv_path, gt_map, out_path):
    """
    For a single CSV file, join with gt_map on example_id and write out_path JSONL
    """
    with csv_path.open("r", encoding="utf-8", newline="") as cf, \
         out_path.open("w", encoding="utf-8") as outf:
        reader = csv.DictReader(cf)
        if "example_id" not in reader.fieldnames:
            print(f"[ERROR] {csv_path.name}: missing 'example_id' column", file=sys.stderr)
            return
        try:
            for rownum, row in enumerate(reader, start=2):
                eid = row.get("example_id")
                if not eid:
                    print(f"[WARN] {csv_path.name} row {rownum}: empty 'example_id', skipping", file=sys.stderr)
                    continue
                gt_rec = gt_map.get(eid)
                if gt_rec is None:
                    print(f"[WARN] {csv_path.name} row {rownum}: no GT record for example_id={eid}, skipping", file=sys.stderr)
                    continue
                # Merge row (CSV) and gt_rec (JSON), with gt_rec overriding on conflict
                combined = {**row, **gt_rec}
                outf.write(json.dumps(combined, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[ERROR] {csv_path.name} row {rownum}: error processing row: {e}", file=sys.stderr)
            return
    # Print summary of processed rows
    print(f"[INFO] Wrote {out_path} ({csv_path.name} → {out_path.name})")

def main():
    p = argparse.ArgumentParser(
        description="For each CSV in a directory, join with a single GT JSONL on example_id, producing one JSONL per CSV."
    )
    p.add_argument("gt_jsonl",    type=Path, help="Ground-truth JSONL file (must have 'example_id')")
    p.add_argument("csv_dir",     type=Path, help="Directory containing CSV files (with 'example_id' column)")
    p.add_argument("output_dir",  type=Path, help="Directory to write per-CSV output JSONL files")
    args = p.parse_args()

    # Validate inputs
    if not args.gt_jsonl.is_file():
        print(f"[ERROR] GT JSONL not found: {args.gt_jsonl}", file=sys.stderr)
        sys.exit(1)
    if not args.csv_dir.is_dir():
        print(f"[ERROR] CSV directory not found: {args.csv_dir}", file=sys.stderr)
        sys.exit(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load ground-truth records once
    gt_map = load_gt_jsonl(args.gt_jsonl)

    # Process each CSV file
    for csv_path in sorted(args.csv_dir.glob("*.csv")):
        out_path = args.output_dir / (csv_path.stem + ".jsonl")
        process_csv_file(csv_path, gt_map, out_path)

if __name__ == "__main__":
    main()
