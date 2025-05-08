#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import sys

pd.set_option('future.no_silent_downcasting', True)

def load_table(path: Path) -> pd.DataFrame:
    """
    Load a table from CSV or JSONL based on file extension.
    """
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported file type: {path.name}")

def main():
    p = argparse.ArgumentParser(
        description=(
            "For each CSV or JSONL in a directory, compute the mean of each specified column "
            "and then average those means into one score per file."
        )
    )
    p.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing .csv or .jsonl files to process"
    )
    p.add_argument(
        "-c", "--columns",
        nargs="+",
        required=True,
        help="One or more column names whose values to average"
    )
    args = p.parse_args()

    input_dir = args.input_dir
    cols = args.columns

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # collect all CSV and JSONL files
    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        print(f"No CSV or JSONL files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    results = []
    for path in files:
        try:
            df = load_table(path)
        except Exception as e:
            print(f"{path.name}\tERROR loading file: {e}", file=sys.stderr)
            continue

        n = len(df)
        if n == 0:
            print(f"{path.name}\tEMPTY")
            continue

        means = []
        for col in cols:
            # map "False" to 0 and "True" to 1
            df.loc[:, col] = df[col].replace({"False": 0, "True": 1})
            if col not in df.columns:
                print(f"Warning: column '{col}' not found in {path.name}", file=sys.stderr)
            else:
                means.append(df[col].sum() / n)

        if not means:
            print(f"{path.name}\tNO_VALID_COLUMNS")
            continue

        # derive an index from the filename (strip suffix and optional tags)
        index_str = path.stem
        if "_True_" in index_str:
            index_str = index_str.split("_True_")[1]
        index_str = index_str.split("_eval")[0]

        row = dict(zip(cols, means))
        row["_file"] = index_str
        results.append(row)

    if not results:
        print("No valid data to display.", file=sys.stderr)
        sys.exit(1)

    # build DataFrame, scale and sort
    out_df = pd.DataFrame(results).set_index("_file")
    out_df = (out_df * 100).round(2)
    out_df = out_df.sort_values(by=out_df.columns[0])

    print(out_df)


if __name__ == "__main__":
    main()
