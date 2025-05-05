#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import sys

def main():
    p = argparse.ArgumentParser(
        description="For each CSV in a directory, compute the mean of each specified column "
                    "and then average those means into one score per file."
    )
    p.add_argument(
        "csv_dir",
        type=Path,
        help="Directory containing .csv files to process"
    )
    p.add_argument(
        "-c", "--columns",
        nargs="+",
        required=True,
        help="One or more column names whose values to average"
    )
    args = p.parse_args()

    csv_dir = args.csv_dir
    cols = args.columns

    if not csv_dir.is_dir():
        print(f"Error: {csv_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    files = sorted(csv_dir.glob("*.csv"))
    if not files:
        print(f"No CSV files found in {csv_dir}", file=sys.stderr)
        sys.exit(1)

    files_df = []
    for csv_path in files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"{csv_path.name}\tERROR reading file: {e}", file=sys.stderr)
            continue

        n = len(df)
        if n == 0:
            print(f"{csv_path.name}\tEMPTY")
            continue

        means = []
        for col in cols:
            if col not in df.columns:
                print(f"Warning: column '{col}' not found in {csv_path.name}", file=sys.stderr)
            else:
                means.append(df[col].sum() / n)

        if not means:
            print(f"{csv_path.name}\tNO_VALID_COLUMNS")
            continue

        # save the means as a df with column names
        means_df = pd.DataFrame([means], columns=cols, index=[csv_path.name.split("_eval")[0]])
        # aggregate means df into files df
        files_df.append(means_df)

    pandas_df = pd.concat(files_df)
    # * 100 and round to 2 decimal places
    pandas_df = pandas_df * 100
    pandas_df = pandas_df.round(2)
    # sort by the first column
    pandas_df = pandas_df.sort_values(by=pandas_df.columns[0])
    # display the means
    print(pandas_df)

if __name__ == "__main__":
    main()
