#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
from pathlib import Path


def get_csv_lines(path: Path) -> list[dict]:
    """
    Read a CSV file and return a list of dictionaries representing each row.

    Args:
        path (Path): The path to the CSV file.

    Returns:
        list[dict]: A list of dictionaries where each dictionary represents a CSV row.
    """
    with open(path, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        return [row for row in reader]


def main(csv_paths: list[Path], output_jsonl_path: Path):
    """
    Merge CSV files and write the merged rows as JSON Lines to the output file.

    Args:
        csv_paths (list[Path]): A list of paths to CSV files.
        output_jsonl_path (Path): The path to the output JSONL file.
    """
    # Read and merge CSV data from all provided paths
    csv_lines = list(itertools.chain(*[get_csv_lines(path) for path in csv_paths]))

    # Write merged CSV rows to the output JSONL file
    with open(output_jsonl_path, "w", encoding="utf-8") as fout:
        for line in csv_lines:
            fout.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multiple CSV files and output the result as a JSON Lines file."
    )
    parser.add_argument(
        "csv_paths",
        nargs="+",
        type=Path,
        help="One or more paths to CSV files to be merged.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_jsonl_path",
        type=Path,
        required=True,
        help="Path to the output JSONL file.",
    )
    args = parser.parse_args()

    main(args.csv_paths, args.output_jsonl_path)
