#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path

def load_jsonl(path: Path):
    """
    Generator that yields each JSON object in a JSONL file.
    """
    with path.open('r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                sys.stderr.write(f"Error parsing {path} at line {lineno}: {e}\n")
                continue


def extract_errors(text: str):
    """
    Given an error traceback string, extract all exception type names (e.g., ValueError, TimeoutError).
    Returns a list of names.
    """
    if not text:
        return []
    # regex to capture words ending with 'Error' or 'Exception'
    pattern = re.compile(r"\b([A-Za-z]+(?:Error|Exception))\b")
    return pattern.findall(text)


def process_file(path: Path, key: str = 'output'):
    """
    Process a single JSONL file, extract error categories from the specified key,
    and return a dict mapping error type -> count.
    Counts each occurrence, even if multiple per record.
    """
    counts = {}
    for record in load_jsonl(path):
        raw = record.get(key, '')
        # ensure string
        if not isinstance(raw, str):
            continue
        errors = extract_errors(raw)
        for err in errors:
            counts[err] = counts.get(err, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Scan a directory of JSONL files, count error types in a given field across each file and overall."
    )
    parser.add_argument('input_dir', type=Path,
                        help='Directory containing .jsonl files to process')
    parser.add_argument('-k', '--key', default='output',
                        help="JSON key containing the traceback text (default: 'output')")
    parser.add_argument('-a', '--aggregate', action='store_true',
                        help='Also print aggregated counts across all files')
    parser.add_argument('-o', '--output', type=Path,
                        help='Output file to write aggregated counts (default: stdout)')
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f"{args.input_dir} is not a directory")

    files = sorted(args.input_dir.glob('*.jsonl'))
    if not files:
        parser.error(f"No .jsonl files found in {args.input_dir}")

    overall = {}
    for path in files:
        counts = process_file(path, key=args.key)
        print(f"{path.name}: {counts}")
        if args.aggregate:
            for err, cnt in counts.items():
                overall[err] = overall.get(err, 0) + cnt

    if args.aggregate:
        print("\nAggregated counts across all files:")
        print(overall)
        if args.output:
            with args.output.open('w', encoding='utf-8') as f:
                json.dump(overall, f, indent=2)
        else:
            print(json.dumps(overall, indent=2))

if __name__ == '__main__':
    main()
