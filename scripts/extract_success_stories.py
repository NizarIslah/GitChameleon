#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def load_jsonl(path: Path):
    """
    Load a JSONL file into a dict mapping example_id to the full record.
    """
    records = {}
    with path.open('r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                sys.stderr.write(f"Error parsing {path} at line {lineno}: {e}\n")
                continue
            ex_id = obj.get('example_id')
            if ex_id is None:
                sys.stderr.write(f"Warning: missing 'example_id' in {path} at line {lineno}\n")
                continue
            records[ex_id] = obj
    return records


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Process pairs of JSONL files in two directories, extract error tracebacks and generate formatted summaries.'
        )
    )
    parser.add_argument('dir1', type=Path,
                        help='Directory of original JSONL files (contains output and output_manual).')
    parser.add_argument('dir2', type=Path,
                        help='Directory of debug JSONL files (contains regenerated answers).')
    parser.add_argument('out_dir', type=Path,
                        help='Directory where formatted text files will be written.')
    args = parser.parse_args()

    if not args.dir1.is_dir():
        parser.error(f"{args.dir1} is not a directory")
    if not args.dir2.is_dir():
        parser.error(f"{args.dir2} is not a directory")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Map filenames to paths
    files1 = {p.name: p for p in args.dir1.glob('*.jsonl')}
    files2 = {p.name: p for p in args.dir2.glob('*.jsonl')}
    common = set(files1) & set(files2)
    if not common:
        parser.error('No matching .jsonl filenames found in both directories.')

    for name in sorted(common):
        path1 = files1[name]
        path2 = files2[name]
        recs1 = load_jsonl(path1)
        recs2 = load_jsonl(path2)
        common_ids = set(recs1) & set(recs2)

        out_path = args.out_dir / f"{Path(name).stem}.txt"
        with out_path.open('w', encoding='utf-8') as out_f:
            for ex_id in sorted(common_ids):
                r1 = recs1[ex_id]
                r2 = recs2[ex_id]

                # only include where passed_manual is False in first and True in second
                if str(r1.get('passed_manual', '')).strip().lower() != 'false':
                    continue
                if str(r2.get('passed_manual', '')).strip().lower() != 'true':
                    continue

                # Extract fields
                python_version = r1.get('python_version', '')
                library = r1.get('library', '')
                version = r1.get('version', '')
                problem = r1.get('problem', '')
                starting_code = r1.get('starting_code', '')
                answer = r1.get('answer', '')
                explanation1 = r1.get('explanation', '')
                passed_manual1 = r1.get('passed_manual', '')
                output_manual = r1.get('output_manual', '')
                answer_debugged = r2.get('answer', '')
                passed_manual2 = r2.get('passed_manual', '')
                explanation2 = r2.get('explanation', '')

                block = f"""
python {python_version} library {library}-{version}
problem: {problem}
starting code: {starting_code}
model output: {answer}
explanation: {explanation1}
passed: {passed_manual1}
error traceback: {output_manual}
model regenerated output: {answer_debugged}
explanation: {explanation2}
passed: {passed_manual2}
"""
                out_f.write(block + "\n")

        print(f"Processed {name}, wrote formatted summary to {out_path}", file=sys.stderr)

if __name__ == '__main__':
    main()
