import argparse
import json
from pathlib import Path

SYS_PROMPT = "You are a skilled Python debugger tasked with resolving issues for a given solution of a coding problem. Your goal is to provide a clear, efficient, and correct solution that meets all the specified requirements and resolves the issue as detailed in the error trace.\n\nPlease provide your solution following these guidelines:\n\n1. Use the required library in your solution.\n2. Incorporate the provided starter code correctly.\n3. Write your solution in Python.\n4. Format your solution within a markdown code block.\n5. Ensure your code is clean and efficient.\n6. Output only the code block and nothing else. Do not add any in-line comments, documentations, references or usage examples.\n7. Make sure your code is executable in a non-interactive environment. For example, do not write code which requires building a Django project or deploying a web-app.\n8. Resolve the issues in the existing solution using the error trace from the unit test provided.\n\nExample output format:\n\n```python\n# [Your code here, incorporating the starter code]\n```\n\nAfter writing your solution, please review it to ensure all requirements are met and the code is correct and efficient.\n\nHere are the key elements for this task: "
SYS_PROMPT = """
You are a skilled Python debugger tasked with resolving issues for a given solution of a coding problem. Your goal is to provide a clear, efficient, and correct solution that meets all the specified requirements and resolves the issue as detailed in the error trace.

Please provide your solution following these guidelines:

1. Use the required library in your solution.
2. Incorporate the provided starter code correctly.
3. Write your solution in Python.
4. Format your solution within a markdown code block.
5. Ensure your code is clean and efficient.
6. Output only the code block and nothing else. Do not add any in-line comments, documentation, references, or usage examples.
7. Make sure your code is executable in a non-interactive environment. For example, do not write code which requires building a Django project or deploying a web app.
8. Resolve the issues in the existing solution using the error trace from the unit test provided.

Example output format:

```python
# [Your code here, incorporating the starter code]
```
After writing your solution, please review it to ensure all requirements are met and the code is correct and efficient.

Here are the key elements for this task: """
COT_SYS_PROMPT = SYS_PROMPT

def process_file(input_path: Path, output_path: Path):
    """
    Read a JSONL from input_path, add 'messages' and 'cot_messages' fields,
    then write to output_path.
    """
    with open(input_path, "r", encoding="utf-8") as fin:
        entries = [json.loads(line) for line in fin]

    for entry in entries:
        prompt = entry.get("prompt", "")
        entry["messages"] = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt},
        ]
        entry["cot_messages"] = [
            {"role": "system", "content": COT_SYS_PROMPT},
            {"role": "user", "content": prompt},
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for entry in entries:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Process a directory of JSONL files to generate formatted prompts for LLMs."
    )
    parser.add_argument(
        "-i", "--input_dir",
        type=Path,
        required=True,
        help="Directory containing input .jsonl files",
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=Path,
        required=True,
        help="Directory where processed .jsonl files will be saved",
    )

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.is_dir():
        parser.error(f"Input path {input_dir} is not a directory.")

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        parser.error(f"No .jsonl files found in {input_dir}.")

    for input_path in files:
        output_path = output_dir / input_path.name
        print(f"Processing {input_path.name} -> {output_path.name}")
        process_file(input_path, output_path)


if __name__ == "__main__":
    main()