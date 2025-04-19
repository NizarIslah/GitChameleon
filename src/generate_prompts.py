import json
from pathlib import Path
import argparse

SYS_PROMPT = """You are a skilled Python programmer tasked with solving a coding problem. Your goal is to provide a clear, efficient, and correct solution that meets all the specified requirements.

Please provide your solution following these guidelines:

1. Use the required library in your solution.
2. Incorporate the provided starter code correctly.
3. Write your solution in Python.
4. Format your solution within a markdown code block.
5. Ensure your code is clean, efficient, and well-commented.
6. Output only the code block and nothing else.

Example output format:

```python
# [Your code here, incorporating the starter code]

# [Additional code and comments as needed]
```

After writing your solution, please review it to ensure all requirements are met and the code is correct and efficient.

Here are the key elements for this task: """

COT_SYS_PROMPT = """You are a skilled Python programmer tasked with solving a coding problem. Your goal is to provide a clear, efficient, and correct solution that meets all the specified requirements.

First, let's think step-by-step. Then, please provide your solution following these guidelines:

1. Use the required library in your solution.
2. Incorporate the provided starter code correctly.
3. Write your solution in Python.
4. Format your solution within a markdown code block.
5. Ensure your code is clean, efficient, and well-commented.
6. Output nothing else after the code block.


Example output format:

[Step-by-step thinking]
```python
# [Your code here, incorporating the starter code]

# [Additional code and comments as needed]
```

After writing your solution, please review it to ensure all requirements are met and the code is correct and efficient.

Here are the key elements for this task: """

PROMPT_TEMPLATE = f"""1. Required Library:
<library>
{{library}}
</library>

2. Python version:
<python>
{{python_version}}
</python>

2. Coding Problem:
<coding_problem>
{{coding_problem}}
</coding_problem>

3. Starter Code:
<starter_code>
{{starter_code}}
</starter_code>"""


def main(input_path: Path, output_path: Path):
    with open(input_path, "r") as fin:
        inputs = [json.loads(line) for line in fin]

    for line in inputs:
        python_version = line["python_version"]

        line["messages"] = [
            {"role": "system", "content": SYS_PROMPT},
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(
                    library=line["library"] + "==" + line["version"],
                    python_version=python_version,
                    coding_problem=line["problem"],
                    starter_code=line["starting_code"],
                ),
            },
        ]

        line["cot_messages"] = [
            {"role": "system", "content": COT_SYS_PROMPT},
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(
                    library=line["library"] + "==" + line["version"],
                    python_version=python_version,
                    coding_problem=line["problem"],
                    starter_code=line["starting_code"],
                ),
            },
        ]

    with open(output_path, "w") as fout:
        for line in inputs:
            fout.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process coding problems from a JSONL file to generate formatted prompts for LLMs."
    )

    # Add argument for input file path
    parser.add_argument(
        "-i",
        "--input",
        type=Path,  # Use Path object directly
        required=True,
        help="Path to the input JSONL file containing coding problems.",
        metavar="INPUT_PATH",
    )

    # Add argument for output file path
    parser.add_argument(
        "-o",
        "--output",
        type=Path,  # Use Path object directly
        required=True,
        help="Path to the output JSONL file where processed prompts will be saved.",
        metavar="OUTPUT_PATH",
    )

    # Parse the command-line arguments
    args = parser.parse_args()
    # --- End of CLI Parser ---

    # Call the main function with the parsed arguments
    main(input_path=args.input, output_path=args.output)
