import argparse
import json

import tiktoken
from transformers import AutoTokenizer


def tokenize_sample(sample, tokenizer):
    """
    Concatenates the starting code and solution,
    tokenizes the combined text, and stores the results in the sample.
    """
    # Adjust these keys if your file uses different names.
    starting_code = sample.get("starting_code", "")
    solution = sample.get("solution", "")
    combined_text = "``` python \n" + starting_code + solution + "\n ```"
    # print(combined_text)
    # Tokenize the text; change add_special_tokens=True if needed.
    token_ids = tokenizer.encode(combined_text)

    # Save the token IDs and token count in the sample
    sample["token_ids"] = token_ids
    sample["token_count"] = len(
        token_ids
    )  # Add 100 to token count to account for the comments
    sample["combined_text"] = combined_text  # Save the combined text
    return sample


def process_file(input_filepath, output_filepath, tokenizer):
    max_sample = None
    max_token_count = -1
    count_above_256 = 0

    with open(input_filepath, "r", encoding="utf-8") as infile, open(
        output_filepath, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON: {line}")
                continue  # Skip lines that are not valid JSON.

            # Process the sample with the tokenizer.
            sample = tokenize_sample(sample, tokenizer)
            token_count = sample["token_count"]
            if int(sample["example_id"]):
                print("sample ", sample["example_id"], " token count ", token_count)
            # Track the sample with the maximum token count.
            if token_count > max_token_count:
                max_token_count = token_count
                max_sample = sample

            # Count samples with more than 256 tokens.
            if token_count > 256:
                count_above_256 += 1

            # Write the updated sample to the output file.
            outfile.write(json.dumps(sample) + "\n")

    return max_sample, max_token_count, count_above_256


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Tokenize samples from a JSONL file.")
    parser.add_argument(
        "input_filepath", type=str, help="Path to the input JSONL file."
    )
    parser.add_argument(
        "output_filepath", type=str, help="Path to the output JSONL file."
    )
    args = parser.parse_args()

    input_filepath = args.input_filepath
    output_filepath = args.output_filepath

    # Load the tokenizer from Hugging Face (here we use GPT-2's tokenizer).
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Alternatively, use tiktoken for OpenAI's tokenizer.
    tokenizer = encoding = tiktoken.get_encoding("cl100k_base")

    max_sample, max_token_count, count_above_256 = process_file(
        input_filepath, output_filepath, tokenizer
    )
