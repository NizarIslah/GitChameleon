import os
import time
import json
import random
import pickle
import statistics
import anthropic
from joblib import Parallel, delayed

import argparse
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Arguments for Anthropic (Claude suite) benchmarking"
)
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for reproducibility"
)
parser.add_argument(
    "--model",
    type=str,
    default="claude-3-7-sonnet-20250219",
    help="Models available: claude-3-7-sonnet-20250219, claude-3-5-haiku-20241022",
)
parser.add_argument(
    "--input_data",
    type=str,
    required=True,
    default="dataset.jsonl",
    help="Path to input data",
)
parser.add_argument(
    "--output_data",
    type=str,
    required=True,
    default="output/",
    help="Path to output data",
)
parser.add_argument(
    "--top_p", type=float, default=0.95, help="Top-p sampling parameter"
)
parser.add_argument(
    "--temperature", type=float, default=0.8, help="Temperature parameter"
)
parser.add_argument(
    "--max_tokens",
    type=int,
    default=4096,
    help="Maximum tokens for the model. (4096 for baseline, 6000 for CoT)",
)
parser.add_argument("--api_key", type=str, required=True, help="Anthropic API key")

parser.add_argument("--wandb", type=bool, default=False, help="Use WandB for logging")
parser.add_argument(
    "--wandb_entity", type=str, help="WandB entity name"
)
parser.add_argument("--wandb_key", type=str, help="WandB API key")
parser.add_argument(
    "--wandb_project", type=str, help="WandB project name"
)
parser.add_argument(
    "--wandb_run_name", type=str, help="WandB run name"
)

parser.add_argument(
    "--system_prompt",
    type=bool,
    default=False,
    help="Include a system prompt to set the role of the model",
)
parser.add_argument(
    "--thinking_mode",
    type=bool,
    default=False,
    help="Enable thinking mode for anthropic models",
)

parser.add_argument("--feedback", help="Whether to include feedback")
parser.add_argument(
    "--cot", type=bool, default=False, help="Whether to include chain of thought"
)
args = parser.parse_args()

# Set the random seed for reproducibility
random.seed(args.seed)

# check if the temp is 0 when feedback is true and set to 0
if args.feedback and args.temperature != 0:
    print("Temperature set to 0 when feedback is true")
    args.temperature = 0

# Adjust max tokens based on cot and model
args.max_tokens = 6000 if args.cot else 4096

# Load the data from the JSONL file
with open(args.input_data) as f:
    prompts = [
        {
            "id": data.get("example_id", str(i)),
            "prompt": data.get("cot_messages" if args.cot else "messages"),
        }
        for i, line in enumerate(f)
        for data in [json.loads(line)]
    ]

client = anthropic.Anthropic(
    api_key=args.api_key,
)


# Function to call Anthropic's chat completion with retry logic, including seed and temperature
def get_completion_with_retry(prompt, seed, args, max_retries=5, delay=10):
    retries = 0
    message = prompt[0]["content"] + prompt[1]["content"]
    message = message.split(".", 1)[-1].strip()

    while retries < max_retries:
        try:
            if args.system_prompt and args.thinking_mode:
                response = client.messages.create(
                    model=args.model,
                    system="You are a seasoned python developer and data scientist at a Fortune 500 company. Your goal is to provide clean, correct and efficient solutions to coding problems.",
                    thinking={
                        "type": "enabled",
                        "budget_tokens": int(0.6 * args.max_tokens),
                    },
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    messages=[{"role": "user", "content": message}],
                )
            elif args.system_prompt and not args.thinking_mode:
                response = client.messages.create(
                    model=args.model,
                    system="You are a seasoned python developer and data scientist at a Fortune 500 company. Your goal is to provide clean, correct and efficient solutions to coding problems.",
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    messages=[{"role": "user", "content": message}],
                )
            elif args.thinking_mode and not args.system_prompt:
                response = client.messages.create(
                    model=args.model,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": int(0.6 * args.max_tokens),
                    },
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    messages=[{"role": "user", "content": message}],
                )
            else:
                response = client.messages.create(
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    messages=[{"role": "user", "content": message}],
                )
            return response
        except anthropic.RateLimitError as e:
            retries += 1
            print(
                f"Rate limit exceeded. Retrying {retries}/{max_retries} in {delay} seconds..."
            )
            time.sleep(delay)
        except Exception as e:
            return f"Error: {str(e)}"
    return f"Failed after {max_retries} retries."


num_samples = 1 if args.temperature == 0 or args.feedback else 10

# Use joblib to run the requests in parallel
# Ensure output directory exists
Path(args.output_data).mkdir(parents=True, exist_ok=True)

for seed in tqdm(random.sample(range(1, 1000), num_samples), desc="Processing seeds"):
    responses = Parallel(n_jobs=-1, prefer="threads")(
        delayed(get_completion_with_retry)(prompt["prompt"], seed, args)
        for prompt in prompts
    )

    r_final = []
    for prompt, response in zip(prompts, responses):
        if isinstance(response, str) or response.startswith("Error"):
            content = response
            thinking = None
        else:
            if args.thinking_mode:
                thinking = getattr(response.content[0], "thinking", None)
                content = getattr(response.content[1], "text", None)
            else:
                thinking = None
                content = getattr(response.content[0], "text", None)

        r_final.append(
            {
                "example_id": prompt["id"],
                "prompt": prompt["prompt"],
                "response": content,
                "thinking": thinking if args.thinking_mode else None,
            }
        )

    output_file = (
        Path(args.output_data)
        / f"responses_{args.temperature}_{args.model}_{'feedback' if args.feedback else ''}_{'cot' if args.cot else ''}_{'sys' if args.system_prompt else ''}_{'thinking' if args.thinking_mode else ''}_{seed}.json"
    )

    if args.wandb:
        import wandb
        os.environ["WANDB_API_KEY"] = args.wandb_key

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "model": args.model,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "seed": seed,
            },
        )
        for arg in vars(args):
            wandb.config[arg] = getattr(args, arg)
        wandb.log({"responses": r_final})

    with output_file.open("w", encoding="utf-8") as out:
        for record in r_final:
            line = json.dumps(record, ensure_ascii=False)
            out.write(line + "\n")
