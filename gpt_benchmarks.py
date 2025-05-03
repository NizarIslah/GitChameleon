import os
import time
import json
import random
import pickle
import statistics
import openai
from openai import AzureOpenAI
from joblib import Parallel, delayed

import argparse
from pathlib import Path
from tqdm import tqdm

from pydantic import BaseModel

# 1) Define your schema as a Pydantic model
class MyResponse(BaseModel):
    answer: str
    explanation: str

parser = argparse.ArgumentParser(description="Arguments for GPT benchmarking")
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for reproducibility"
)
parser.add_argument(
    "--model",
    type=str,
    default="gpt-4o",
    help="Models available: gpt-4o-mini, gpt-4o, o1, o3-mini, o1-mini",
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
parser.add_argument("--struct_output", type=bool, default=False, help="Use structured output")
parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
parser.add_argument("--azure_endpoint", type=str, required=True, help="Azure endpoint")
parser.add_argument(
    "--azure_api_version",
    type=str,
    default="2024-05-01-preview",
    help="Azure API version",
)
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
    "--logprobs", type=bool, default=True, help="Whether to include logprobs"
)
parser.add_argument(
    "--feedback", type=bool, default=False, help="Whether to include feedback"
)
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
if args.model == "gpt-4o":
    args.max_tokens = 4096

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

client = AzureOpenAI(
    azure_endpoint=args.azure_endpoint,
    api_key=args.api_key,
    api_version=args.azure_api_version,
)


# Function to call OpenAI's chat completion with retry logic, including seed and temperature
def get_completion_with_retry(prompt, seed, args, max_retries=5, delay=10):
    retries = 0
    if args.model == "o1-mini":
        prompt = [
            {"role": "user", "content": prompt[0]["content"] + prompt[1]["content"]}
        ]
    while retries < max_retries:
        try:
            if args.struct_output:
                if args.model in ["o1", "o1-mini", "o3-mini"]:
                    response = client.beta.chat.completions.parse(
                        model=args.model, messages=prompt, seed=seed, response_format=MyResponse
                    )
                else:
                    response = client.beta.chat.completions.parse(
                        model=args.model,
                        messages=prompt,
                        seed=seed,
                        max_tokens=args.max_tokens,
                        top_p=args.top_p,
                        temperature=args.temperature,
                        logprobs=args.logprobs,
                        response_format=MyResponse,
                    )
            else:
                if args.model in ["o1", "o1-mini", "o3-mini"]:
                    response = client.chat.completions.create(
                        model=args.model, messages=prompt, seed=seed
                    )
                else:
                    response = client.chat.completions.create(
                        model=args.model,
                        messages=prompt,
                        seed=seed,
                        max_tokens=args.max_tokens,
                        top_p=args.top_p,
                        temperature=args.temperature,
                        logprobs=args.logprobs,
                    )
            # Check if the response is empty or None
            if response is None or not hasattr(response, "choices"):
                raise ValueError("Empty response received from the model.")

            return response
        except openai.RateLimitError as e:
            retries += 1
            print(
                f"Rate limit exceeded. Retrying {retries}/{max_retries} in {delay} seconds..."
            )
            time.sleep(delay)
        except Exception as e:
            return f"Error: {str(e)}"
    return f"Failed after {max_retries} retries."


num_samples = (
    1
    if args.temperature == 0
    or args.feedback
    or args.model in ["o1", "o1-mini", "o3-mini"]
    else 10
)

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
        content = (
            response.choices[0].message.parsed
            if not isinstance(response, str) # and not response.startswith("Error") # this would throw an error for pydantic because its not a string
            else response
        )
        explanation = content.explanation if hasattr(content, "explanation") else ""
        answer = content.answer if hasattr(content, "answer") else ""

        log_prob_mean = None
        log_prob_sum = None
        
        if not args.struct_output:
            answer = content
            explanation = ""
            log_probs = (
                response.choices[0].logprobs.get("content", [])
                if args.logprobs
                and args.model in ["gpt-4o", "gpt-4o-mini"]
                and not isinstance(response, str)
                else []
            )
            log_prob_mean = statistics.mean(log_probs) if log_probs else None
            log_prob_sum = sum(log_probs) if log_probs else None

        r_final.append(
            {
                "example_id": prompt["id"],
                "prompt": prompt["prompt"],
                "answer": answer,
                "explanation": explanation,
                "log_prob_mean": log_prob_mean,
                "log_prob_sum": log_prob_sum,
            }
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

    output_file = (
        Path(args.output_data)
        / f"responses_{args.temperature}_{args.struct_output}_{args.model}_{'feedback' if args.feedback else ''}_{'cot' if args.cot else ''}_{seed}.json"
    )

    with output_file.open("w", encoding="utf-8") as out:
        for record in r_final:
            line = json.dumps(record, ensure_ascii=False)
            out.write(line + "\n")
