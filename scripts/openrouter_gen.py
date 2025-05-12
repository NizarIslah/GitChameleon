import time
from joblib import Parallel, delayed
from openai import OpenAI
import openai
import tqdm
import json
import argparse
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from pydantic import BaseModel
from together import Together
import os

# 1) Define your schema as a Pydantic model
class MyResponse(BaseModel):
    answer: str
    explanation: str
        
class Step(BaseModel):
    explanation: str
    output: str

class CodeReasoning(BaseModel):
    steps: list[Step]
    answer: str

# argparse for CoT(boolean), temperature (float, default=0), model (string)
import argparse
parser = argparse.ArgumentParser(description="Script for running OpenAI completions with CoT, temperature, and model options.")
parser.add_argument("--cot", action="store_true", help="Enable Chain of Thought (CoT) reasoning.")
parser.add_argument("--temperature", type=float, default=0.0, help="Set the temperature for the model.")
parser.add_argument("--model", type=str, default="x-ai/grok-3-beta", help="Specify the model to use.")
parser.add_argument("--wandb", action="store_true", help="Enable Weights and Biases logging.")
parser.add_argument("--test", action="store_true", help="Enable test mode for debugging.")
parser.add_argument("--together", action="store_true", help="Use Together API instead of OpenAI.")
parser.add_argument("--self_debug_file", type=str, default="", help="File to save self-debugging information.")
parser.add_argument("--non_struct", action="store_true", help="Use non-structured output.")
args = parser.parse_args()

cot = args.cot
temperature = args.temperature
model = args.model
# extra_prompt = "Only answer in JSON." if args.together else ""
extra_prompt = ""


if args.self_debug_file != "":
    in_file = args.self_debug_file
else:
    in_file = "dataset/final_fix_dataset.jsonl"
with open(in_file) as f:
    prompts = [
        {
            "id": data.get("example_id", str(i)),
            "prompt": data.get("cot_messages" if cot else "messages"),
        }
        for i, line in enumerate(f)
        for data in [json.loads(line)]
    ]

if args.together:
    for prompt in prompts:
        prompt["prompt"][0]["content"] = prompt["prompt"][0]["content"] + extra_prompt
    os.environ["TOGETHER_API_KEY"] = "tgp_v1_a0V28hncTnzyYorKFt0cXPZrK0ium_r0FDAQzyA9Bzo" 
    client = Together()
else:
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
    )
print(client)

def get_completion_together_with_retry(prompt, max_retries=5, delay=10):
    retries = 0
    while retries < max_retries:
        try:
            if cot:
                response = client.chat.completions.create(model=model, messages=prompt, seed=16, temperature=temperature, response_format={"type": "json_object", "schema": CodeReasoning.model_json_schema()})
            else:
                response = client.chat.completions.create(model=model, messages=prompt, seed=16, temperature=temperature, response_format={"type": "json_object", "schema": MyResponse.model_json_schema()})
            return response
        except openai.RateLimitError as e:
            retries += 1
            print(f"Rate limit exceeded. Retrying {retries}/{max_retries} in {delay} seconds...")
            time.sleep(delay)
        except Exception as e:
            return f"Error: {str(e)}"
    return f"Failed after {max_retries} retries."


def get_completion_together_with_retry_nonstruct(prompt, max_retries=5, delay=10):
    retries = 0
    while retries < max_retries:
        try:
            if cot:
                response = client.chat.completions.create(model=model, messages=prompt, seed=16, temperature=temperature)
            else:
                response = client.chat.completions.create(model=model, messages=prompt, seed=16, temperature=temperature)
            return response
        except openai.RateLimitError as e:
            retries += 1
            print(f"Rate limit exceeded. Retrying {retries}/{max_retries} in {delay} seconds...")
            time.sleep(delay)
        except Exception as e:
            return f"Error: {str(e)}"
    return f"Failed after {max_retries} retries."

# Function to call OpenAI's chat completion with retry logic, including seed and temperature
def get_completion_with_retry(prompt, max_retries=5, delay=10):
    retries = 0
    while retries < max_retries:
        try:
            if cot:
                response = client.beta.chat.completions.parse(model=model, messages=prompt, seed=16, temperature=temperature, response_format=CodeReasoning)
            else:
                response = client.beta.chat.completions.parse(model=model, messages=prompt, seed=16, temperature=temperature, response_format=MyResponse)
            return response
        except openai.RateLimitError as e:
            retries += 1
            print(f"Rate limit exceeded. Retrying {retries}/{max_retries} in {delay} seconds...")
            time.sleep(delay)
        except Exception as e:
            return f"Error: {str(e)}"
    return f"Failed after {max_retries} retries."

r_final = []

if args.test:
    # For testing, limit the number of prompts to 2
    print("Running in test mode. Limiting to 2 prompts.")
    prompts = prompts[:2]

get_completion_fn = get_completion_together_with_retry if args.together else get_completion_with_retry
if args.non_struct:
    get_completion_fn = get_completion_together_with_retry_nonstruct if args.together else get_completion_with_retry

with tqdm_joblib(tqdm(desc="Fetching completions", total=len(prompts))) as progress_bar:
    responses = Parallel(n_jobs=-1, prefer="threads")(
        delayed(get_completion_fn)(prompt["prompt"])
        for prompt in prompts
    )
print('All done')

import json
failed_count = 0
for prompt, response in zip(prompts, responses):
    try:
        if args.test:
            print(f"Processing response for prompt {prompt['id']}, response: {response}")

        if isinstance(response, str):
            content = response
            print(content)
        else:
            message = response.choices[0].message
            print(message)
            
            if args.together:
                if args.non_struct:
                    content = message.content
                else:
                    content = json.loads(message.content)
                    explanation = content.get("explanation", "")
                    answer = content.get("answer", "")
            else:
                content = message.parsed
                explanation = getattr(content, "explanation", "")
                answer = getattr(content, "answer", "")

        if cot:
            steps = getattr(content, "steps", [])
            steps = [vars(step) for step in steps]

        if cot:
            r_final.append(
                {
                    "example_id": prompt["id"],
                    "prompt": prompt["prompt"],
                    "answer": answer,
                    "steps": steps,
                }
            )
        else:
            if args.non_struct:
                r_final.append(
                    {
                        "example_id": prompt["id"],
                        "prompt": prompt["prompt"],
                        "output": content,
                    }
                )
            else:
                r_final.append(
                    {
                        "example_id": prompt["id"],
                        "prompt": prompt["prompt"],
                        "answer": answer,
                        "explanation": explanation,
                    }
            )
    except json.JSONDecodeError as e:
        failed_count += 1
        print(f"JSON decode error for prompt {prompt['id']}: {e}")
        r_final.append(
            {
                "example_id": prompt["id"],
                "prompt": prompt["prompt"],
                "answer": "",
                "explanation": "Error decoding JSON",
                "error": str(e),
            }
        )
    except AttributeError as e:
        print(f"Error processing response for prompt {prompt['id']}: {e}")
        r_final.append(
            {
                "example_id": prompt["id"],
                "prompt": prompt["prompt"],
                "error": str(e),
            }
        )
    except Exception as e:
        print(f"Unexpected error for prompt {prompt['id']}: {e}")
        r_final.append(
            {
                "example_id": prompt["id"],
                "prompt": prompt["prompt"],
                "error": str(e),
            }
        )

model_path = model.split("/")[-1]
        
from pathlib import Path
output_file = Path(f'final_gpt/{model_path}_cot.jsonl' if cot else f'final_gpt/{model_path}_t0.jsonl')
    
with output_file.open("w", encoding="utf-8") as out:
    for record in r_final:
        line = json.dumps(record, ensure_ascii=False)
        out.write(line + "\n")


# Log the JSONL file as a WandB artifact

if args.wandb:
    import wandb

    name = f'{model_path}_cot' if cot else f'{model_path}_t0'
    name = name + '_self_debug' if args.self_debug_file != "" else name


    wandb.init(
        project="GC_EMNLP",
        entity="cl4code",
        name=name,
        config={
            "model": model_path,
            "temperature": temperature,
        },
    )
        
    artifact = wandb.Artifact(
        name=f'responses_cot_struct_{model_path}' if cot else f'responses_t0_struct_{model_path}',
        type="responses",
    )
    artifact.add_file(str(output_file))
    wandb.log_artifact(artifact)
    print(f"JSON Failed count: {failed_count}")
    wandb.log({"json_failed_count": failed_count})


    try:
        if cot:
            table = wandb.Table(columns=["example_id", "prompt", "answer", "steps"])
            for record in r_final:
                table.add_data(
                    record["example_id"],
                    record["prompt"],
                    record["answer"],
                    record["steps"]
                )
        else:
            table = wandb.Table(columns=["example_id", "prompt", "answer", "explanation"])
            for record in r_final:
                table.add_data(
                    record["example_id"],
                    record["prompt"],
                    record["answer"],
                    record["explanation"]
                )

        wandb.log({"responses_table": table})
    except Exception as e:
        print(f"Error logging table to WandB: {e}")
