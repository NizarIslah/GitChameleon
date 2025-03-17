import json
from pathlib import Path
import argparse

def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)

def save_config(config_path, config):
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def get_generate_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-project", type=str, default="GitChameleon")
    parser.add_argument("--wandb-entity", type=str, default="cl4code")
    parser.add_argument("--disable-wandb", action="store_true", default=False)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument(
        "--dataset_path", default="dataset/combined_dataset.csv", type=str
    )
    parser.add_argument("--save_path", default=None, type=str)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--n_samples", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--strip_newlines", action="store_true")
    parser.add_argument("--datatype_jsonl", action="store_true")
    parser.add_argument("--feedback", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--id_range", nargs=2, type=int)
    parser.add_argument(
        "--backend",
        default="vllm",
        type=str,
        choices=["vllm", "hf", "openai", "mistral", "anthropic", "google"],
    )  # TODO: are these even implemented?
    parser.add_argument(
        "--base_url", default=None, type=str
    )  # TODO: is this even implemented?
    parser.add_argument("--tp", default=1, type=int)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--tokenizer_legacy", action="store_true")
    parser.add_argument("--tokenizer_name", default=None, type=str)

    args = parser.parse_args()
    return args

def get_evaluate_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default="")
    parser.add_argument('--instruct', default=False, action='store_true')
    parser.add_argument('--size', type=int, default=0)
    parser.add_argument('--dataset-path', type=str, default="dataset/combined_dataset.csv")
    parser.add_argument('--base-path', type=str, default="eval_venvs/")
    parser.add_argument('--enable-wandb', action='store_true', default=False)
    parser.add_argument('--wandb-project', type=str, default='GitChameleon_new')
    parser.add_argument('--wandb-entity', type=str, default='cl4code')
    parser.add_argument('--json-out-file', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--verbose-mode', action='store_true', default=False)
    parser.add_argument('--debug-mode', action='store_true', default=False)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--max_tokens', type=int, default=256)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--scratch', type=str, default="./")
    parser.add_argument('--output-path', type=str, default="results/starcoder2-15b-instruct-v0.1_temperature0.8.csv")
    parser.add_argument('--cot', action='store_true', default=False)
    parser.add_argument('--cot-output-path', type=str, default="cot_generations.jsonl")
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument("--library", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--k", type=int, default=1)  # for pass @ k evaluation
    parser.add_argument(
        "--n-generate", type=int, default=20
    )  # number of generations to generate
    parser.add_argument(
        "--n-jobs", type=int, default=-1
    )  # number of jobs to run in parallel
    args = parser.parse_args()
    return args