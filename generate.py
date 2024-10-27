import os
import json
import argparse
import math
import wandb

from model import DecoderBase, make_model
from utils import get_prompt, write_jsonl, load_dataset, get_prompt_doc
from sanitize import sanitize
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

def codegen(
    model: DecoderBase,
    save_path: str,
    data_path: str,
    cot=False,
    greedy=False,
    strip_newlines=False,
    n_samples=1,
    id_range=None,
    resume=True,
    batch_size=1,  # New parameter
    args=None,
):
    with Progress(
        TextColumn(f"GitChameleon •" + "[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    ) as p:
            
        if not args.datatype_jsonl:
            dataset = load_dataset(data_path)
        else:
            # datatype_jsonl data
            with open(data_path, "r") as f:
                dataset = [json.loads(line) for line in f]
                dataset = [(d["task_id"], d) for d in dataset]

        # create save_path if it doesn't exist, e.g., a/b.jsonl
        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname) and dirname != "":
            os.makedirs(dirname)

        batch_prompts = []
        batch_task_ids = []
        batch_nsamples = []

        # Read existing data once if resuming
        existing_data = {}
        if resume and os.path.exists(save_path):
            with open(save_path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    existing_data[item["task_id"]] = existing_data.get(item["task_id"], 0) + 1
        
        # from dataset
        if isinstance(dataset, dict):
            dataset_ = dataset.items()
        else:
            dataset_ = dataset
        for id_num, (task_id, task) in enumerate(p.track(dataset_)):
            if id_range is not None:
                low, high = id_range
                if id_num < low:
                    p.console.print(f"Skipping {task_id} as it is not in {id_range}")
                    continue
                if id_num > id_range[1]:
                    break

            p_name = task_id

            n_existing = existing_data.get(task_id, 0)
            nsamples = n_samples - n_existing

            if args.datatype_jsonl:
                prompt = task["prompt"]
            elif args.oracle:
                prompt = get_prompt_doc(task, not model.is_direct_completion())
            else:
                prompt = get_prompt(task, not model.is_direct_completion(), cot)
            if strip_newlines:
                prompt = prompt.strip("\n")
            if nsamples > 0:
                batch_prompts.append(prompt)
                batch_task_ids.append(task_id)
                batch_nsamples.append(nsamples)
                
                log = f"Codegen: {p_name} @ {model}"
                if n_existing > 0:
                    log += f" (resuming from {n_existing})"
                p.console.print(log)
            
            if (len(batch_prompts) == batch_size) or (id_num == len(dataset) - 1) or (id_range and id_num == id_range[1] - 1):
                if not batch_prompts:
                    break
                outputs = model.codegen(
                    batch_prompts,
                    do_sample=not greedy,
                    num_samples=max(batch_nsamples),
                )
                assert outputs, "No outputs from model!"

                try:
                    outputs = [[sanitize(x) for x in output] for output in outputs]
                except Exception as e:
                    print("Could not sanitize outputs:", e)
                
                samples = []
                for task_id, content, nsamples, task_outputs in zip(batch_task_ids, batch_prompts, batch_nsamples, outputs):
                    if model.is_direct_completion():
                        samples.extend([
                            dict(task_id=task_id, solution=content+completion)
                            for completion in task_outputs[:nsamples]
                        ])
                    else:
                        samples.extend([
                            dict(task_id=task_id, solution=completion)
                            for completion in task_outputs[:nsamples]
                        ])
                print(f"Generated {len(samples)} samples")
                write_jsonl(save_path, samples, append=True)
            
                # Clear batches
                batch_prompts = []
                batch_task_ids = []
                batch_nsamples = []

    # log to wandb
    if not args.disable_wandb:
        wandb.save(save_path)
        wandb.log({"generated": len(samples)})
        wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb-project', type=str, default='GitChameleon')
    parser.add_argument('--wandb-entity', type=str, default='cl4code')
    parser.add_argument('--disable-wandb', action='store_true', default=False)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--data_path", default="data/combined_dataset.csv", type=str)
    parser.add_argument("--save_path", default=None, type=str)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--n_samples", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--strip_newlines", action="store_true")
    parser.add_argument('--datatype_jsonl', action='store_true')
    parser.add_argument('--feedback', action='store_true')
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--id_range", nargs=2, type=int)
    parser.add_argument("--backend", default="vllm", type=str, choices=["vllm", "hf", "openai", "mistral", "anthropic", "google"])
    parser.add_argument("--base_url", default=None, type=str)
    parser.add_argument("--tp", default=1, type=int)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--tokenizer_legacy", action="store_true")
    parser.add_argument("--tokenizer_name", default=None, type=str)

    args = parser.parse_args()

    # wandb
    if not args.disable_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.model, config=vars(args))
        

    if args.greedy or (args.temperature == 0 and args.n_samples == 1):
        args.temperature = 0
        args.n_samples = 1
        args.greedy = True
        print("Greedy decoding ON (--greedy): setting bs=1, n_samples=1, temperature=0")

    if args.id_range is not None:
        assert len(args.id_range) == 2, "id_range must be a list of length 2"
        assert args.id_range[0] < args.id_range[1], "id_range must be increasing"
        args.id_range = tuple(args.id_range)

    # Make dir for codes generated by each model
    model_runner = make_model(
        model=args.model,
        backend=args.backend,
        temperature=args.temperature,
        base_url=args.base_url,
        tp=args.tp,
        trust_remote_code=args.trust_remote_code,
        tokenizer_name=args.tokenizer_name,
        tokenizer_legacy=args.tokenizer_legacy,
        cot=args.cot
    )
    
    if not args.save_path:
        save_path = args.model.replace("/", "--") + f"--gitchameleon--{args.backend}-{args.temperature}-{args.n_samples}"
        if args.cot:
            save_path += f"--cot"
        if args.feedback:
            save_path += f"--feedback"
        save_path += ".jsonl"
    else:
        save_path = args.save_path

    codegen(
        model=model_runner,
        save_path=save_path,
        data_path=args.data_path,
        cot=args.cot,
        greedy=args.greedy,
        strip_newlines=args.strip_newlines,
        n_samples=args.n_samples,
        resume=args.resume,
        id_range=args.id_range,
        batch_size=args.bs,
        args=args
    )


if __name__ == "__main__":
    main()
