import os
import json
import wandb
from src.model import DecoderBase, make_model
from src.utils import get_prompt, write_jsonl, load_dataset, get_prompt_doc
from src.sanitize import sanitize
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
    dataset_path: str,
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
            dataset = load_dataset(dataset_path)
            prompt_key = "prompt"
        else:
            # datatype_jsonl data
            with open(dataset_path, "r") as f:
                dataset = [json.loads(line) for line in f]
                dataset = [(d.get("task_id", id), d) for id, d in enumerate(dataset)]
                prompt_key = "prompt" if "prompt" in dataset[0][1] else "content"
                assert prompt_key in dataset[0][1], f"Prompt key not found in {dataset[0]}"
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
                    existing_data[item["task_id"]] = (
                        existing_data.get(item["task_id"], 0) + 1
                    )

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
                prompt = task[prompt_key]
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

            if (
                (len(batch_prompts) == batch_size)
                or (id_num == len(dataset) - 1)
                or (id_range and id_num == id_range[1] - 1)
            ):
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
                for task_id, content, nsamples, task_outputs in zip(
                    batch_task_ids, batch_prompts, batch_nsamples, outputs
                ):
                    if model.is_direct_completion():
                        samples.extend(
                            [
                                dict(task_id=task_id, solution=content + completion)
                                for completion in task_outputs[:nsamples]
                            ]
                        )
                    else:
                        samples.extend(
                            [
                                dict(task_id=task_id, solution=completion)
                                for completion in task_outputs[:nsamples]
                            ]
                        )
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
