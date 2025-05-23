import wandb
from configs import get_generate_args
from src.model import make_model
from src.generate_code import codegen


def main():
    args = get_generate_args()

    # wandb
    if not args.disable_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.model,
            config=vars(args),
        )

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
        cot=args.cot,
    )

    if not args.save_path:
        save_path = (
            args.model.replace("/", "--")
            + f"--gitchameleon--{args.backend}-{args.temperature}-{args.n_samples}"
        )
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
        dataset_path=args.dataset_path,
        cot=args.cot,
        greedy=args.greedy,
        strip_newlines=args.strip_newlines,
        n_samples=args.n_samples,
        resume=args.resume,
        id_range=args.id_range,
        batch_size=args.bs,
        args=args,
    )


if __name__ == "__main__":
    main()
