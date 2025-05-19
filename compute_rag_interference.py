import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, DefaultDict
from collections import defaultdict
import pandas as pd

# ----------------------------- Utils -------------------------------------


def normalize_url(url: str) -> str:
    """Normalize doc URLs (case‑insensitive, trim trailing slash)."""
    return url.rstrip("/").lower()


# ------------------------- Ground‑truth loader ---------------------------


def load_dataset(dataset_path: Path) -> Dict[str, Set[str]]:
    """Return mapping example_id -> set of ground‑truth doc URLs."""
    gt: Dict[str, Set[str]] = {}
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            example_id = str(obj["example_id"])
            doc_key = "docs" if "docs" in obj else "used_docs"
            gt[example_id] = {normalize_url(u) for u in obj[doc_key]}
    return gt


# --------------------------- Per‑file eval -------------------------------


def evaluate_file(gen_path: Path, gt_docs: Dict[str, Set[str]]) -> Tuple[int, int, int]:
    """Return (n_examples, hit_count, strict_precision_count)."""
    n_examples = hit_count = strict_prec_count = 0
    with gen_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "example_id" not in obj or "used_docs" not in obj:
                continue  # skip malformed rows
            example_id = str(obj["example_id"])
            if example_id not in gt_docs:
                continue  # skip rows not in ground truth
            retrieved = {normalize_url(u) for u in obj.get("used_docs", [])}
            gold = gt_docs[example_id]
            n_examples += 1
            if gold & retrieved:
                hit_count += 1
            if retrieved and retrieved <= gold:
                strict_prec_count += 1
    return n_examples, hit_count, strict_prec_count


# ------------------------------ Main -------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate doc retrieval precision/recall of RAG generations, outputting a pandas DataFrame with separate K columns."
    )
    parser.add_argument(
        "--dataset",
        default=Path("dataset/final_fix_dataset.jsonl"),
        type=Path,
        help="Path to ground‑truth dataset JSONL file.",
    )
    parser.add_argument(
        "--generation_dir",
        default=Path("RAG_generation"),
        type=Path,
        help="Directory containing model output JSONL files.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default="rag_metrics.csv",
        help="Optional path to save the DataFrame as CSV.",
    )
    args = parser.parse_args()

    if not args.generation_dir.exists():
        raise FileNotFoundError(
            f"Generation directory '{args.generation_dir}' does not exist."
        )

    jsonl_files = list(args.generation_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in '{args.generation_dir}'.")

    gt_docs = load_dataset(args.dataset)

    metrics: DefaultDict[str, Dict[str, Tuple[int, int, int]]] = defaultdict(dict)

    for file in jsonl_files:
        stem = file.stem  # e.g. rag_commanda_k1
        if "_k" not in stem:
            continue  # skip unexpected names
        model_part, k_part = stem.rsplit("_k", 1)
        model_name = model_part.replace("rag_", "")
        k_val = f"k{k_part}"
        metrics[model_name][k_val] = evaluate_file(file, gt_docs)

    # -------------------- Build DataFrame -------------------------------
    records: List[Dict[str, object]] = []

    def unpack(vals: Tuple[int, int, int] | None):
        if not vals:
            return (None, None, None)
        n, hits, precs = vals
        hit_rate = hits / n * 100 if n else 0.0
        prec_rate = precs / n * 100 if n else 0.0
        return (n, hit_rate, prec_rate)

    for model, row in metrics.items():
        n1, hit1, prec1 = unpack(row.get("k1"))
        n3, hit3, prec3 = unpack(row.get("k3"))
        records.append(
            {
                "model": model,
                "N1": n1,
                "Hit1%": hit1,
                "Prec1%": prec1,
                "N3": n3,
                "Hit3%": hit3,
                "Prec3%": prec3,
            }
        )

    df = pd.DataFrame(records)
    # Keep column order
    df = df[["model", "N1", "Hit1%", "Prec1%", "N3", "Hit3%", "Prec3%"]]

    # Print nicely
    print("\nRetrieval metrics (pandas DataFrame):\n")
    print(df.to_string(index=False, justify="left"))

    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved CSV to {args.output_csv}\n")


if __name__ == "__main__":
    main()
