import os
import shutil
import wandb
from datetime import datetime

ENTITY     = "cl4code"
PROJECT    = "GC_EMNLP"
OUTPUT_DIR = "./jsonl_by_run"

api = wandb.Api()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# runs found:
print(f"Number of runs in {ENTITY}/{PROJECT}: {len(api.runs(f'{ENTITY}/{PROJECT}'))}")

for run in api.runs(f"{ENTITY}/{PROJECT}"):
    print(f"Processing run {run.id!r}…")
    run_dir = os.path.join(OUTPUT_DIR, run.id)
    os.makedirs(run_dir, exist_ok=True)

    # gather (timestamp, path) pairs
    candidates = []
    for art in run.logged_artifacts():
        # download the artifact into run_dir/artifact_name_version/
        art_dir = art.download(root=run_dir)
        # try to get the artifact's creation time
        try:
            # W&B artifact objects usually expose created_at as ISO string
            ts = datetime.fromisoformat(art.created_at.rstrip("Z"))
        except Exception:
            ts = None

        # scan for any .jsonl inside that artifact
        for root, _, files in os.walk(art_dir):
            for fn in files:
                if fn.endswith(".jsonl"):
                    path = os.path.join(root, fn)
                    # fallback to file mtime if we couldn't get art.created_at
                    mtime = datetime.fromtimestamp(os.path.getmtime(path))
                    timestamp = ts if ts and ts > datetime(1970,1,1) else mtime
                    candidates.append((timestamp, path))

    if not candidates:
        raise RuntimeError(f"Run {run.id}: no JSONL file found in any artifact")

    # pick the one with the latest timestamp
    latest_ts, latest_path = max(candidates, key=lambda x: x[0])
    dst = os.path.join(run_dir, f"{run.id}.jsonl")
    shutil.copy(latest_path, dst)
    print(f"  ✔ selected {os.path.basename(latest_path)} (ts={latest_ts}) → {dst}")

print("Done. One JSONL per run, chosen by most recent artifact.")    
