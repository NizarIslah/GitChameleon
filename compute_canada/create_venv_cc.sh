#!/bin/bash
#SBATCH --job-name=make_venvs
#SBATCH --array=0-340:10             
#SBATCH --time=2:59:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --output=slurm-%A_%a.out
SLURM_ARRAY_TASK_ID=100
module load apptainer

# ────────────────
# 0) Config
# ────────────────
REPO_PATH=$HOME/GitChameleon
TARFILE="/home/raubyb/scratch/gc_1.0.tar.gz"
cp -r $REPO_PATH $SLURM_TMPDIR
WORKDIR="$SLURM_TMPDIR/GitChameleon"

export ENV_STORAGE_PATH="/home/raubyb/scratch/eval_venvs"
mkdir -p "$ENV_STORAGE_PATH"
# Compute the ID range [START .. END]
START=$SLURM_ARRAY_TASK_ID 
END=$(( START + 10 - 1 ))

cd "$WORKDIR"

# ────────────────
# 2) Build the SIF (once per node)
# ────────────────
if [ ! -f "$WORKDIR/gc_1.0.sif" ]; then
  cp "$TARFILE" "$SLURM_TMPDIR/"
  cd "$SLURM_TMPDIR"
  gunzip gc_1.0.tar.gz
  apptainer build gc_1.0.sif docker-archive://gc_1.0.tar
  mv gc_1.0.sif "$WORKDIR/"
  cd "$WORKDIR"
fi

# ────────────────
# 3) Inside container: build venvs for [START..END]
# ────────────────
apptainer exec \
  --bind "$WORKDIR:/app/repo" \
  --env PYENV_VERSION=3.10.14 \
  --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
  "$WORKDIR/gc_1.0.sif" \
  bash -lc "\
    cd /app/repo && \
    python -m venv bootstrap_venv && \
    source bootstrap_venv/bin/activate && \
    pip install -r requirements.txt && \
    python src/create_venvs.py \
      --dataset dataset/final_fix_dataset.jsonl \
      --base_path eval_venvs \
      --start ${START} \
      --end   ${END}"


# ────────────────────────────────────
# 4) Tar up each env and copy it back
# ────────────────────────────────────
for ID in $(seq $START $END); do
  # 1) Tar the env directory into one big file
  tar -czf "$WORKDIR/gcham_venv_${ID}.tar.gz" \
      -C "$WORKDIR/eval_venvs" "gcham_venv_${ID}"
  # 2) Copy the tarball back
  cp "$WORKDIR/gcham_venv_${ID}.tar.gz" \
     "$ENV_STORAGE_PATH/"
done