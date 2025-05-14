#!/bin/bash
#SBATCH --job-name=jsonl_eval
#SBATCH --output=logs_eval/jsonl_eval_%A_%a.out
#SBATCH --error=logs_eval/jsonl_eval_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition=long
set -euo pipefail
# Load modules
module load python/3.10

# 1) Read the JSONL path for *this* array index
JSONL_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" missing.txt)
echo "[$SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID] Working on $JSONL_FILE"

# Thread-affinity
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Paths
REPO_DIR="$SCRATCH/GitChameleon/GitChameleon"
SIF_SRC="$SCRATCH/GitChameleon/gc_1.0.sif"
CACHE_DIR="$SCRATCH/$USER/apptainer_cache/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
TMP_DIR="$SCRATCH/$USER/apptainer_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export WANDB_CACHE_DIR="$REPO_DIR/wandb_cache/$SLURM_JOB_ID"
export XDG_CACHE_HOME="$WANDB_CACHE_DIR"
mkdir -p "$WANDB_CACHE_DIR"

JSONL_DIR="$(dirname "$JSONL_FILE")"
CSV_FILE="${JSONL_DIR}/$(basename "${JSONL_FILE%.jsonl}.csv")"

# Prepare directories
mkdir -p "$CACHE_DIR" "$TMP_DIR"

# for verification / coverage run this (interactive)
# apptainer exec \
#     --bind "$PWD:/app/repo" \
#     --env PYENV_VERSION=3.10.14 \
#     --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
#     ../gc_1.0.sif \
#     bash -lc "\
#       cd /app/repo && \
#       python -m venv eval_main_venv && \
#       source eval_main_venv/bin/activate && \
#       python verify_dataset.py \
#         dataset/final_fix_dataset.jsonl \
#         eval_venvs \
#         dataset/solutions/tests \
#         --cov"

# for eval interactive run this
# Execute evaluation inside the container
# apptainer exec \
#     --bind "$PWD:/app/repo" \
#     --env PYENV_VERSION=3.10.14 \
#     --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
#     ../gc_1.0.sif \
#     bash -lc "\
#       cd /app/repo && \
#       python -m venv eval_main_venv && \
#       source eval_main_venv/bin/activate && \
#       export WANDB_CACHE_DIR=/app/repo/wandb_cache && \
#       python parallel_eval_jsonl.py \
#         dataset/final_fix_dataset.jsonl \
#         rag_gpt_4o.jsonl \
#         eval_venvs \
#         dataset/solutions/tests" \
#         --wandb"

apptainer exec \
  --bind "$REPO_DIR:/app/repo" \
  --bind "$CACHE_DIR:/var/apptainer/cache" \
  --bind "$TMP_DIR:/tmp" \
  "$SIF_SRC" \
  bash -lc "
  cd /app/repo && \
  python -m venv eval_main_venv && \
  source eval_main_venv/bin/activate && \
  export WANDB_CACHE_DIR=/app/repo/wandb_cache/$SLURM_JOB_ID && \
  export XDG_CACHE_HOME=/app/repo/wandb_cache/$SLURM_JOB_ID && \
  python parallel_eval_jsonl.py \
    dataset/final_fix_dataset.jsonl \
    "${JSONL_FILE}" \
    eval_venvs \
    dataset/solutions/tests \
    --wandb"

# Copy back the resulting CSV to the same directory as the JSONL
if [[ -f "$CSV_FILE" ]]; then
  mv "$CSV_FILE" "$JSONL_DIR/"
  echo "[${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID}] Moved $(basename "$CSV_FILE") to $JSONL_DIR"
else
  echo "[${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID}] Warning: output CSV not found: $CSV_FILE" >&2
fi

# Optional cleanup
rm -rf "$CACHE_DIR" "$TMP_DIR"
