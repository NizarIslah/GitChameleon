#!/bin/bash
#SBATCH --job-name=verify_envs
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=verify-%j.out

REPO_PATH=$HOME/GitChameleon
TARFILE="/home/raubyb/scratch/gc_1.0.tar.gz"
export ENV_STORAGE_PATH="/home/raubyb/scratch/eval_venvs"


module load apptainer

rm -r $HOME/.apptainer/cache
mkdir -p $SLURM_TMPDIR/.apptainer
export APPTAINER_CACHEDIR=$SLURM_TMPDIR/.apptainer



# ────────────────
# Copy the repo
# ────────────────
cp -r $REPO_PATH $SLURM_TMPDIR
WORKDIR="$SLURM_TMPDIR/GitChameleon"
# ────────────────

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


# 1) Copy your tarballs into fast local disk
cp $ENV_STORAGE_PATH/gcham_venv_*.tar.gz "$WORKDIR/eval_venvs/"

# 2) Unpack each one
for t in gcham_venv_*.tar.gz; do
  tar -xzf "$t"
done

apptainer exec \
    --bind "$WORKDIR:/app/repo" \
    --env PYENV_VERSION=3.10.14 \
    --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    gc_1.0.sif \
    bash -lc "\
      cd /app/repo && \
      python -m venv eval_main_venv && \
      source eval_main_venv/bin/activate && \
      python verify_dataset.py \
        dataset/final_fix_dataset.jsonl \
        eval_venvs \
        dataset/solutions/tests"
cp $WORKDIR/*.csv $SLURM_SUBMIT_DIR

