#!/bin/bash
#SBATCH --job-name=eval_job
#SBATCH --output=logs/eval_job-%j.out
#SBATCH --error=logs/eval_job-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=long-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16


cot_flag=""
if [ "$7" -eq 1 ]; then
    cot_flag="--cot"
fi
python evaluate.py --json-out-file $6 \
                --output-path results/$5 \
                --seed $1 \
                --model-name $5 \
                --temperature $2 \
                --n-generate $3 \
                --k $4 \
                --base-path $SCRATCH/eval_venvs \
                --id_start 0 \
                --dataset-path dataset/fix_dataset.jsonl \
                --scratch $SCRATCH \
                --enable-wandb \
                $cot_flag
