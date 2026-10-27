#!/usr/bin/env bash
# One array task per config in configs/*.yaml (20 tasks).
# Submit from gorina11 inside /bhome/$USER/ele680 after the setup job finished:
#   sbatch --array=0-19 scripts/slurm_array.sbatch
#SBATCH --job-name=ele680
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs results
CFG=$(ls configs/*.yaml | sort | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
echo "config: $CFG"
uenv verbose cuda-11.4.4 cudnn-11.x-8.8.0
uenv miniconda3-py39
conda activate ele680_env
python -u -m src.train --config "$CFG" --seeds 5
