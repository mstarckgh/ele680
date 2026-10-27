#!/usr/bin/env bash
# Seasonal-naive baselines (CPU only).
#SBATCH --job-name=ele680_base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --time=00:20:00
#SBATCH --output=logs/baselines.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p results
uenv verbose cuda-11.4.4 cudnn-11.x-8.8.0
uenv miniconda3-py39
conda activate ele680_env
python -u -m src.run_baselines
