#!/usr/bin/env bash
# Build the conda env (no GPU requested, as the UiS GPU manual asks). Submit once:
#   sbatch scripts/slurm_setup.sbatch
#SBATCH --job-name=ele680_setup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --time=02:15:00
#SBATCH --output=logs/setup.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
uenv verbose cuda-11.4.4 cudnn-11.x-8.8.0
uenv miniconda3-py39
conda create -n ele680_env -c pytorch -c conda-forge \
  pytorch numpy pandas pyarrow matplotlib pyyaml pytest -y
