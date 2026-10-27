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
# pytorch-cuda=11.8 (nvidia channel) forces a CUDA build; a bare `pytorch` can resolve to CPU-only.
conda create -n ele680_env -c pytorch -c nvidia -c conda-forge python=3.11 \
  pytorch=2.5.1 pytorch-cuda=11.8 numpy pandas pyarrow matplotlib pyyaml pytest -y
