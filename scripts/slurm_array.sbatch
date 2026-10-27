#!/usr/bin/env bash
# One array task per config in configs/.
# Usage: sbatch --array=0-$(( $(ls configs/*.yaml | wc -l) - 1 )) scripts/slurm_array.sbatch
# Adjust partition/GRES/time to your cluster.
#SBATCH --job-name=ele680
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs results
CFG=$(ls configs/*.yaml | sort | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
echo "config: $CFG"
source .venv/bin/activate
python -m src.train --config "$CFG" --seeds 5
