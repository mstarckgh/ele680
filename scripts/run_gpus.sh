#!/usr/bin/env bash
# Spread configs over the given GPUs of a manual-reservation machine, one process per GPU.
# Skips configs whose results/<name>.json already exists. Logs go to logs/<name>.out.
#   tmux new -s ele680
#   ./scripts/run_gpus.sh "0 1 2" [seeds] [config-glob]
# Example: ./scripts/run_gpus.sh "1 4" 5 'configs/main_*.yaml'
set -euo pipefail
cd "$(dirname "$0")/.."
GPUS=(${1:?usage: run_gpus.sh "0 1 2" [seeds] [glob]})
SEEDS=${2:-5}
GLOB=${3:-configs/*.yaml}
mkdir -p logs results
uenv verbose cuda-11.4.4 cudnn-11.x-8.8.0
uenv miniconda3-py39
conda activate ele680_env
export OMP_NUM_THREADS=4

CFGS=()
for c in $(ls $GLOB | sort); do
  [[ -f results/$(basename "$c" .yaml).json ]] || CFGS+=("$c")
done
echo "${#CFGS[@]} configs on ${#GPUS[@]} GPUs: ${GPUS[*]}"

worker() {  # $1 = gpu index, rest = configs
  local gpu=$1; shift
  for c in "$@"; do
    n=$(basename "$c" .yaml)
    echo "[gpu $gpu] $n start $(date +%T)"
    CUDA_VISIBLE_DEVICES=$gpu python -u -m src.train --config "$c" --seeds "$SEEDS" > "logs/$n.out" 2>&1 \
      || echo "[gpu $gpu] $n FAILED (see logs/$n.out)"
  done
}
for i in "${!GPUS[@]}"; do
  mine=()
  for j in "${!CFGS[@]}"; do (( j % ${#GPUS[@]} == i )) && mine+=("${CFGS[$j]}"); done
  worker "${GPUS[$i]}" "${mine[@]}" &
done
wait
echo "done $(date +%T)"
