#!/usr/bin/env bash
# Retrain the configs used in the paired analysis of the report (Sec. 3.2-3.5) with test forecasts saved
# to results/preds/, spread over the given GPUs of a manual-reservation machine (one process per GPU).
#   ./scripts/save_preds.sh "0 1 2"
# Config:suffix pairs; the abl_res4_* runs are saved under their own name, the others with `_preds`
# (keeps the mean-MAE results of the original runs untouched).
set -eo pipefail
cd "$(dirname "$0")/.."
read -ra G <<< "${1:-0}"
JOBS=(main_gru:_preds main_lstm:_preds main_tcn:_preds
      abl_res_gru_L168:_preds abl_res_tcn_L168:_preds abl_res_cnngru_L168:_preds
      abl_res_gru_L672:_preds abl_res_tcn_L336:_preds abl_res_l1_tcn_L672:_preds
      abl_res4_gru_L672: abl_res4_tcn_L672: abl_res4_l1_tcn_L672:)
uenv verbose cuda-11.4.4 cudnn-11.x-8.8.0
uenv miniconda3-py312
source .venv/bin/activate
mkdir -p logs
worker() {  # $1 = gpu index, rest = config:suffix
  local gpu=$1; shift
  for j in "$@"; do
    c=${j%%:*}; s=${j#*:}
    CUDA_VISIBLE_DEVICES=$gpu python -u -m src.train --config configs/$c.yaml --seeds 5 \
      --save-preds --suffix "$s" > "logs/$c$s.out" 2>&1 || echo "[gpu $gpu] $c FAILED (see logs/$c$s.out)"
  done
}
for i in "${!G[@]}"; do
  mine=()
  for k in "${!JOBS[@]}"; do (( k % ${#G[@]} == i )) && mine+=("${JOBS[$k]}"); done
  worker "${G[$i]}" "${mine[@]}" &
done
wait
echo "done; analysis: see README, section 'Reproducing the report'"
