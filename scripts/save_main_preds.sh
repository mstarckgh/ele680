#!/usr/bin/env bash
# Re-run the three main configs saving test forecasts (results/main_*_preds.json stay separate
# from the originals), one per GPU, then run the analysis.   ./scripts/save_main_preds.sh "0 1 2"
set -eo pipefail
cd "$(dirname "$0")/.."
read -ra G <<< "${1:-0 1 2}"
uenv verbose cuda-11.4.4 cudnn-11.x-8.8.0
uenv miniconda3-py312
source .venv/bin/activate
mkdir -p logs
i=0
for m in gru lstm tcn; do
  CUDA_VISIBLE_DEVICES=${G[$((i % ${#G[@]}))]} python -u -m src.train --config configs/main_$m.yaml \
    --seeds 5 --save-preds --suffix _preds > logs/main_${m}_preds.out 2>&1 &
  i=$((i + 1))
done
wait
python -m src.analyze | tee results/analysis_main.txt
