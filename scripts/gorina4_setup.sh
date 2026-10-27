#!/usr/bin/env bash
# One-time venv on gorina4 (manual-reservation machine, no Slurm). Run from the repo root:
#   ./scripts/gorina4_setup.sh
# Interpreter comes from miniconda (has SSL; uenv python-3.12.1 does not), packages from pip.
# P100s (sm_60) need a CUDA 11.x build: cu118 wheels still support them, newer ones may not.
# (conda create was tried first and the solver was far too slow.)
set -eo pipefail
cd "$(dirname "$0")/.."
uenv verbose cuda-11.4.4 cudnn-11.x-8.8.0
uenv miniconda3-py312
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas pyarrow matplotlib pyyaml pytest
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0)); x=torch.randn(8,8,device='cuda'); print((x@x).sum().item())"
