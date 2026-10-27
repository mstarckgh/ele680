#!/usr/bin/env bash
# One-time conda env on gorina4 (manual-reservation machine, no Slurm). Run interactively:
#   uenv avail                      # check the names below exist on this server
#   ./scripts/gorina4_setup.sh
# P100s (sm_60) need a CUDA 11.x PyTorch build; recent cu12/cu13 wheels may drop sm_60.
set -euo pipefail
uenv verbose cuda-11.4.4 cudnn-11.x-8.8.0
uenv miniconda3-py39
# pytorch-cuda=11.8 (nvidia channel) forces a CUDA build; a bare `pytorch` can resolve to CPU-only.
conda create -n ele680_env -c pytorch -c nvidia -c conda-forge python=3.11 \
  pytorch=2.5.1 pytorch-cuda=11.8 numpy pandas pyarrow matplotlib pyyaml pytest -y
conda activate ele680_env
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0)); x=torch.randn(8,8,device='cuda'); print((x@x).sum().item())"
