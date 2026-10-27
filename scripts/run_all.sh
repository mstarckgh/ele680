#!/usr/bin/env bash
# Local / interactive alternative to the Slurm array: run everything sequentially.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.run_baselines
for c in configs/*.yaml; do python -m src.train --config "$c" --seeds 5; done
