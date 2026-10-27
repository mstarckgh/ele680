"""Seasonal-naive baselines on the test set.  python -m src.run_baselines"""
import json

from .data import load_splits
from .train import ROOT, run_baselines

if __name__ == "__main__":
    out = run_baselines(168, load_splits())
    (ROOT / "results" / "baselines.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
