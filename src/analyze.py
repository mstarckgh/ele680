"""Per-client / per-weekday / DST-sensitivity analysis of saved test forecasts.

  python -m src.analyze main_gru_preds main_lstm_preds main_tcn_preds
Needs results/preds/<name>_seed*.npz (train with --save-preds --suffix _preds) and data/.
"""
import sys

import numpy as np
import pandas as pd
import torch

from . import baselines
from .data import WindowDataset, load_splits
from .metrics import evaluate
from .train import ROOT

# forecast days hit by the Oct 2014 clock change (01:00 hour doubled) or copying it a week later
DST_DAYS = pd.to_datetime(["2014-10-26", "2014-11-02"])


def load_preds(name):
    fs = sorted((ROOT / "results" / "preds").glob(f"{name}_seed*.npz"))
    zs = [np.load(f) for f in fs]
    return np.stack([z["pred"] for z in zs]), zs[0]["true"], pd.to_datetime(zs[0]["times"])


def summarize(pred, true, days, clients):
    """pred (S, N, 24, C) or (N, 24, C); returns dict of mean-over-seeds metrics."""
    pred = pred if pred.ndim == 4 else pred[None]
    ae = np.abs(pred - true).mean(axis=(0, 2))            # (N, C): MAE per window and client
    wk = np.asarray(days.dayofweek >= 5)
    out = dict(
        mae=ae.mean(), mae_weekday=ae[~wk].mean(), mae_weekend=ae[wk].mean(),
        wape_per_client=(ae.sum(0) / np.abs(true).mean(1).sum(0)),
    )
    keep = ~np.asarray(days.normalize().isin(DST_DAYS))
    out["mae_noDST"] = ae[keep].mean()
    out["rmse_noDST"] = float(np.sqrt(((pred[:, keep] - true[keep]) ** 2).mean()))
    out["mae_per_client"] = ae.mean(0)
    return out


def main(names):
    data = load_splits()
    clients = data["clients"]
    ds = WindowDataset(data["test"], data["index"]["test"], 168, align_midnight=True)
    x = torch.stack([ds[i][0] for i in range(len(ds))]).numpy() * data["std"] + data["mean"]
    rows, per_client = {}, {}
    ref = None
    for name in names:
        pred, true, times = load_preds(name)
        days = pd.DatetimeIndex(times)
        ref = (true, days) if ref is None else ref
        rows[name] = summarize(pred, true, days, clients)
    true, days = ref
    for nm, period in (("naive_168h", 168), ("naive_24h", 24)):
        rows[nm] = summarize(baselines.seasonal_naive(x, period), true, days, clients)
    print(f"{'model':<22}{'MAE':>8}{'wkday':>8}{'wkend':>8}{'MAE-noDST':>11}{'RMSE-noDST':>12}")
    for n, r in rows.items():
        print(f"{n:<22}{r['mae']:>8.2f}{r['mae_weekday']:>8.2f}{r['mae_weekend']:>8.2f}"
              f"{r['mae_noDST']:>11.2f}{r['rmse_noDST']:>12.2f}")
    print("\nper-client WAPE [%] (scale-free; mean load in kW in header)")
    print(f"{'model':<22}" + "".join(f"{c:>8}" for c in clients))
    print(f"{'mean load kW':<22}" + "".join(f"{m:>8.0f}" for m in true.mean(axis=(0, 1))))
    for n, r in rows.items():
        print(f"{n:<22}" + "".join(f"{100 * w:>8.1f}" for w in r["wape_per_client"]))


if __name__ == "__main__":
    main(sys.argv[1:] or ["main_gru_preds", "main_lstm_preds", "main_tcn_preds"])
