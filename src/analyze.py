"""Per-client / per-weekday / DST-sensitivity analysis of saved test forecasts.

  python -m src.analyze main_gru_preds main_lstm_preds main_tcn_preds [--ref naive_168h]
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


def day_ae(pred, true):
    """MAE per seed and forecast day: (S, N)."""
    pred = pred if pred.ndim == 4 else pred[None]
    return np.abs(pred - true).mean(axis=(2, 3))


def block_bootstrap_ci(d, block=7, n_boot=5000, seed=0):
    """95% CI of mean(d) from a moving-block bootstrap (daily errors are autocorrelated)."""
    rng = np.random.default_rng(seed)
    n = len(d)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    return tuple(np.percentile(d[idx].mean(axis=1), [2.5, 97.5]))


def paired(ref, other):
    """Compare two (ae (S, N), times) entries on the forecast days both cover.

    Models with a longer input window L lose the first L hours of the test split, so they are
    scored on fewer days; only the common days are compared here. diff > 0: `other` is better.
    """
    (ae_r, t_r), (ae_o, t_o) = ref, other
    common = t_r.intersection(t_o)
    ir, io = t_r.get_indexer(common), t_o.get_indexer(common)
    r, o = ae_r[:, ir], ae_o[:, io]
    d = r.mean(0) - o.mean(0)                       # per day, averaged over seeds
    per_seed = r.mean() - o.mean(1)                 # per seed: mean over days
    lo, hi = block_bootstrap_ci(d)
    return dict(days=len(common), diff=d.mean(), lo=lo, hi=hi, better=(d > 0).mean(),
                seed_min=per_seed.min(), seed_max=per_seed.max())


def print_paired(entries, ref_name):
    ref = entries[ref_name]
    print(f"\npaired vs {ref_name} (common days; diff = ref MAE - model MAE in kW, >0 = model better;"
          " 95% CI from 7-day block bootstrap)")
    print(f"{'model':<22}{'days':>6}{'diff':>8}{'CI low':>9}{'CI high':>9}{'days better':>13}{'seed range':>16}")
    for n, e in entries.items():
        if n == ref_name:
            continue
        r = paired(ref, e)
        print(f"{n:<22}{r['days']:>6}{r['diff']:>8.2f}{r['lo']:>9.2f}{r['hi']:>9.2f}"
              f"{100 * r['better']:>12.0f}%{r['seed_min']:>8.2f}..{r['seed_max']:<5.2f}")


def main(names, ref_name="naive_168h"):
    data = load_splits()
    clients = data["clients"]
    ds = WindowDataset(data["test"], data["index"]["test"], 168, align_midnight=True)
    x = torch.stack([ds[i][0] for i in range(len(ds))]).numpy() * data["std"] + data["mean"]
    rows, entries = {}, {}
    for name in names:
        pred, true, times = load_preds(name)
        days = pd.DatetimeIndex(times)
        rows[name] = summarize(pred, true, days, clients)
        entries[name] = (day_ae(pred, true), days)
    # baselines are scored on all L=168 windows, i.e. their own (longer) set of days
    true = torch.stack([ds[i][1] for i in range(len(ds))]).numpy() * data["std"] + data["mean"]
    days = pd.DatetimeIndex(data["index"]["test"][ds.starts])
    for nm, period in (("naive_168h", 168), ("naive_24h", 24)):
        pred_b = baselines.seasonal_naive(x, period)
        rows[nm] = summarize(pred_b, true, days, clients)
        entries[nm] = (day_ae(pred_b, true), days)
    # multi-week averages need 4 weeks of history, so they are scored on the (fewer) L=672 days
    ds4 = WindowDataset(data["test"], data["index"]["test"], 672, align_midnight=True)
    x4 = torch.stack([ds4[i][0] for i in range(len(ds4))]).numpy() * data["std"] + data["mean"]
    true4 = torch.stack([ds4[i][1] for i in range(len(ds4))]).numpy() * data["std"] + data["mean"]
    days4 = pd.DatetimeIndex(data["index"]["test"][ds4.starts])
    for nm, weeks, agg in (("mean_2w", 2, "mean"), ("mean_4w", 4, "mean"), ("median_4w", 4, "median")):
        pred_b = baselines.seasonal_avg(x4, weeks, agg)
        rows[nm] = summarize(pred_b, true4, days4, clients)
        entries[nm] = (day_ae(pred_b, true4), days4)
    print(f"{'model':<22}{'MAE':>8}{'wkday':>8}{'wkend':>8}{'MAE-noDST':>11}{'RMSE-noDST':>12}")
    for n, r in rows.items():
        print(f"{n:<22}{r['mae']:>8.2f}{r['mae_weekday']:>8.2f}{r['mae_weekend']:>8.2f}"
              f"{r['mae_noDST']:>11.2f}{r['rmse_noDST']:>12.2f}")
    print("\nper-client WAPE [%] (scale-free; mean load in kW in header)")
    print(f"{'model':<22}" + "".join(f"{c:>8}" for c in clients))
    print(f"{'mean load kW':<22}" + "".join(f"{m:>8.0f}" for m in true.mean(axis=(0, 1))))
    for n, r in rows.items():
        print(f"{n:<22}" + "".join(f"{100 * w:>8.1f}" for w in r["wape_per_client"]))
    print_paired(entries, ref_name)


if __name__ == "__main__":
    args = sys.argv[1:]
    ref = args[args.index("--ref") + 1] if "--ref" in args else "naive_168h"
    if "--ref" in args:
        del args[args.index("--ref"):args.index("--ref") + 2]
    main(args or ["main_gru_preds", "main_lstm_preds", "main_tcn_preds"], ref)
