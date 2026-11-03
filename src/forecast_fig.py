"""Forecast vs. actual for one typical test weekday (sum of the 10 clients): mean-4w and the directly trained
LSTM, GRU and TCN, seed 0, L = 168.
python -m src.forecast_fig

The day is chosen by a rule that favours no model: among weekdays, the one whose errors are jointly closest
to the median error of each of the four models (a day picked for one model's median can be an outlier for
another, e.g. 16 Jun 2014 is at the 95th percentile of mean-4w's error).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import START, get_clients, load_hourly

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e3e3e3"
# name -> (results/preds name, colour, linestyle). Colours: categorical slots 1-3 of the reference palette
# (validated all-pairs); the non-learned baseline is neutral gray and dashed.
MODELS = {"mean-4w": (None, MUTED, (0, (4, 2))),
          "LSTM": ("main_lstm_preds", "#2a78d6", "-"),
          "GRU": ("main_gru_preds", "#eb6834", "-"),
          "TCN": ("main_tcn_preds", "#1baf7a", "-")}


def load(name):
    z = np.load(ROOT / "results" / "preds" / f"{name}_seed0.npz")
    return pd.to_datetime(z["times"]), z["pred"].sum(2), z["true"].sum(2)   # (N, 24) total load in kW


def main():
    x = load_hourly().loc[START:, get_clients(load_hourly())]
    total = x.sum(axis=1)
    times, _, true = load("main_tcn_preds")
    day = {n: load(f)[1] for n, (f, _, _) in MODELS.items() if f}
    # mean of the same 24 hours on the previous 4 weeks (as in src/baselines.py; needs no model input window)
    day["mean-4w"] = np.stack([np.mean([total.loc[t - pd.Timedelta(days=7 * k):
                                                  t - pd.Timedelta(days=7 * k) + pd.Timedelta(hours=23)].values
                                        for k in range(1, 5)], axis=0) for t in times])
    day = {n: day[n] for n in MODELS}
    assert all(np.allclose(load(f)[2], true) for f, _, _ in MODELS.values() if f)

    err = np.stack([np.abs(p - true).mean(1) for p in day.values()], axis=1)       # (N, 4) MAE per day and model
    wk = np.flatnonzero(times.day_of_week < 5)
    med = np.median(err[wk], axis=0)
    i = wk[np.argmin((np.abs(err[wk] - med) / med).sum(1))]

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.plot(range(24), true[i], color=INK, lw=1.8, label="actual", zorder=5)
    for (name, (_, colour, ls)), e in zip(MODELS.items(), err[i]):
        ax.plot(range(24), day[name][i], color=colour, lw=1.3, ls=ls, label=f"{name} (MAE {e:.0f} kW)")
    ax.set_xlabel("forecast hour (0 = midnight)", fontsize=7)
    ax.set_ylabel("total load of the 10 clients [kW]", fontsize=6.5)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_title(f"{times[i].strftime('%A %d %B %Y')}", fontsize=7, loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.tick_params(labelsize=6.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, columnspacing=1.0, handlelength=1.6)
    fig.tight_layout()
    out = ROOT / "figures"
    (out / "png").mkdir(exist_ok=True)
    fig.savefig(out / "forecast_examples.pdf", bbox_inches="tight")
    fig.savefig(out / "png" / "forecast_examples.png", dpi=170, bbox_inches="tight")
    print("day:", times[i].date(), "MAE per model [kW]:", dict(zip(MODELS, err[i].round(0))),
          "| medians:", dict(zip(MODELS, med.round(0))))


if __name__ == "__main__":
    main()
