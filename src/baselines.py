"""Naive seasonal baselines. Operate on a raw input window (B, L, C)."""
import numpy as np

from .data import HORIZON


def seasonal_naive(x, period=24):
    """Repeat the last `period` hours to fill 24 h (period=24: yesterday; 168: last week)."""
    last = x[:, -period:]
    reps = int(np.ceil(HORIZON / period))
    return np.tile(last, (1, reps, 1))[:, :HORIZON]


def seasonal_avg(x, weeks=4, agg="mean"):
    """Mean/median of the same 24 h over the last `weeks` weeks (needs L >= 168 * weeks)."""
    past = np.stack([x[:, -168 * k:-168 * k + HORIZON] for k in range(1, weeks + 1)])
    return (np.median if agg == "median" else np.mean)(past, axis=0)
