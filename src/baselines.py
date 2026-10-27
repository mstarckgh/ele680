"""Naive seasonal baselines. Operate on a raw input window (B, L, C)."""
import numpy as np

from .data import HORIZON


def seasonal_naive(x, period=24):
    """Repeat the last `period` hours to fill 24 h (period=24: yesterday; 168: last week)."""
    last = x[:, -period:]
    reps = int(np.ceil(HORIZON / period))
    return np.tile(last, (1, reps, 1))[:, :HORIZON]
