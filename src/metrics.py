"""Error metrics on the original (un-normalised) scale. Inputs: (N, 24, C) arrays."""
import numpy as np


def evaluate(pred, true):
    err = pred - true
    mae = np.abs(err).mean()
    rmse = np.sqrt((err ** 2).mean())
    # LSTNet metrics (Lai et al. 2018): RSE and mean per-series correlation
    rse = np.sqrt((err ** 2).sum()) / np.sqrt(((true - true.mean()) ** 2).sum())
    p, t = pred.reshape(-1, pred.shape[-1]), true.reshape(-1, true.shape[-1])
    corr = np.mean([np.corrcoef(p[:, c], t[:, c])[0, 1] for c in range(p.shape[1])])
    wape = np.abs(err).sum() / np.abs(true).sum()
    return dict(mae=mae, rmse=rmse, rse=rse, corr=corr, wape=wape)


def per_horizon_mae(pred, true):
    return np.abs(pred - true).mean(axis=(0, 2))  # (24,)
