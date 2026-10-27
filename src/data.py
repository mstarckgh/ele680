"""Load, aggregate and window the UCI ElectricityLoadDiagrams20112014 data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "LD2011_2014.txt"
HOURLY = ROOT / "data" / "hourly.parquet"
CLIENTS_JSON = ROOT / "configs" / "clients.json"

HORIZON = 24
START = "2012-01-01"


def load_hourly() -> pd.DataFrame:
    """All clients, hourly mean of the 15-min readings (cached as parquet)."""
    if HOURLY.exists():
        return pd.read_parquet(HOURLY)
    df = pd.read_csv(RAW, sep=";", decimal=",", index_col=0, parse_dates=True)
    df = df.resample("1h").mean()
    df.to_parquet(HOURLY)
    return df


def select_clients(df: pd.DataFrame, n: int = 10, seed: int = 0) -> list[str]:
    """Seeded sample of clients with no zero-load hours after START."""
    d = df.loc[START:]
    ok = [c for c in d.columns if (d[c] > 0).all()]
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(ok, size=n, replace=False).tolist())


def get_clients(df: pd.DataFrame) -> list[str]:
    if CLIENTS_JSON.exists():
        return json.loads(CLIENTS_JSON.read_text())
    clients = select_clients(df)
    CLIENTS_JSON.write_text(json.dumps(clients, indent=1))
    return clients


def load_splits(train=0.7, val=0.1):
    """Chronological split, z-scored with train statistics only.

    Returns dict of float32 arrays (T, C) plus mean/std and the time index per split.
    """
    df = load_hourly()
    clients = get_clients(df)
    x = df.loc[START:, clients]
    n = len(x)
    i1, i2 = int(n * train), int(n * (train + val))
    parts = {"train": x.iloc[:i1], "val": x.iloc[i1:i2], "test": x.iloc[i2:]}
    mean = parts["train"].mean().values.astype(np.float32)
    std = parts["train"].std().values.astype(np.float32)
    out = {k: ((v.values - mean) / std).astype(np.float32) for k, v in parts.items()}
    out["index"] = {k: v.index for k, v in parts.items()}
    out["mean"], out["std"], out["clients"] = mean, std, clients
    return out


class WindowDataset(Dataset):
    """(L past hours of all clients) -> (next 24 h of all clients).

    Windows never cross split boundaries. `align_midnight` keeps only windows whose
    forecast starts at 00:00 (one forecast per day, the usual day-ahead setting).
    """

    def __init__(self, series: np.ndarray, index: pd.DatetimeIndex, L: int,
                 stride: int = 1, align_midnight: bool = False):
        self.x = torch.from_numpy(series)
        starts = np.arange(L, len(series) - HORIZON + 1, stride)
        if align_midnight:
            starts = starts[index[starts].hour == 0]
        self.starts, self.L = starts, L

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        return self.x[s - self.L:s], self.x[s:s + HORIZON]  # (L,C), (24,C)
