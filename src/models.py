"""Sequential (LSTM/GRU) and convolutional (TCN) one-shot 24 h forecasters.

All models map (B, L, C) -> (B, 24, C) with a linear head on the last time step.
"""
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from .data import HORIZON


class RNNForecaster(nn.Module):
    def __init__(self, channels, hidden=64, layers=2, dropout=0.1, cell="lstm"):
        super().__init__()
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn(channels, hidden, layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, HORIZON * channels)
        self.channels = channels

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(h[:, -1]).view(-1, HORIZON, self.channels)


class _Chomp(nn.Module):
    """Drop the right padding so convolutions are causal."""
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[:, :, :-self.n] if self.n else x


class _Block(nn.Module):
    def __init__(self, cin, cout, k, dilation, dropout):
        super().__init__()
        pad = (k - 1) * dilation

        def conv(i, o):
            return [weight_norm(nn.Conv1d(i, o, k, padding=pad, dilation=dilation)),
                    _Chomp(pad), nn.ReLU(), nn.Dropout(dropout)]

        self.net = nn.Sequential(*conv(cin, cout), *conv(cout, cout))
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + self.skip(x))


class TCNForecaster(nn.Module):
    def __init__(self, channels, hidden=64, n_blocks=7, kernel=3, dropout=0.1):
        super().__init__()
        blocks = [_Block(channels if i == 0 else hidden, hidden, kernel, 2 ** i, dropout)
                  for i in range(n_blocks)]
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, HORIZON * channels)
        self.channels = channels
        self.receptive_field = receptive_field(n_blocks, kernel)

    def forward(self, x):
        h = self.tcn(x.transpose(1, 2))  # (B, H, L)
        return self.head(h[:, :, -1]).view(-1, HORIZON, self.channels)


def receptive_field(n_blocks, kernel=3):
    """Two convs per block, dilation 2^i: RF = 1 + 2 (k-1) (2^n - 1)."""
    return 1 + 2 * (kernel - 1) * (2 ** n_blocks - 1)


def build(name, channels, **kw):
    if name in ("lstm", "gru"):
        return RNNForecaster(channels, cell=name, **kw)
    if name == "tcn":
        return TCNForecaster(channels, **kw)
    raise ValueError(name)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
