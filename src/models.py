"""Sequential (LSTM/GRU), convolutional (TCN) and hybrid (CNN-GRU) one-shot 24 h forecasters.

All models map (B, L, C) -> (B, 24, C) with a linear head on the last time step.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class CNNGRUForecaster(nn.Module):
    """Causal conv front-end (local shape, e.g. the intraday profile) feeding a GRU (long-range state).

    Two causal convolutions with kernel `kernel` extract short-range features from the raw series;
    their output sequence, with a 1x1 skip from the input, is read by a 2-layer GRU.
    """
    def __init__(self, channels, conv=32, hidden=64, layers=2, kernel=6, dropout=0.1):
        super().__init__()
        self.pad = kernel - 1
        self.conv1 = nn.Conv1d(channels, conv, kernel)
        self.conv2 = nn.Conv1d(conv, conv, kernel)
        self.skip = nn.Conv1d(channels, conv, 1)
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.GRU(conv, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, HORIZON * channels)
        self.channels = channels

    def forward(self, x):
        z = x.transpose(1, 2)                                   # (B, C, L)
        h = self.drop(torch.relu(self.conv1(F.pad(z, (self.pad, 0)))))  # left pad: causal
        h = torch.relu(self.conv2(F.pad(h, (self.pad, 0)))) + self.skip(z)
        o, _ = self.rnn(self.drop(h).transpose(1, 2))
        return self.head(o[:, -1]).view(-1, HORIZON, self.channels)


def receptive_field(n_blocks, kernel=3):
    """Two convs per block, dilation 2^i: RF = 1 + 2 (k-1) (2^n - 1)."""
    return 1 + 2 * (kernel - 1) * (2 ** n_blocks - 1)


def build(name, channels, **kw):
    if name in ("lstm", "gru"):
        return RNNForecaster(channels, cell=name, **kw)
    if name == "tcn":
        return TCNForecaster(channels, **kw)
    if name == "cnngru":
        return CNNGRUForecaster(channels, **kw)
    raise ValueError(name)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
