"""Data overview figure: the 10 clients and the chronological split.  python -m src.overview"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .data import START, get_clients, load_hourly

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e3e3e3"
LINE, SPLIT = "#2a6fb0", {"train": "#f2f2f2", "val": "#dcdcdc", "test": "#f2f2f2"}
WEEKS = ("2014-09-01", "2014-09-14 23:00")        # Monday to Sunday, two weeks in the test period


def main():
    df = load_hourly()
    clients = get_clients(df)
    x = df.loc[START:, clients]
    n = len(x)
    i1, i2 = int(n * 0.7), int(n * 0.8)
    bounds = {"train": (x.index[0], x.index[i1]), "val": (x.index[i1], x.index[i2]),
              "test": (x.index[i2], x.index[-1])}

    # mean load as in the report: the 212 forecast days of the test split with L = 168 h of history
    t = x.iloc[i2 + 168:]
    t = t.loc[t.index[t.index.hour == 0][0]:]
    mean_load = t.iloc[:len(t) // 24 * 24].mean()

    fig = plt.figure(figsize=(7.2, 8.2))
    gs = fig.add_gridspec(6, 2, height_ratios=[1.5, 1, 1, 1, 1, 1], hspace=0.55, wspace=0.18)

    # A: total load with the chronological split
    ax = fig.add_subplot(gs[0, :])
    for k, (a, b) in bounds.items():
        ax.axvspan(a, b, color=SPLIT[k], lw=0)
        ax.text(a + (b - a) / 2, 1.02, k, transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=8, color=MUTED)
    daily = x.sum(axis=1).resample("1D").mean()
    ax.plot(daily.index, daily.values, color=LINE, lw=0.8)
    for d, lab in (("2014-06-14", "14 Jun"), ("2014-10-26", "26 Oct")):
        ax.axvline(pd.Timestamp(d), color=INK, lw=0.6, ls=":")
        ax.text(pd.Timestamp(d), 0.02, " " + lab, transform=ax.get_xaxis_transform(), fontsize=7,
                color=INK, rotation=90, va="bottom", ha="right")
    ax.set_xlim(x.index[0], x.index[-1])
    ax.set_ylabel("total load, daily mean [kW]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("A  Sum of the 10 clients and the chronological split (70/10/20)", loc="left", fontsize=9, pad=16)

    # B: two test weeks per client, own y-scale
    seg = x.loc[WEEKS[0]:WEEKS[1]]
    sat = [d for d in pd.date_range(WEEKS[0], WEEKS[1], freq="D") if d.day_of_week == 5]
    for i, c in enumerate(clients):
        ax = fig.add_subplot(gs[1 + i // 2, i % 2])
        for s in sat:
            ax.axvspan(s, s + pd.Timedelta(days=2), color=SPLIT["val"], lw=0, alpha=0.6)
        ax.plot(seg.index, seg[c].values, color=LINE, lw=0.8)
        ax.set_ylim(0, seg[c].max() * 1.15)
        ax.set_xlim(seg.index[0], seg.index[-1])
        ax.text(0.01, 0.97, f"{c}  (test mean {mean_load[c]:.0f} kW)", transform=ax.transAxes, fontsize=7,
                color=INK, va="top")
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_xticks(pd.date_range(WEEKS[0], WEEKS[1], freq="7D"))
        ax.set_xticklabels([d.strftime("%d %b") for d in pd.date_range(WEEKS[0], WEEKS[1], freq="7D")])
        if i % 2 == 0:
            ax.set_ylabel("kW", fontsize=7)
        if i == 0:
            ax.set_title("B  Two weeks of hourly load per client (1-14 Sep 2014, weekends shaded; own y-scale)",
                         loc="left", fontsize=9, x=0)
    for a in fig.axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    out = ROOT / "figures"
    fig.savefig(out / "data_overview.pdf", bbox_inches="tight")
    fig.savefig(out / "png" / "data_overview.png", dpi=170, bbox_inches="tight")
    print("wrote", out / "data_overview.pdf")


if __name__ == "__main__":
    main()
