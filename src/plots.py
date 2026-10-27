"""Summarise results/*.json into tables and figures.  python -m src.plots"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load():
    return {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "results").glob("*.json"))}


def ms(runs, key):
    v = [r[key] for r in runs]
    return np.mean(v), np.std(v)


def main():
    res = load()
    base = res.pop("baselines", {})
    print(f"{'config':<18}{'params':>8}{'MAE':>16}{'RMSE':>16}{'RSE':>8}{'CORR':>8}{'time[s]':>9}")
    for name, r in res.items():
        runs = r["runs"]
        (mae, mae_s), (rmse, rmse_s) = ms(runs, "mae"), ms(runs, "rmse")
        print(f"{name:<18}{runs[0]['params']:>8}{mae:>10.2f}±{mae_s:<5.2f}{rmse:>10.2f}±{rmse_s:<5.2f}"
              f"{ms(runs,'rse')[0]:>8.3f}{ms(runs,'corr')[0]:>8.3f}{ms(runs,'train_time')[0]:>9.0f}")
    for k, v in base.items():
        print(f"{k:<18}{'-':>8}{v['mae']:>10.2f}{'':7}{v['rmse']:>10.2f}")

    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    # input-length sweep (L=168 comes from main_*)
    plt.figure(figsize=(5, 3.5))
    for m in ("lstm", "gru", "tcn"):
        pts = {int(n.split("_L")[1]): ms(r["runs"], "mae") for n, r in res.items()
               if n.startswith(f"len_{m}_L")}
        if f"main_{m}" in res:
            pts[168] = ms(res[f"main_{m}"]["runs"], "mae")
        Ls = sorted(pts)
        plt.errorbar(Ls, [pts[l][0] for l in Ls], [pts[l][1] for l in Ls], marker="o", label=m.upper())
    plt.xscale("log"); plt.xlabel("input length L [h]"); plt.ylabel("test MAE [kW]")
    plt.legend(); plt.tight_layout(); plt.savefig(fig_dir / "input_length.pdf")
    # per-horizon error
    plt.figure(figsize=(5, 3.5))
    for m in ("lstm", "gru", "tcn"):
        if f"main_{m}" in res:
            h = np.mean([r["horizon_mae"] for r in res[f"main_{m}"]["runs"]], axis=0)
            plt.plot(range(1, 25), h, label=m.upper())
    plt.xlabel("horizon [h]"); plt.ylabel("test MAE [kW]"); plt.legend(); plt.tight_layout()
    plt.savefig(fig_dir / "horizon_mae.pdf")


if __name__ == "__main__":
    main()
