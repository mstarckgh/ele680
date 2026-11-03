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
    residual_figs(res, base, fig_dir)


def residual_figs(res, base, fig_dir):
    """Residual-to-naive models vs input length and the naive-168h line (needs L >= 168)."""
    plt.figure(figsize=(5, 3.5))
    for m, ls in (("gru", "--"), ("tcn", "-")):
        for tag, mk in (("res", "o"),):
            pts = {int(n.split("_L")[1]): ms(r["runs"], "mae") for n, r in res.items()
                   if n.startswith(f"abl_{tag}_{m}_L") and not n.endswith("_preds")}
            if pts:
                Ls = sorted(pts)
                plt.errorbar(Ls, [pts[l][0] for l in Ls], [pts[l][1] for l in Ls], marker=mk,
                             ls=ls, label=m.upper())
    if "naive_168h" in base:
        plt.axhline(base["naive_168h"]["mae"], color="k", lw=1, label="naive-168h")
    plt.xscale("log"); plt.xticks([168, 336, 672], ["168", "336", "672"])
    plt.gca().xaxis.set_minor_formatter(plt.NullFormatter())
    plt.xlabel("input length L [h]"); plt.ylabel("test MAE [kW]")
    plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(fig_dir / "residual_length.pdf")
    # TCN receptive field (n_blocks) with the residual, L=672
    pts = {int(n.split("_n")[1]): ms(r["runs"], "mae") for n, r in res.items()
           if n.startswith("rfres_tcn_n")}
    if "abl_res_tcn_L672" in res:
        pts[7] = ms(res["abl_res_tcn_L672"]["runs"], "mae")
    if pts:
        plt.figure(figsize=(5, 3.5))
        ns = sorted(pts)
        plt.errorbar([1 + 4 * (2 ** n - 1) for n in ns], [pts[n][0] for n in ns],
                     [pts[n][1] for n in ns], marker="o")
        plt.axvline(672, color="gray", ls=":", label="input length")
        if "naive_168h" in base:
            plt.axhline(base["naive_168h"]["mae"], color="k", lw=1, label="naive-168h")
        plt.xscale("log"); plt.xticks([29, 125, 509, 1021], ["29", "125", "509", "1021"])
        plt.gca().xaxis.set_minor_formatter(plt.NullFormatter())
        plt.xlabel("TCN receptive field [h]"); plt.ylabel("test MAE [kW]")
        plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(fig_dir / "rf_residual.pdf")


if __name__ == "__main__":
    main()
