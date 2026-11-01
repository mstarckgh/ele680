"""Generate all experiment configs into configs/ (run once; outputs are committed).

  main_*            L=168, parameter-matched LSTM / GRU / TCN
  len_<model>_L<L>  input-length sweep
  rf_tcn_n<n>       TCN receptive-field ablation at L=672 (RF = 1 + 4 (2^n - 1))
"""
from pathlib import Path

import yaml

OUT = Path(__file__).resolve().parents[1] / "configs"
KW = {"lstm": {"hidden": 64}, "gru": {"hidden": 72}, "tcn": {"hidden": 38, "n_blocks": 7}}


def dump(name, model, L, **kw):
    cfg = dict(model=model, L=L, epochs=40, model_kwargs={**KW[model], **kw})
    (OUT / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


for m in KW:
    dump(f"main_{m}", m, 168)
    for L in (24, 72, 336, 672):  # L=168 is main_*
        dump(f"len_{m}_L{L}", m, L)
for n in (3, 4, 5, 6, 8):  # RF = 29, 61, 125, 253, 1021 (n=7 is len_tcn_L672)
    dump(f"rf_tcn_n{n}", "tcn", 672, n_blocks=n)

# Why do trained models lose to "repeat last week"?  L1 loss and residual-to-naive ablations.
for m in ("gru", "tcn"):
    for L in (168, 672):
        for tag, extra in (("l1", dict(loss="l1")), ("res", dict(residual=True)),
                           ("res_l1", dict(residual=True, loss="l1"))):
            cfg = dict(model=m, L=L, epochs=40, model_kwargs=dict(KW[m]), **extra)
            (OUT / f"abl_{tag}_{m}_L{L}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

# Residual-to-naive needs L >= 168: input-length sweep (168, 672 above) and TCN receptive-field ablation.
for m in ("gru", "tcn"):
    cfg = dict(model=m, L=336, epochs=40, model_kwargs=dict(KW[m]), residual=True)
    (OUT / f"abl_res_{m}_L336.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
for n in (3, 4, 5, 6, 8):  # n=7 is abl_res_tcn_L672
    cfg = dict(model="tcn", L=672, epochs=40, model_kwargs={**KW["tcn"], "n_blocks": n}, residual=True)
    (OUT / f"rfres_tcn_n{n}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

# New model: causal-conv front-end + GRU (LSTNet-style), same ~68k budget; plain and residual-to-naive.
KW["cnngru"] = {"conv": 32, "hidden": 64}
dump("main_cnngru", "cnngru", 168)
for L in (168, 336, 672):
    cfg = dict(model="cnngru", L=L, epochs=40, model_kwargs=dict(KW["cnngru"]), residual=True)
    (OUT / f"abl_res_cnngru_L{L}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

# Residual on the 4-week mean (the strongest simple baseline, see report): needs L >= 672.
for m, extra in (("gru", {}), ("tcn", {}), ("tcn", dict(loss="l1"))):
    tag = "res4_l1" if extra else "res4"
    cfg = dict(model=m, L=672, epochs=40, model_kwargs=dict(KW[m]), residual="mean4", **extra)
    (OUT / f"abl_{tag}_{m}_L672.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
