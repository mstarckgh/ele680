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
