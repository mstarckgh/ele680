"""Train one model config over several seeds.  python -m src.train --config configs/main_tcn.yaml"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from . import baselines
from .data import HORIZON, WindowDataset, load_splits
from .metrics import evaluate, per_horizon_mae
from .models import build, n_params

ROOT = Path(__file__).resolve().parents[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def forward(model, x, residual):
    """Model output; with `residual` the net only learns a correction to last week's load."""
    out = model(x)
    if residual:
        out = out + x[:, -168:-168 + HORIZON]  # same hours one week earlier (needs L >= 168)
    return out


@torch.no_grad()
def predict(model, ds, residual=False, bs=512):
    model.eval()
    out, tgt = [], []
    for x, y in DataLoader(ds, batch_size=bs):
        out.append(forward(model, x.to(DEV), residual).cpu())
        tgt.append(y)
    return torch.cat(out).numpy(), torch.cat(tgt).numpy()


def run(cfg, seed, data):
    torch.manual_seed(seed)
    np.random.seed(seed)
    L, C = cfg["L"], len(data["clients"])
    mk = lambda s, **kw: WindowDataset(data[s], data["index"][s], L, **kw)
    train_dl = DataLoader(mk("train"), batch_size=cfg.get("batch_size", 64), shuffle=True,
                          drop_last=True)
    val_ds, test_ds = mk("val", align_midnight=True), mk("test", align_midnight=True)

    model = build(cfg["model"], C, **cfg.get("model_kwargs", {})).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    loss_fn = {"mse": torch.nn.MSELoss, "l1": torch.nn.L1Loss}[cfg.get("loss", "mse")]()
    residual = cfg.get("residual", False)
    assert not residual or L >= 168, "residual needs L >= 168"
    best, best_state, bad, log = 1e9, None, 0, []
    t0 = time.time()
    for ep in range(cfg.get("epochs", 40)):
        model.train()
        tot = 0.0
        for x, y in train_dl:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            loss = loss_fn(forward(model, x, residual), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip", 1.0))
            opt.step()
            tot += loss.item()
        p, t = predict(model, val_ds, residual)
        val = float(((p - t) ** 2).mean())
        sched.step(val)
        log.append(dict(epoch=ep, train=tot / len(train_dl), val=val, time=time.time() - t0))
        print(f"seed {seed} ep {ep} train {log[-1]['train']:.4f} val {val:.4f} "
              f"{log[-1]['time']:.0f}s", flush=True)
        if val < best:
            best, bad = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.get("patience", 6):
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state)

    p, t = predict(model, test_ds, residual)
    sd, mu = data["std"], data["mean"]
    p, t = p * sd + mu, t * sd + mu  # back to kW
    res = evaluate(p, t)
    res.update(seed=seed, params=n_params(model), train_time=train_time, epochs=len(log),
               horizon_mae=per_horizon_mae(p, t).tolist(), log=log)
    return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in res.items()}


def run_baselines(L, data):
    ds = WindowDataset(data["test"], data["index"]["test"], L, align_midnight=True)
    x = torch.stack([ds[i][0] for i in range(len(ds))]).numpy()
    t = torch.stack([ds[i][1] for i in range(len(ds))]).numpy()
    sd, mu = data["std"], data["mean"]
    out = {}
    for name, period in [("naive_24h", 24), ("naive_168h", 168)]:
        p = baselines.seasonal_naive(x, period)
        out[name] = evaluate(p * sd + mu, t * sd + mu)
    return {k: {m: float(v) for m, v in d.items()} for k, d in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    data = load_splits()
    name = Path(a.config).stem
    results = [run(cfg, s, data) for s in range(a.seeds)]
    out = ROOT / "results" / f"{name}.json"
    out.write_text(json.dumps(dict(cfg=cfg, runs=results), indent=1))
    for m in ("mae", "rmse", "rse", "corr"):
        v = [r[m] for r in results]
        print(f"{name} {m}: {np.mean(v):.4f} +- {np.std(v):.4f}")


if __name__ == "__main__":
    main()
