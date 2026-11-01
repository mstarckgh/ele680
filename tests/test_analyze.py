import numpy as np
import pandas as pd

from src.analyze import block_bootstrap_ci, day_ae, paired


def test_day_ae_shape_and_value():
    true = np.zeros((5, 24, 3))
    pred = np.ones((2, 5, 24, 3))
    ae = day_ae(pred, true)
    assert ae.shape == (2, 5) and np.allclose(ae, 1.0)


def test_paired_uses_only_common_days():
    t_long = pd.date_range("2014-01-01", periods=60, freq="D")
    t_short = t_long[20:]                                   # longer input window: loses first days
    ref = (np.full((1, 60), 2.0), t_long)
    other = (np.full((3, 40), 1.5), t_short)
    r = paired(ref, other)
    assert r["days"] == 40 and np.isclose(r["diff"], 0.5) and r["better"] == 1.0
    assert r["lo"] <= r["diff"] <= r["hi"]


def test_bootstrap_ci_covers_mean_of_noisy_diffs():
    rng = np.random.default_rng(1)
    d = rng.normal(0.4, 1.0, size=200)
    lo, hi = block_bootstrap_ci(d)
    assert lo < d.mean() < hi and hi - lo < 1.0


def test_seasonal_avg_matches_naive_for_one_week_and_averages_weeks():
    from src.baselines import seasonal_avg, seasonal_naive
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 672, 2))
    assert np.allclose(seasonal_avg(x, 1), seasonal_naive(x[:, -168:], 168))
    w1, w2 = x[:, -168:-144], x[:, -336:-312]
    assert np.allclose(seasonal_avg(x, 2), (w1 + w2) / 2)
    assert seasonal_avg(x, 4, "median").shape == (3, 24, 2)
