import torch
from src.models import build, receptive_field, n_params


def test_shapes():
    x = torch.randn(4, 200, 10)
    for name in ("lstm", "gru", "tcn", "cnngru"):
        assert build(name, 10)(x).shape == (4, 24, 10)


def test_tcn_is_causal_and_rf_matches():
    m = build("tcn", 3, n_blocks=4, dropout=0.0).eval()
    rf = m.receptive_field
    assert rf == receptive_field(4) == 61
    x = torch.randn(1, 200, 3)
    y = m(x)
    x2 = x.clone()
    x2[:, -rf - 1] += 10.0          # just outside receptive field of the last step
    assert torch.allclose(m(x2), y, atol=1e-5)
    x3 = x.clone()
    x3[:, -rf] += 10.0              # first step inside the receptive field
    assert not torch.allclose(m(x3), y, atol=1e-5)


def test_cnngru_is_causal_and_params_match():
    m = build("cnngru", 3, dropout=0.0).eval()
    x = torch.randn(1, 100, 3)
    y = m(x)
    # the conv front-end is causal: changing the last step must change the output ...
    x2 = x.clone()
    x2[:, -1] += 10.0
    assert not torch.allclose(m(x2), y, atol=1e-5)
    # ... and the conv features at step t only see steps <= t (checked on the conv stack)
    z = x.transpose(1, 2)
    f = lambda a: torch.relu(m.conv2(torch.nn.functional.pad(
        torch.relu(m.conv1(torch.nn.functional.pad(a, (m.pad, 0)))), (m.pad, 0))))
    z2 = z.clone()
    z2[:, :, 60:] += 10.0
    assert torch.allclose(f(z)[:, :, :60], f(z2)[:, :, :60], atol=1e-5)
    assert abs(n_params(build("cnngru", 10)) - n_params(build("gru", 10, hidden=72))) < 1500
