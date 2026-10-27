import torch
from src.models import build, receptive_field, n_params


def test_shapes():
    x = torch.randn(4, 200, 10)
    for name in ("lstm", "gru", "tcn"):
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
