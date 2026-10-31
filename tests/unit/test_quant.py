import numpy as np
import pytest

from l3m import export, quant
from tests.conftest import SPECS, tiny_spec


def q4_0_values(raw, shape):
    raw = raw.reshape(-1, 18)
    d, nib = raw[:, :2].copy().view(np.float16).astype(np.float32), raw[:, 2:]
    return ((np.concatenate([nib & 15, nib >> 4], axis=1).astype(np.float32) - 8) * d).reshape(shape)


@pytest.mark.parametrize("spec", SPECS)
def test_awq_folds_exactly(spec, monkeypatch):
    monkeypatch.setattr(quant, "ALPHAS", [1.0])
    m = export.load_model(tiny_spec(spec))
    ids, H = list(range(0, 260, 5)), {}
    want = quant.forward(m, ids, stats=H)
    for key, mats, producers in quant.groups(m):
        if producers:
            quant.awq(m.tensors, H[key], [w for w in mats if w in m.tensors], [p for p in producers if p in m.tensors], m.dim // m.n_heads)
    assert not np.allclose(m.tensors["blk.0.attn_norm"], export.load_model(tiny_spec(spec)).tensors["blk.0.attn_norm"])
    assert np.abs(quant.forward(m, ids) - want).max() < 1e-5 * np.abs(want).max()


def test_gptq_beats_round_to_nearest():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((512, 128)) @ rng.standard_normal((128, 128)) * np.exp(rng.standard_normal(128))
    W, H = rng.standard_normal((64, 128)).astype(np.float32), X.T @ X
    Q, D = quant.gptq(W, H)
    assert (q4_0_values(export.q4_0(Q, D), W.shape) == Q).all()

    def err(V):
        return np.trace((W - V) @ H @ (W - V).T)
    assert err(Q) < 0.5 * err(q4_0_values(export.q4_0(W), W.shape))
