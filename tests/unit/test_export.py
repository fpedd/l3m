import struct
from types import SimpleNamespace

import numpy as np
import pytest

from l3m import archs, export
from tests.conftest import MODELS, ROOT, SPECS, toc


def blocks(n=64, seed=0):
    return np.random.default_rng(seed).standard_normal((n, 32)).astype(np.float32) * 3


def f16(u8):
    return u8.copy().view(np.float16).astype(np.float32).ravel()


def test_bf16_rounds_to_nearest_even():
    x = np.array([1.0, 1 + 2**-8, 1 + 3 * 2**-8, 1 + 2**-7 + 2**-9, -2.5], dtype=np.float32)
    got = (export.to_bf16(x).astype(np.uint32) << 16).view(np.float32)
    assert got.tolist() == [1.0, 1.0, 1 + 2**-6, 1 + 2**-7, -2.5]


def test_q8_0_round_trip():
    x = blocks()
    raw = export.q8_0(x).reshape(len(x), 34)
    d, q = f16(raw[:, :2]), raw[:, 2:].view(np.int8).astype(np.float32)
    err = np.abs(q * d[:, None] - x).max(axis=1)
    assert (err <= d * 0.6 + 1e-6).all()


HALVES = np.arange(-16, 16, dtype=np.float32) / 2
TIES = np.float32([127, 0.49999997, -0.49999997, 0.5, -0.5, 1.5, -2.5] + [0] * 25)


@pytest.mark.parametrize("dtype,x,ggml", [
    ("q8_0", HALVES, "082c81899199a1a9b1b9c0c8d0d8e0e8f0f8000810182028303840474f575f676f77"),
    ("q8_0", TIES, "003c7f000001ff02fd00000000000000000000000000000000000000000000000000"),
    ("q4_0", HALVES, "003c809191a2a2b3b3c4c4d5d5e6e6f7f7f8"),
    ("q4_0", TIES, "f0cb80888888888888888888888888888888"),
    ("mxfp4", HALVES, "800e0e1d1d2d2d3c3c4c4b4b5a5a595960"),
    ("mxfp4", TIES, "8307000000000000000000000000000000"),
])
def test_quantizers_match_ggml(dtype, x, ggml):
    """Bytes from ggml's quantize_row_*_ref, ties included."""
    assert getattr(export, dtype)(x).tobytes().hex() == ggml


def test_q8_0_rounds_like_ggml():
    """ggml multiplies by 1 / d; here that rounds differently from x / d."""
    x = np.array([-0.3798859715461731, 1.1904633045196533, -0.45387399196624756, 0.552739679813385, 0.8244103789329529,
                  -0.7386881709098816, 1.2165724039077759, 1.803329586982727, -0.15666693449020386, -0.9455423355102539,
                  0.9239668846130371, -1.5731656551361084, 1.2064008712768555, -0.7623586654663086, 1.5758832693099976,
                  -0.26313507556915283, 1.0155436992645264, 2.7921290397644043, -1.4192817211151123, -0.7748820185661316,
                  0.3686099946498871, 1.7029051780700684, 2.9029390811920166, -1.6679418087005615, 0.14639870822429657,
                  -1.3074840307235718, -1.3833712339401245, 0.25268295407295227, -0.09841255098581314, -1.2458053827285767,
                  -0.05227144435048103, -0.02282484993338585], dtype=np.float32)
    assert export.q8_0(x)[0, 2:].view(np.int8)[21] == 75


def test_q4_0_round_trip():
    x = blocks()
    raw = export.q4_0(x).reshape(len(x), 18)
    d, nib = f16(raw[:, :2]), raw[:, 2:]
    q = np.concatenate([nib & 15, nib >> 4], axis=1).astype(np.float32) - 8
    err = np.abs(q * d[:, None] - x).max(axis=1)
    assert (err <= np.abs(d) + 1e-6).all()
    assert (q[np.arange(len(x)), np.abs(x).argmax(axis=1)] == -8).all()


def test_mxfp4_round_trip():
    x = blocks()
    raw = export.mxfp4(x).reshape(len(x), 17)
    e, nib = raw[:, 0].astype(np.int32), raw[:, 1:]
    q = np.concatenate([nib & 15, nib >> 4], axis=1)
    d = np.ldexp(0.5, e - 127)
    got = export.LUT[q] * d[:, None]
    amax = np.abs(x).max(axis=1)
    bound = np.maximum(2 * d, amax - 12 * d)
    assert (np.abs(got - x).max(axis=1) <= bound + 1e-6).all()
    assert np.abs(got - x).mean() < 0.15 * amax.mean()


def test_tensor_dtype_rules():
    m = SimpleNamespace(flags=0, tensors={"embed": np.zeros((4, 32)), "norm": np.zeros(32), "blk.0.wq": np.zeros((32, 32)),
                                          "pos_embed": np.zeros((2, 32)), "lm_head": np.zeros((4, 32))})
    assert export.tensor_dtype("norm", m, "q4_0", "q8_0") == "f32"
    assert export.tensor_dtype("pos_embed", m, "q4_0", "q8_0") == "f32"
    assert export.tensor_dtype("embed", m, "q4_0", "q8_0") == "f32"
    assert export.tensor_dtype("lm_head", m, "q4_0", "q8_0") == "q8_0"
    assert export.tensor_dtype("blk.0.wq", m, "q4_0", "q8_0") == "q4_0"
    m.flags = export.archs.TIED
    assert export.tensor_dtype("embed", m, "q4_0", "q8_0") == "q8_0"


def test_quantized_needs_k_multiple_of_32():
    with pytest.raises(ValueError):
        export.encode(np.zeros((2, 40), dtype=np.float32), "q8_0")


@pytest.mark.parametrize("spec", SPECS)
def test_header_and_toc(exports, spec):
    path = exports[(spec, "q8_0")]
    raw = np.fromfile(path, dtype=np.uint8)
    magic, version, arch, flags, dim, layers, heads, kv, ffn, vocab, ctx = struct.unpack_from("<4sI16sIIIIIIII", raw, 0)
    assert magic == b"l3m\0" and version == 1 and arch.rstrip(b"\0") == spec.split("_")[1].encode()
    assert (dim, layers, heads, vocab, ctx) == (64, 2, 4, 260, 64)
    n_tensors, _, toc_off, tok_off = struct.unpack_from("<IIQQ", raw, 64)
    assert 1024 <= toc_off < tok_off <= len(raw)
    for i in range(n_tensors):
        name, dtype, ndim, r0, r1, _, _, off, nbytes = struct.unpack_from("<64sIIQQQQQQ", raw, toc_off + i * 120)
        assert off % 64 == 0 and off + nbytes <= toc_off
        if ndim == 2 and dtype == 2:
            assert nbytes == r0 * (r1 // 32) * 34


def dtypes(path):
    raw = path.read_bytes()
    return {name: struct.unpack_from("<I", raw, e + 64)[0] for name, e in toc(raw).items()}


def test_cli_head_dtype(tmp_path, monkeypatch):
    out = tmp_path / "m.l3m"
    export.main([str(MODELS / "tiny_llama.py"), "--dtype", "q4_0", "-o", str(out)])
    assert dtypes(out)["blk.1.w_down"] == export.DTYPES["q4_0"] and dtypes(out)["lm_head"] == export.DTYPES["q8_0"]
    export.main([str(MODELS / "tiny_llama.py"), "--dtype", "q4_0", "--head-dtype", "mxfp4", "-o", str(out)])
    assert dtypes(out)["lm_head"] == export.DTYPES["mxfp4"]
    monkeypatch.chdir(tmp_path)
    export.main([str(MODELS / "tiny_gpt2.py")])
    assert dtypes(tmp_path / "tiny_gpt2.l3m")["embed"] == export.DTYPES["f32"]


def test_cli_errors(tmp_path):
    with pytest.raises(SystemExit, match="no model spec"):
        export.main(["no_such_model"])
    with pytest.raises(SystemExit, match="no directory"):
        export.main([str(MODELS / "tiny_llama.py"), "-o", str(tmp_path / "missing" / "m.l3m")])


@pytest.mark.parametrize("name", sorted(p.stem for p in (ROOT / "models").glob("*.py")))
def test_model_specs(name):
    s = export.load_spec(name)
    assert callable(getattr(archs, s.source)) and isinstance(s.repo, str)
    assert s.dtype in export.DTYPES and s.head_dtype in export.DTYPES
    assert s.__doc__ and s.__doc__.startswith(f"{name}:")
