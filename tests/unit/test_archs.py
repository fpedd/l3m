import json
import struct
from pathlib import Path

import numpy as np

from l3m import archs


def piece(text, score, kind=1):
    s = text.encode()
    body = b"\x0a" + bytes([len(s)]) + s + b"\x15" + struct.pack("<f", score) + b"\x18" + bytes([kind])
    return b"\x0a" + bytes([len(body)]) + body


def test_sentencepiece(tmp_path):
    path = tmp_path / "tokenizer.model"
    trainer = b"\x12\x02\x18\x02"
    path.write_bytes(piece("<s>", 0.0, 3) + trainer + piece("<0x2C>", 0.0, 6) + piece(",", -5.0) + piece("▁a", -1.0))
    tokens = archs.sentencepiece(path, vocab=6)
    assert tokens[0] == (b"<s>", archs.NO_MERGE)
    assert tokens[1] == (b",", archs.BYTE)
    assert tokens[2] == (b",", -5.0) and tokens[3] == (b" a", -1.0)
    assert tokens[4:] == [(b"", archs.NO_MERGE)] * 2


def test_real_piece_outranks_byte_fallback():
    tok = {"model": {"vocab": {"▁": 0, "<0x2C>": 1, ",": 2, "▁a": 3}, "merges": []}}
    tokens, flags = archs.hf_tokenizer(tok, 4)
    assert flags == archs.SP_PREFIX and tokens[1][0] == tokens[2][0] == b","
    assert tokens[2][1] > tokens[1][1]


def test_torch_bin():
    """pytorch_model.bin holds want as f32 (a transposed view), f16, bf16 and bool tensors."""
    x = np.arange(24, dtype=np.float32).reshape(4, 6)
    want = {"w": x.T, "h": x[1:], "b": x[:, 2], "mask": np.ones(3, dtype=np.float32)}
    got = archs.torch_bin(Path(__file__).parent / "pytorch_model.bin")
    assert got.keys() == want.keys()
    for k, v in want.items():
        assert got[k].dtype == np.float32 and np.array_equal(got[k], v)


def test_unpermute_turns_rotate_half_into_adjacent_pairs():
    H, hd, dim = 3, 8, 16
    rng = np.random.default_rng(0)
    wq, wk, x, y = (rng.standard_normal(shape) for shape in ((H * hd, dim), (H * hd, dim), dim, dim))
    freq = 10000.0 ** (-np.arange(0, hd, 2) / hd)

    def rotate_half(v, pos):
        v, c, s = v.reshape(H, hd), np.tile(np.cos(pos * freq), 2), np.tile(np.sin(pos * freq), 2)
        return v * c + np.concatenate([-v[:, hd // 2:], v[:, :hd // 2]], axis=1) * s

    def adjacent(v, pos):
        v, c, s = v.reshape(H, hd).copy(), np.cos(pos * freq), np.sin(pos * freq)
        re, im = v[:, 0::2].copy(), v[:, 1::2].copy()
        v[:, 0::2], v[:, 1::2] = re * c - im * s, re * s + im * c
        return v

    for pq, pk in [(0, 0), (5, 2), (37, 36)]:
        hf = (rotate_half(wq @ x, pq) * rotate_half(wk @ y, pk)).sum(1)
        ours = (adjacent(archs.unpermute(wq, H) @ x, pq) * adjacent(archs.unpermute(wk, H) @ y, pk)).sum(1)
        assert np.allclose(hf, ours)


def test_safetensors(tmp_path):
    x = np.arange(-6, 6, dtype=np.float32).reshape(3, 4) / 4
    blobs = {"a": ("F32", x.tobytes()), "b": ("F16", x.astype(np.float16).tobytes()),
             "c": ("BF16", (x.view(np.uint32) >> 16).astype(np.uint16).tobytes())}
    header, off = {"__metadata__": {"format": "pt"}}, 0
    for k, (dt, b) in blobs.items():
        header[k], off = {"dtype": dt, "shape": [3, 4], "data_offsets": [off, off + len(b)]}, off + len(b)
    h = json.dumps(header).encode()
    (tmp_path / "m.safetensors").write_bytes(struct.pack("<Q", len(h)) + h + b"".join(b for _, b in blobs.values()))
    got = archs.safetensors(tmp_path / "m.safetensors")
    assert got.keys() == blobs.keys() and all(np.array_equal(v, x) for v in got.values())
