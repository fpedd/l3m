"""Export a model from models/<name>.py to a .l3m file (layout: src/model.h, blocks: ggml's).

    uv run l3m-export stories15M [--dtype f32|bf16|q8_0|q4_0|mxfp4] [--head-dtype ...] [-o out.l3m]
"""
import argparse
import importlib.util
import struct
from pathlib import Path

import numpy as np

from . import archs

ROOT = Path(__file__).resolve().parents[2]
DTYPES = {"f32": 0, "bf16": 1, "q8_0": 2, "q4_0": 3, "mxfp4": 4}
HEAD_DTYPE = {"f32": "f32", "bf16": "bf16", "q8_0": "bf16", "q4_0": "q8_0", "mxfp4": "q8_0"}
LUT = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)
MiB = 1 << 20


def roundf(x):  # C roundf: half away from zero
    return np.sign(x) * np.floor(np.abs(x).astype(np.float64) + 0.5)


# ---------------------------------------------------------------- quantizers

def to_bf16(x):
    u = x.astype(np.float32).view(np.uint32)
    return np.where(np.isnan(x), (u >> 16) | 0x40, (u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def inverse(d):   # ggml's x * (1 / d), which rounds differently from x / d
    return np.where(d != 0, np.float32(1) / np.where(d != 0, d, 1), 0).astype(np.float32)


def q8_0(x):
    b = x.astype(np.float32).reshape(-1, 32)
    d = np.abs(b).max(axis=1) / np.float32(127)
    q = roundf(b * inverse(d)[:, None]).astype(np.int8)
    return np.concatenate([d.astype(np.float16).view(np.uint8).reshape(-1, 2), q.view(np.uint8)], axis=1)


def q4_0(x):
    b = x.astype(np.float32).reshape(-1, 32)
    imax = np.abs(b).argmax(axis=1)
    d = b[np.arange(len(b)), imax] / np.float32(-8)
    q = np.clip(np.floor(b * inverse(d)[:, None] + np.float32(8.5)), 0, 15).astype(np.uint8)
    return np.concatenate([d.astype(np.float16).view(np.uint8).reshape(-1, 2), q[:, :16] | (q[:, 16:] << 4)], axis=1)


def mxfp4(x):
    b = x.reshape(-1, 32)
    amax = np.abs(b).max(axis=1)
    e = np.where(amax > 0, np.floor(np.log2(np.where(amax > 0, amax, 1))) - 2 + 127, 0).astype(np.uint8)
    d = np.ldexp(0.5, e.astype(np.int32) - 127)   # E8M0_TO_FP32_HALF
    q = np.abs(b[:, :, None] - LUT * d[:, None, None]).argmin(axis=2).astype(np.uint8)
    return np.concatenate([e[:, None], q[:, :16] | (q[:, 16:] << 4)], axis=1)


def encode(x, dtype):
    if dtype == "f32":
        return x.astype(np.float32).tobytes()
    if dtype == "bf16":
        return to_bf16(x).tobytes()
    if x.shape[-1] % 32:
        raise ValueError(f"{dtype}: inputs {x.shape[-1]} not a multiple of 32")
    return {"q8_0": q8_0, "q4_0": q4_0, "mxfp4": mxfp4}[dtype](x).tobytes()


# ---------------------------------------------------------------- writer

def tensor_dtype(name, m, dtype, head_dtype):
    """Matrices take the body dtype, the lm_head (the embed when tied) its own, vectors and gathers f32."""
    if m.tensors[name].ndim == 1 or name == "pos_embed":
        return "f32"
    if name == "lm_head" or (name == "embed" and m.flags & archs.TIED):
        return head_dtype
    return "f32" if name == "embed" else dtype


def write(m, dtype, head_dtype, out):
    names = sorted(m.tensors)
    dt = {n: tensor_dtype(n, m, dtype, head_dtype) for n in names}
    blobs = {n: encode(m.tensors[n], dt[n]) for n in names}
    max_len = max(len(b) for b, _ in m.tokens)
    with open(out, "wb") as f:
        f.write(b"\0" * 1024)
        toc = []
        for n in names:
            f.seek((f.tell() + 63) // 64 * 64)
            t = m.tensors[n]
            toc.append(struct.pack("<64sIIQQQQQQ", n.encode(), DTYPES[dt[n]], t.ndim,
                                   *(list(t.shape) + [0] * (4 - t.ndim)), f.tell(), len(blobs[n])))
            f.write(blobs[n])
        f.seek((f.tell() + 7) // 8 * 8)
        toc_offset = f.tell()
        f.write(b"".join(toc))
        tok_offset = f.tell()
        f.write(struct.pack("<IiiI", len(m.tokens), m.bos, m.eos, max_len))
        f.write(b"".join(struct.pack("<fH", s, len(b)) + b for b, s in m.tokens))
        f.seek(0)
        f.write(struct.pack("<4sI16sIIIIIIIIffIIQQ", b"l3m\0", 1, m.arch.encode(), m.flags, m.dim, m.n_layers,
                            m.n_heads, m.n_kv_heads, m.ffn_dim, m.vocab, m.ctx, m.rope_theta, m.norm_eps,
                            len(names), 0, toc_offset, tok_offset))
    return {n: len(blobs[n]) for n in names}


def summary(m, sizes, out):
    groups = {"embed": 0, "pos_embed": 0, "attention": 0, "mlp": 0, "norms": 0, "lm_head": 0}
    for n, nb in sizes.items():
        key = n if n in groups else "attention" if any(k in n for k in ("wq", "wk", "wv", "wo", "bq", "bk", "bv", "bo")) \
            else "mlp" if "w_" in n or "b_" in n else "norms"
        groups[key] += nb
    resident = sum(groups.values()) - groups["pos_embed"] - (groups["embed"] if not m.flags & archs.TIED else 0)
    print(f"{out}: {m.arch}, dim {m.dim}, {m.n_layers} layers, {m.n_heads} heads ({m.n_kv_heads} kv), ffn {m.ffn_dim}, vocab {m.vocab}")
    for k, v in groups.items():
        if v:
            print(f"  {k:10s} {v / MiB:8.2f} MiB")
    print(f"  {'resident':10s} {resident / MiB:8.2f} MiB   plus the KV cache")


def load_spec(name):
    """A name in models/, or a path to a spec file."""
    path = Path(name) if name.endswith(".py") or "/" in name else ROOT / "models" / f"{name}.py"
    if not path.exists():
        raise SystemExit(f"no model spec {path}: a name in {ROOT / 'models'}, or a path to a spec file")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model(spec):
    return archs.random(spec) if spec.source == "random" else getattr(archs, spec.source)(spec.repo, spec.weights)


def export(name, dtype=None, head_dtype=None, out=None):
    spec = load_spec(name)
    dtype = dtype or spec.dtype
    head_dtype = head_dtype or (spec.head_dtype if dtype == spec.dtype else HEAD_DTYPE[dtype])
    out = Path(out or f"{Path(name).stem}.l3m")
    if not out.parent.is_dir():
        raise SystemExit(f"no directory {out.parent} for {out}")
    m = load_model(spec)
    summary(m, write(m, dtype, head_dtype, out), out)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("model", help="name of a file in models/, or a path to a spec file")
    p.add_argument("--dtype", choices=DTYPES, help="weight dtype, default from the model file")
    p.add_argument("--head-dtype", choices=DTYPES, help="lm_head dtype, default from the model file, else finer than a quantized --dtype")
    p.add_argument("-o", "--out", help="output path, default <model>.l3m")
    a = p.parse_args(argv)
    export(a.model, a.dtype, a.head_dtype, a.out)


if __name__ == "__main__":
    main()
