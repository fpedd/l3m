#!/usr/bin/env python3
"""Compare l3m against an independent reference: numpy fp64, or transformers for HF repos.

    uv run bench/vs_hf.py [stories15M] [--dtype bf16 | --file model.l3m] [--prompt ...] [--new 32]

Passes if the export matches transformers (mean KL < 1e-8) and both paths are within 1e-3 nats of
the reference, or for quantized weights the fast path is within 10% (+1e-3) of the reference path's KL.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

import l3m
from l3m import archs, export

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- numpy reference

def rmsnorm(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def layernorm(x, w, b, eps):
    x = x - x.mean(-1, keepdims=True)
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w + b


def rope(x, theta):   # x [T, H, hd], adjacent pairs
    T, _, hd = x.shape
    freq = theta ** (-np.arange(0, hd, 2) / hd)
    ang = np.arange(T)[:, None] * freq[None, :]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    re, im = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2], out[..., 1::2] = re * c - im * s, re * s + im * c
    return out


def bf16(x):
    return (export.to_bf16(x).astype(np.uint32) << 16).view(np.float32)


def numpy_forward(m, ids, bf16_kv=False):
    """Float64 logits [T, vocab] over canonical tensors; bf16_kv rounds K and V as l3m's cache does."""
    t, T = m.tensors, len(ids)
    H, KV, hd = m.n_heads, m.n_kv_heads, m.dim // m.n_heads
    gpt2 = m.flags & archs.LAYERNORM

    def norm(x, w):
        return layernorm(x, t[w], t[w + "_b"], m.norm_eps) if gpt2 else rmsnorm(x, t[w], m.norm_eps)

    def linear(x, w, b=None):
        return x @ t[w].T + t.get(b, 0)

    x = t["embed"][ids].astype(np.float64) + (t["pos_embed"][:T] if gpt2 else 0)
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for i in range(m.n_layers):
        b = f"blk.{i}."
        h = norm(x, b + "attn_norm")
        q, k, v = (linear(h, b + w, b + "b" + w[1]).reshape(T, -1, hd) for w in ("wq", "wk", "wv"))
        if not gpt2:
            q, k = rope(q, m.rope_theta), rope(k, m.rope_theta)
        if bf16_kv:
            k, v = bf16(k), bf16(v)
        k, v = np.repeat(k, H // KV, axis=1), np.repeat(v, H // KV, axis=1)
        att = np.einsum("thd,shd->hts", q, k) / np.sqrt(hd) + mask
        att = np.exp(att - att.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        x = x + linear(np.einsum("hts,shd->thd", att, v).reshape(T, -1), b + "wo", b + "bo")
        h = norm(x, b + "ffn_norm")
        u = linear(h, b + "w_up", b + "b_up")
        if gpt2:
            u = 0.5 * u * (1 + np.tanh(np.sqrt(2 / np.pi) * (u + 0.044715 * u ** 3)))
        else:
            g = linear(h, b + "w_gate")
            u = g / (1 + np.exp(-g)) * u
        x = x + linear(u, b + "w_down", b + "b_down")
    return linear(norm(x, "norm"), "lm_head" if "lm_head" in t else "embed")


def hf_forward(repo, ids):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32)
    with torch.no_grad():
        return model(torch.tensor([ids])).logits[0].double().numpy()


# ---------------------------------------------------------------- comparison

def log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def l3m_logits(model, ids):
    """Logits at every position, one prefix at a time."""
    rows = []
    for i in range(len(ids)):
        model.generate(ids[:i + 1], 0)
        rows.append(model.logits.astype(np.float64))
    return np.stack(rows)


def compare(name, ref, got):
    lr, lg = log_softmax(ref), log_softmax(got)
    kl = (np.exp(lr) * (lr - lg)).sum(-1).mean()
    top1 = (ref.argmax(-1) == got.argmax(-1)).mean()
    print(f"  {name:10s} max|dlogit| {np.abs(ref - got).max():9.5f}   top-1 agree {top1:5.1%}   mean KL {kl:.2e}")
    return kl


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("model", nargs="?", default="stories15M", help="name of a file in models/, or a path to a spec file")
    p.add_argument("--dtype", choices=export.DTYPES, help="weight dtype, default from the model file")
    p.add_argument("--prompt", default="Once upon a time, there was a little girl named Lily.")
    p.add_argument("--new", type=int, default=32, help="greedy tokens to check")
    p.add_argument("--file", help=".l3m to test, default <model>[-<dtype>].l3m (exported if missing)")
    a = p.parse_args(argv)

    spec = export.load_spec(a.model)
    path = Path(a.file) if a.file else ROOT / f"{Path(a.model).stem}{f'-{a.dtype}' if a.dtype else ''}.l3m"
    if not path.exists():
        export.export(a.model, a.dtype or spec.dtype, out=path)
    fast, slow = l3m.Model(path, force=True), l3m.Model(path, reference=True)
    print(fast.describe())
    dtype = re.search(r": (\w+), \d+ layers", fast.describe())[1]

    hf = spec.source not in ("llama2c", "random")
    if hf:
        from transformers import AutoTokenizer
        ids = AutoTokenizer.from_pretrained(spec.repo)(a.prompt)["input_ids"]
    else:
        ids = fast.encode(a.prompt)

    gen = fast.generate(ids, max(0, min(a.new, fast.ctx - 1 - len(ids))), temperature=0.0)
    seq = ids + gen
    ref = numpy_forward(export.load_model(spec), seq)
    print(f"prompt {len(ids)} tokens + {len(gen)} greedy: {fast.decode(seq)!r}")
    ok = True
    if hf:
        ref, exported = hf_forward(spec.repo, seq), ref
        ok = compare("export", ref, exported) < 1e-8
    print(f"  greedy agree {(ref[len(ids) - 1:-1].argmax(-1) == np.array(gen)).mean():5.1%}")
    lr, lf = l3m_logits(slow, seq), l3m_logits(fast, seq)
    kl_ref, kl_fast = compare("reference", ref, lr), compare("fast", ref, lf)
    compare("fast/ref", lr, lf)
    ok &= bool(max(kl_ref, kl_fast) < 1e-3 if dtype == "f32" else kl_fast <= 1.1 * kl_ref + 1e-3)
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
