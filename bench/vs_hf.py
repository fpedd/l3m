#!/usr/bin/env python3
"""Compare l3m against an independent reference: numpy fp64, or transformers for HF repos.

    uv run bench/vs_hf.py [stories15M] [--dtype bf16 | --file model.l3m] [--prompt ...] [--new 32]

Passes if the export matches transformers (mean KL < 1e-8) and both paths are within 1e-3 nats of
the reference, or for quantized weights the fast path is within 0.03 nats of the reference path.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

import l3m
from l3m import export, quant

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- transformers

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
    ref = quant.forward(export.load_model(spec), seq)
    print(f"prompt {len(ids)} tokens + {len(gen)} greedy: {fast.decode(seq)!r}")
    ok = True
    if hf:
        ref, exported = hf_forward(spec.repo, seq), ref
        ok = compare("export", ref, exported) < 1e-8
    print(f"  greedy agree {(ref[len(ids) - 1:-1].argmax(-1) == np.array(gen)).mean():5.1%}")
    lr, lf = l3m_logits(slow, seq), l3m_logits(fast, seq)
    kl_ref, kl_fast = compare("reference", ref, lr), compare("fast", ref, lf)
    kl_both = compare("fast/ref", lr, lf)
    ok &= bool(max(kl_ref, kl_fast) < 1e-3 if dtype == "f32" else kl_both < 0.03)
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
