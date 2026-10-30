#!/usr/bin/env python3
"""Perplexity of each dtype on the same tokens, against f32: PPL change, KL divergence, top-token agreement.

    uv run bench/ppl.py stories15M --text TinyStories-valid.txt [--dtypes f32,q4_0] [--chunks 16] [--ctx 128] [--hf]

Each chunk of --ctx tokens, BOS first, is scored on its second half, as llama.cpp's perplexity does.
--hf scores the same tokens with transformers f32 as the base.
"""
import argparse
import time
from pathlib import Path

import numpy as np

import l3m
from l3m import export

ROOT = Path(__file__).resolve().parents[1]


def log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def make_chunks(m, text, ctx, n):
    bos = [m.bos] if m.bos >= 0 else []
    step = ctx - len(bos)
    ids = m.encode(text[:8 * ctx * n], bos=False)
    return [bos + ids[i:i + step] for i in range(0, step * n, step) if i + step <= len(ids)]


def l3m_log_probs(m, chunk):
    """Log-probabilities at the scored positions: row j predicts chunk[len(chunk) // 2 + j]."""
    rows = []
    for i in range(len(chunk) // 2, len(chunk)):
        m.generate(chunk[:i], 0)
        rows.append(m.logits)
    return log_softmax(np.array(rows, dtype=np.float64)).astype(np.float32)


def hf_log_probs(model, chunk):
    import torch
    with torch.no_grad():
        z = model(torch.tensor([chunk])).logits[0].double().numpy()
    return log_softmax(z[len(chunk) // 2 - 1:-1]).astype(np.float32)


def row(name, lp, base, targets):
    nll = -np.take_along_axis(lp, targets[:, None], 1)[:, 0]
    line = f"  {name:6s} PPL {np.exp(nll.mean()):9.3f}"
    if base is not None:
        b_nll = -np.take_along_axis(base, targets[:, None], 1)[:, 0]
        d = nll - b_nll
        ratio, se = np.exp(d.mean()), np.exp(d.mean()) * d.std(ddof=1) / np.sqrt(len(d))
        kld = (np.exp(base) * (base - lp)).sum(-1)
        top = (base.argmax(-1) == lp.argmax(-1)).mean()
        line += (f"   dPPL {100 * (ratio - 1):+7.3f} ± {100 * se:.3f} %   KLD {kld.mean():.5f} ± {kld.std(ddof=1) / np.sqrt(len(kld)):.5f}"
                 f"   same top {100 * top:6.2f} %")
    print(line, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("model", help="name of a file in models/, or a path to a spec file")
    p.add_argument("--text", required=True, help="plain-text corpus, e.g. wikitext-2-raw/wiki.test.raw")
    p.add_argument("--dtypes", default=",".join(export.DTYPES), help="comma-separated; the first is the base without --hf")
    p.add_argument("--chunks", type=int, default=16)
    p.add_argument("--ctx", type=int, default=128)
    p.add_argument("--dir", default=".", help="where the exports go, reused if present")
    p.add_argument("--hf", action="store_true", help="transformers f32 as the base")
    a = p.parse_args(argv)

    dtypes = a.dtypes.split(",")
    paths = {d: Path(a.dir) / f"{Path(a.model).stem}-{d}.l3m" for d in dtypes}
    for d, path in paths.items():
        if not path.exists():
            export.export(a.model, d, out=path)
    text = Path(a.text).read_text(encoding="utf-8").replace("<|endoftext|>", "\n")
    with l3m.Model(paths[dtypes[0]], force=True) as m:
        chunks = make_chunks(m, text, a.ctx, a.chunks)
    targets = np.concatenate([c[len(c) // 2:] for c in chunks])
    print(f"{a.model}: {len(chunks)} chunks of {a.ctx}, {len(targets)} scored tokens of {a.text}", flush=True)

    base = None
    if a.hf:
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(export.load_spec(a.model).repo, torch_dtype=torch.float32)
        base = np.concatenate([hf_log_probs(model, c) for c in chunks])
        row("hf", base, None, targets)
    for d in dtypes:
        t0 = time.time()
        with l3m.Model(paths[d], force=True) as m:
            lp = np.concatenate([l3m_log_probs(m, c) for c in chunks])
        row(d, lp, base, targets)
        print(f"         ({time.time() - t0:.0f} s)", flush=True)
        base = lp if base is None else base


if __name__ == "__main__":
    main()
