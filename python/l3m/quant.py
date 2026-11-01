"""Calibrated q4_0: per-input-channel scales folded exactly into the model (AWQ), then GPTQ on the q4_0 grid.

Input statistics come from text the model writes itself, through a numpy fp64 forward (also bench/vs_hf.py's reference).
"""
import numpy as np

from . import Model, archs, export

DAMP, ALPHAS, NMAX = 0.01, np.linspace(0, 1, 21), np.linspace(7.5, 9.5, 41)


def rmsnorm(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def layernorm(x, w, b, eps):
    x = x - x.mean(-1, keepdims=True)
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w + b


def rope(x, theta):   # x [..., T, H, hd], adjacent pairs
    T, _, hd = x.shape[-3:]
    freq = theta ** (-np.arange(0, hd, 2) / hd)
    ang = np.arange(T)[:, None] * freq[None, :]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    re, im = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2], out[..., 1::2] = re * c - im * s, re * s + im * c
    return out


def bf16(x):
    return (export.to_bf16(x).astype(np.uint32) << 16).view(np.float32)


def forward(m, ids, bf16_kv=False, stats=None):
    """Float64 logits [..., T, vocab] of ids [..., T] over canonical tensors; bf16_kv rounds K and V as l3m's cache does,
    stats sums x^T x of each matrix's input."""
    ids = np.asarray(ids)
    t, T = m.tensors, ids.shape[-1]
    H, KV, hd = m.n_heads, m.n_kv_heads, m.dim // m.n_heads
    gpt2 = m.flags & archs.LAYERNORM

    def norm(x, w):
        return layernorm(x, t[w], t[w + "_b"], m.norm_eps) if gpt2 else rmsnorm(x, t[w], m.norm_eps)

    def linear(x, w, b=None):
        if stats is not None and not w.endswith(("wk", "wv", "w_gate")):   # same input as wq, w_up
            stats[w] = stats.get(w, 0) + (x2 := x.reshape(-1, x.shape[-1])).T @ x2
        return x @ t[w].T + t.get(b, 0)

    x = t["embed"][ids].astype(np.float64) + (t["pos_embed"][:T] if gpt2 else 0)
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for i in range(m.n_layers):
        b = f"blk.{i}."
        h = norm(x, b + "attn_norm")
        q, k, v = (linear(h, b + w, b + "b" + w[1]).reshape(*ids.shape, -1, hd) for w in ("wq", "wk", "wv"))
        if not gpt2:
            q, k = rope(q, m.rope_theta), rope(k, m.rope_theta)
        if bf16_kv:
            k, v = bf16(k), bf16(v)
        k, v = np.repeat(k, H // KV, axis=-2), np.repeat(v, H // KV, axis=-2)
        att = np.einsum("...thd,...shd->...hts", q, k, optimize=True) / np.sqrt(hd) + mask
        att = np.exp(att - att.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        x = x + linear(np.einsum("...hts,...shd->...thd", att, v, optimize=True).reshape(*ids.shape, -1), b + "wo", b + "bo")
        h = norm(x, b + "ffn_norm")
        u = linear(h, b + "w_up", b + "b_up")
        if gpt2:
            u = 0.5 * u * (1 + np.tanh(np.sqrt(2 / np.pi) * (u + 0.044715 * u ** 3)))
        else:
            g = linear(h, b + "w_gate")
            u = g / (1 + np.exp(-g)) * u
        x = x + linear(u, b + "w_down", b + "b_down")
    return linear(norm(x, "norm"), "lm_head" if "lm_head" in t else "embed")


def f16(x):
    return x.astype(np.float16).astype(np.float64)


def levels(b, d):
    return np.clip(np.round(b / np.where(d == 0, 1, d)), -8, 7)


def rtn(w):
    b = w.reshape(-1, 32)
    d = f16(b[np.arange(len(b)), np.abs(b).argmax(1)] / -8)[:, None]
    return (levels(b, d) * d).reshape(w.shape)


def scale(b, w):
    """Per row of 32 inputs the f16 d minimising sum w (b - d q)^2: max/-8 nearby, each refit twice by least squares."""
    mx = b[np.arange(len(b)), np.abs(b).argmax(1)]
    best, best_d = np.full(len(b), np.inf), np.zeros(len(b))
    for nmax in NMAX:
        d = mx / -nmax
        for _ in range(3):
            d = f16(d)
            q = levels(b, d[:, None])
            err = (w * (b - q * d[:, None]) ** 2).sum(1)
            best_d, best = np.where(err < best, d, best_d), np.minimum(err, best)
            s2 = (w * q * q).sum(1)
            d = np.where(s2 > 0, (w * q * b).sum(1) / np.where(s2 > 0, s2, 1), d)
    return best_d


def gptq(W, H, B=128):
    """q4_0 values of W [rows, n] and their d [rows, n / 32], each column's rounding error compensated in the later ones."""
    W, H = W.astype(np.float64), H.copy()
    dead = np.diag(H) == 0
    H[dead, dead], W[:, dead] = 1, 0
    H += DAMP * np.diag(H).mean() * np.eye(len(H))
    U, hd = np.linalg.cholesky(np.linalg.inv(H)).T, np.diag(H)   # upper, U^T U = H^-1
    Q, D = np.zeros_like(W), np.zeros((len(W), W.shape[1] // 32))
    for i in range(0, W.shape[1], B):
        W1 = W[:, i:i + B]
        E = np.zeros_like(W1)
        for j in range(W1.shape[1]):
            c = i + j
            if c % 32 == 0:
                d = D[:, c // 32] = scale(W1[:, j:j + 32], hd[c:c + 32])
            Q[:, c] = levels(W1[:, j], d) * d
            E[:, j] = (W1[:, j] - Q[:, c]) / U[c, c]
            W1[:, j:] -= np.outer(E[:, j], U[c, c:i + B])
        W[:, i + B:] -= E @ U[i:i + B, i + B:]
    return Q, D


def awq(t, h, mats, producers, hd):
    """Multiplies the input columns of mats by s = rms(x)^alpha, alpha the best of ALPHAS on the q4_0 output error
    tr(E H E^T), and divides the producers of x by s. Returns H of the scaled x."""
    c = np.arange(len(h))
    idx = c // (len(h) // len(t[producers[0]]) * hd) * hd + c % hd   # input channel -> producer channel (kv for wo)
    ms = np.bincount(idx, np.diag(h)) / np.bincount(idx)

    def err(s):
        return sum((((e := t[w] - rtn(t[w] * s[idx]) / s[idx]) @ h) * e).sum() for w in mats)

    def scales(a):
        s = np.maximum(ms / ms.mean(), 1e-8) ** (a / 2)
        return s / np.sqrt(s.max() * s.min())

    s = min(map(scales, ALPHAS), key=err)
    for p in producers:
        t[p] = (t[p] / (s[:, None] if t[p].ndim == 2 else s)).astype(np.float32)
    for w in mats:
        t[w] = (t[w] * s[idx]).astype(np.float32)
    return h / np.outer(s[idx], s[idx])


def groups(m):
    """(statistics key, matrices reading that input, its producers if scaling it is exact), wo and w_down first: their
    folds rescale rows of wv and w_up."""
    out = []
    for i in range(m.n_layers):
        b = f"blk.{i}."
        out += [(b + "wo", [b + "wo"], [b + "wv", b + "bv"]),
                (b + "wq", [b + "wq", b + "wk", b + "wv"], [b + "attn_norm", b + "attn_norm_b"]),
                (b + "w_down", [b + "w_down"], [] if m.flags & archs.GELU else [b + "w_up"]),   # gelu(u / s) != gelu(u) / s
                (b + "w_up", [b + "w_gate", b + "w_up"], [b + "ffn_norm", b + "ffn_norm_b"])]
    head = "lm_head" if "lm_head" in m.tensors else "embed"
    return out + [(head, [head], [])]   # a tied embedding is also gathered, so the final norm cannot absorb a scale


def sample(m, path, rows=64, T=256):
    """rows of BOS and T - 1 tokens of text the model writes itself, through the engine on a q8_0 draft at path."""
    export.write(m, "q8_0", "q8_0", path)
    with Model(path, force=True) as lm:
        T, text = min(T, lm.ctx), []
        while len(text) < rows * (T - 1):
            text += lm.generate([m.bos], T, temperature=1, seed=len(text)) + [m.bos]
    return np.array([[m.bos] + text[i:i + T - 1] for i in range(0, rows * (T - 1), T - 1)])


def calibrate(m, ids, dtype, head_dtype):
    """Replaces m's q4_0 matrices by calibrated q4_0 values and returns their d."""
    H, t, d = {}, m.tensors, {}
    for i in range(0, len(ids), 4):
        forward(m, ids[i:i + 4], stats=H)
    for key, mats, producers in groups(m):
        mats = [w for w in mats if w in t and export.tensor_dtype(w, m, dtype, head_dtype) == "q4_0"]
        if not mats:
            continue
        h = awq(t, H[key], mats, [p for p in producers if p in t], m.dim // m.n_heads) if producers else H[key]
        Q, D = gptq(np.concatenate([t[w] for w in mats]), h)
        rows = np.cumsum([len(t[w]) for w in mats])[:-1]
        for w, q, s in zip(mats, np.split(Q, rows), np.split(D, rows)):
            t[w], d[w] = q.astype(np.float32), s.astype(np.float32)
    return d
