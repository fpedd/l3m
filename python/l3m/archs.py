"""Weight loaders, one per source, each returning a Model of canonical f32 tensors (names: src/model.h).

Matrices are [outputs][inputs]; RoPE uses llama2.c's adjacent pairs, so HF's rotate_half is un-permuted.
"""
import hashlib
import json
import pickle
import struct
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

# Flags from src/model.h
LAYERNORM, LEARNED_POS, GELU, TIED, SP_PREFIX, DIGITS = 1, 2, 4, 8, 16, 32
NO_MERGE = -1e9  # score of a token no merge creates
BYTE = -2e9      # byte fallback, below NO_MERGE so a real piece with the same bytes wins


@dataclass
class Model:
    arch: str
    flags: int
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    ffn_dim: int
    vocab: int
    ctx: int
    rope_theta: float
    norm_eps: float
    tensors: dict = field(default_factory=dict)   # name -> float32 array
    tokens: list = field(default_factory=list)    # (bytes, score) per id
    bos: int = -1
    eos: int = -1


# ---------------------------------------------------------------- llama2.c .bin

def llama2c(repo, weights):
    """Karpathy's legacy .bin: 7 int32 header, then fp32 weights in export order."""
    path = hf_hub_download(repo, weights)
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    dim, ffn, L, H, KV, vocab, ctx = struct.unpack("<7i", raw[:28].tobytes())
    tied, vocab = vocab > 0, abs(vocab)
    hd, kvd = dim // H, dim // H * KV
    f = np.frombuffer(raw[28:], dtype=np.float32)
    pos = 0

    def take(*shape):
        nonlocal pos
        n = int(np.prod(shape))
        out = f[pos:pos + n].reshape(shape).astype(np.float32)
        pos += n
        return out

    m = Model("llama", TIED if tied else 0, dim, L, H, KV, ffn, vocab, ctx, 10000.0, 1e-5)
    t = m.tensors
    t["embed"] = take(vocab, dim)
    attn_norm, wq, wk, wv, wo = take(L, dim), take(L, dim, dim), take(L, kvd, dim), take(L, kvd, dim), take(L, dim, dim)
    ffn_norm, w_gate, w_down, w_up = take(L, dim), take(L, ffn, dim), take(L, dim, ffn), take(L, ffn, dim)
    t["norm"] = take(dim)
    take(ctx, hd // 2); take(ctx, hd // 2)   # freq_cis, recomputed by the engine
    if not tied:
        t["lm_head"] = take(vocab, dim)
    for i in range(L):
        t[f"blk.{i}.attn_norm"], t[f"blk.{i}.wq"], t[f"blk.{i}.wk"], t[f"blk.{i}.wv"], t[f"blk.{i}.wo"] = attn_norm[i], wq[i], wk[i], wv[i], wo[i]
        t[f"blk.{i}.ffn_norm"], t[f"blk.{i}.w_gate"], t[f"blk.{i}.w_up"], t[f"blk.{i}.w_down"] = ffn_norm[i], w_gate[i], w_up[i], w_down[i]
    m.tokens, m.bos, m.eos = llama2c_tokenizer(vocab), 1, 2
    m.flags |= SP_PREFIX
    return m


TOKENIZER_BIN = "https://raw.githubusercontent.com/karpathy/llama2.c/c02865df300f3bd9e567ce061000dc23bf785a17/tokenizer.bin"
TOKENIZER_SHA256 = "50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"


def llama2c_tokenizer(vocab):
    """llama2.c's tokenizer.bin: max_len, then per token {f32 score, i32 len, bytes}."""
    path = Path.home() / ".cache/l3m/tokenizer.bin"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TOKENIZER_BIN, path)
    data, pos, tokens = path.read_bytes(), 4, []
    if hashlib.sha256(data).hexdigest() != TOKENIZER_SHA256:
        raise ValueError(f"{path} is not llama2.c's tokenizer.bin; delete it to download again")
    for _ in range(vocab):
        score, n = struct.unpack_from("<fi", data, pos)
        piece = data[pos + 8:pos + 8 + n].decode("utf-8")
        pos += 8 + n
        tokens.append(sp_piece(piece, score))
    return tokens


def sp_piece(piece, score):
    """SentencePiece piece -> raw bytes: '▁' is a space, '<0xNN>' is a byte."""
    if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
        return bytes([int(piece[3:5], 16)]), BYTE
    return piece.replace("▁", " ").encode("utf-8"), score


# ---------------------------------------------------------------- HF weights: safetensors or a PyTorch zip

def to_f32(buf, dtype):
    """Raw little-endian F32, F16, BOOL or BF16 (which numpy lacks) -> float32."""
    if dtype == "BF16":
        return (np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    return np.frombuffer(buf, dtype={"F32": np.float32, "F16": np.float16, "BOOL": np.bool_}[dtype]).astype(np.float32)


def safetensors(path):
    """.safetensors -> float32 arrays."""
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    n = struct.unpack("<Q", raw[:8].tobytes())[0]
    header, base = json.loads(raw[8:8 + n].tobytes()), 8 + n
    out = {}
    for name, t in header.items():
        if name == "__metadata__":
            continue
        a, b = t["data_offsets"]
        out[name] = to_f32(raw[base + a:base + b], t["dtype"]).reshape(t["shape"])
    return out


def torch_bin(path):
    """pytorch_model.bin -> float32 arrays, without torch; the unpickler refuses anything but tensors."""
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split("/")[0]
    storage = {"FloatStorage": "F32", "HalfStorage": "F16", "BFloat16Storage": "BF16", "BoolStorage": "BOOL"}

    def tensor(st, offset, shape, stride, *_):
        return np.lib.stride_tricks.as_strided(st[offset:], shape, [s * 4 for s in stride]).copy()

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                return tensor
            if (module, name) == ("collections", "OrderedDict"):
                return dict
            if module == "torch" and name in storage:
                return storage[name]
            raise pickle.UnpicklingError(f"{module}.{name} in {path}")

        def persistent_load(self, pid):   # ("storage", type, key, location, numel)
            return to_f32(z.read(f"{root}/data/{pid[2]}"), pid[1])

    return Unpickler(z.open(f"{root}/data.pkl")).load()


def hf_files(repo, weights):
    cfg = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
    path = hf_hub_download(repo, weights or "model.safetensors")
    return cfg, safetensors(path) if path.endswith(".safetensors") else torch_bin(path)


def unpermute(w, n_heads):
    """HF rotate_half rows [re..., im...] per head -> adjacent pairs [re0, im0, re1, im1, ...]."""
    hd = w.shape[0] // n_heads
    return w.reshape(n_heads, 2, hd // 2, -1).transpose(0, 2, 1, 3).reshape(w.shape)


def hf_llama(repo, weights=None):
    cfg, w = hf_files(repo, weights)
    H, KV = cfg["num_attention_heads"], cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    tied = cfg.get("tie_word_embeddings", False)
    if (cfg.get("hidden_act", "silu") != "silu" or cfg.get("head_dim", cfg["hidden_size"] // H) != cfg["hidden_size"] // H
            or any(cfg.get(k) for k in ("rope_scaling", "attention_bias", "mlp_bias"))):
        raise ValueError(f"{repo}: unsupported llama config")
    m = Model("llama", TIED if tied else 0, cfg["hidden_size"], cfg["num_hidden_layers"], H, KV,
              cfg["intermediate_size"], cfg["vocab_size"], cfg["max_position_embeddings"],
              cfg.get("rope_theta", 10000.0), cfg["rms_norm_eps"])
    t = m.tensors
    t["embed"], t["norm"] = w["model.embed_tokens.weight"], w["model.norm.weight"]
    if not tied:
        t["lm_head"] = w["lm_head.weight"]
    for i in range(m.n_layers):
        p = f"model.layers.{i}."
        t[f"blk.{i}.attn_norm"], t[f"blk.{i}.ffn_norm"] = w[p + "input_layernorm.weight"], w[p + "post_attention_layernorm.weight"]
        t[f"blk.{i}.wq"], t[f"blk.{i}.wk"] = unpermute(w[p + "self_attn.q_proj.weight"], H), unpermute(w[p + "self_attn.k_proj.weight"], KV)
        t[f"blk.{i}.wv"], t[f"blk.{i}.wo"] = w[p + "self_attn.v_proj.weight"], w[p + "self_attn.o_proj.weight"]
        t[f"blk.{i}.w_gate"], t[f"blk.{i}.w_up"], t[f"blk.{i}.w_down"] = w[p + "mlp.gate_proj.weight"], w[p + "mlp.up_proj.weight"], w[p + "mlp.down_proj.weight"]
    m.tokens, tok_flags = hf_tokens(repo, m.vocab)
    m.flags |= tok_flags
    eos = cfg.get("eos_token_id", -1)
    m.bos, m.eos = cfg.get("bos_token_id", -1), eos[0] if isinstance(eos, list) else eos
    return m


def hf_gpt2(repo, weights=None):
    cfg, w = hf_files(repo, weights)
    w = {k.removeprefix("transformer."): v for k, v in w.items()}
    dim, L = cfg["n_embd"], cfg["n_layer"]
    m = Model("gpt2", LAYERNORM | LEARNED_POS | GELU | TIED, dim, L, cfg["n_head"], cfg["n_head"],
              cfg.get("n_inner") or 4 * dim, cfg["vocab_size"], cfg["n_positions"], 0.0, cfg["layer_norm_epsilon"])
    t = m.tensors
    t["embed"], t["pos_embed"] = w["wte.weight"], w["wpe.weight"]
    t["norm"], t["norm_b"] = w["ln_f.weight"], w["ln_f.bias"]
    for i in range(L):
        p = f"h.{i}."
        t[f"blk.{i}.attn_norm"], t[f"blk.{i}.attn_norm_b"] = w[p + "ln_1.weight"], w[p + "ln_1.bias"]
        t[f"blk.{i}.ffn_norm"], t[f"blk.{i}.ffn_norm_b"] = w[p + "ln_2.weight"], w[p + "ln_2.bias"]
        qkv, bqkv = w[p + "attn.c_attn.weight"].T, w[p + "attn.c_attn.bias"]   # Conv1D stores [in, out]
        t[f"blk.{i}.wq"], t[f"blk.{i}.wk"], t[f"blk.{i}.wv"] = np.split(np.ascontiguousarray(qkv), 3)
        t[f"blk.{i}.bq"], t[f"blk.{i}.bk"], t[f"blk.{i}.bv"] = np.split(bqkv, 3)
        t[f"blk.{i}.wo"], t[f"blk.{i}.bo"] = np.ascontiguousarray(w[p + "attn.c_proj.weight"].T), w[p + "attn.c_proj.bias"]
        t[f"blk.{i}.w_up"], t[f"blk.{i}.b_up"] = np.ascontiguousarray(w[p + "mlp.c_fc.weight"].T), w[p + "mlp.c_fc.bias"]
        t[f"blk.{i}.w_down"], t[f"blk.{i}.b_down"] = np.ascontiguousarray(w[p + "mlp.c_proj.weight"].T), w[p + "mlp.c_proj.bias"]
    m.tokens, tok_flags = hf_tokens(repo, m.vocab)
    m.flags |= tok_flags
    m.bos = m.eos = cfg["eos_token_id"]
    return m


def hf_gpt_neo(repo, weights=None):
    """GPT-2 with Linear layers, no q/k/v bias and no 1/sqrt(hd) attention scale (cancelled in wq); ctx is capped at
    window_size so the local layers are exact."""
    cfg, w = hf_files(repo, weights)
    w = {k.removeprefix("transformer."): v for k, v in w.items()}
    dim, L, H = cfg["hidden_size"], cfg["num_layers"], cfg["num_heads"]
    ctx = cfg["max_position_embeddings"]
    if "local" in cfg["attention_layers"]:
        ctx = min(ctx, cfg["window_size"])
    m = Model("gpt_neo", LAYERNORM | LEARNED_POS | GELU | TIED, dim, L, H, H,
              cfg["intermediate_size"] or 4 * dim, cfg["vocab_size"], ctx, 0.0, cfg["layer_norm_epsilon"])
    t = m.tensors
    t["embed"], t["pos_embed"] = w["wte.weight"], w["wpe.weight"][:ctx]
    t["norm"], t["norm_b"] = w["ln_f.weight"], w["ln_f.bias"]
    for i in range(L):
        p, a = f"h.{i}.", f"h.{i}.attn.attention."
        t[f"blk.{i}.attn_norm"], t[f"blk.{i}.attn_norm_b"] = w[p + "ln_1.weight"], w[p + "ln_1.bias"]
        t[f"blk.{i}.ffn_norm"], t[f"blk.{i}.ffn_norm_b"] = w[p + "ln_2.weight"], w[p + "ln_2.bias"]
        t[f"blk.{i}.wq"] = w[a + "q_proj.weight"] * np.float32(np.sqrt(dim // H))
        t[f"blk.{i}.wk"], t[f"blk.{i}.wv"] = w[a + "k_proj.weight"], w[a + "v_proj.weight"]
        t[f"blk.{i}.wo"], t[f"blk.{i}.bo"] = w[a + "out_proj.weight"], w[a + "out_proj.bias"]
        t[f"blk.{i}.w_up"], t[f"blk.{i}.b_up"] = w[p + "mlp.c_fc.weight"], w[p + "mlp.c_fc.bias"]
        t[f"blk.{i}.w_down"], t[f"blk.{i}.b_down"] = w[p + "mlp.c_proj.weight"], w[p + "mlp.c_proj.bias"]
    m.tokens, tok_flags = hf_tokens(repo, m.vocab)
    m.flags |= tok_flags
    m.bos = m.eos = cfg["eos_token_id"]
    return m


# ---------------------------------------------------------------- HF tokenizers: tokenizer.json or tokenizer.model

def hf_tokens(repo, vocab):
    """tokenizer.json, else a SentencePiece tokenizer.model. Returns (tokens, tokenizer flags)."""
    try:
        tok = json.loads(Path(hf_hub_download(repo, "tokenizer.json")).read_text())
    except EntryNotFoundError:
        return sentencepiece(hf_hub_download(repo, "tokenizer.model"), vocab), SP_PREFIX
    return hf_tokenizer(tok, vocab)


def gpt2_byte_map():
    """GPT-2's printable-unicode-per-byte mapping, inverted: char -> byte."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def hf_tokenizer(tok, vocab):
    """tokenizer.json -> ([(bytes, -merge rank)], SP_PREFIX or DIGITS flags)."""
    vocab_map = dict(tok["model"]["vocab"])
    for a in tok.get("added_tokens", []):
        vocab_map.setdefault(a["content"], a["id"])
    sp = any(s.startswith("▁") for s in list(vocab_map)[:2000])
    byte_map = gpt2_byte_map()

    def piece(s):   # (bytes, score if no merge creates it)
        if sp:
            return sp_piece(s, NO_MERGE)
        return bytes(byte_map[c] for c in s) if all(c in byte_map for c in s) else s.encode("utf-8"), NO_MERGE

    scores = {}
    for rank, merge in enumerate(tok["model"]["merges"]):
        a, b = merge.split(" ") if isinstance(merge, str) else merge
        scores.setdefault(a + b, -float(rank))
    tokens = [(b"", NO_MERGE)] * vocab
    for s, i in vocab_map.items():
        if i < vocab:
            b, score = piece(s)
            tokens[i] = (b, scores.get(s, score))
    digits = '"individual_digits": true' in json.dumps(tok.get("pre_tokenizer"))
    return tokens, SP_PREFIX if sp else DIGITS if digits else 0


def sentencepiece(path, vocab):
    """A SentencePiece tokenizer.model (protobuf) -> [(bytes, score)], padded to vocab."""
    def fields(b):
        i = 0
        while i < len(b):
            key, i = varint(b, i)
            if key & 7 == 0:
                v, i = varint(b, i)
            elif key & 7 == 2:
                n, i = varint(b, i)
                v, i = b[i:i + n], i + n
            else:
                n = 4 if key & 7 == 5 else 8
                v, i = b[i:i + n], i + n
            yield key >> 3, v

    def varint(b, i):
        v = shift = 0
        while b[i] & 0x80:
            v, shift, i = v | (b[i] & 0x7F) << shift, shift + 7, i + 1
        return v | b[i] << shift, i + 1

    tokens = []
    for f, v in fields(Path(path).read_bytes()):
        if f == 1:   # SentencePiece {1: piece, 2: score, 3: type}, type 1 is a normal piece
            p = dict(fields(v))
            score = struct.unpack("<f", p[2])[0] if 2 in p else 0.0
            tokens.append(sp_piece(p[1].decode(), score if p.get(3, 1) == 1 else NO_MERGE))
    return tokens + [(b"", NO_MERGE)] * (vocab - len(tokens))


# ---------------------------------------------------------------- synthetic

def random(spec):
    """A tiny random model from a spec, with a byte-level tokenizer: 256 bytes, then bos and eos."""
    rng = np.random.default_rng(spec.seed)
    gpt2 = spec.arch == "gpt2"
    dim, L, H, KV, ffn, vocab = spec.dim, spec.n_layers, spec.n_heads, spec.n_kv_heads, spec.ffn, spec.vocab
    kvd = dim // H * KV
    flags = (LAYERNORM | LEARNED_POS | GELU if gpt2 else 0) | (TIED if spec.tied else 0)
    m = Model(spec.arch, flags, dim, L, H, KV, ffn, vocab, spec.ctx, 10000.0, 1e-5)

    def mat(rows, cols):
        return (rng.standard_normal((rows, cols)) / np.sqrt(cols)).astype(np.float32)

    def vec(n, scale=0.1):
        return (rng.standard_normal(n) * scale).astype(np.float32)

    t = m.tensors
    t["embed"] = mat(vocab, dim)
    if gpt2:
        t["pos_embed"] = mat(spec.ctx, dim) * 0.1
    if not spec.tied:
        t["lm_head"] = mat(vocab, dim)
    t["norm"] = 1 + vec(dim)
    if gpt2:
        t["norm_b"] = vec(dim)
    for i in range(L):
        b = f"blk.{i}."
        t[b + "attn_norm"], t[b + "ffn_norm"] = 1 + vec(dim), 1 + vec(dim)
        t[b + "wq"], t[b + "wk"], t[b + "wv"], t[b + "wo"] = mat(dim, dim), mat(kvd, dim), mat(kvd, dim), mat(dim, dim)
        t[b + "w_up"], t[b + "w_down"] = mat(ffn, dim), mat(dim, ffn)
        if gpt2:
            t[b + "attn_norm_b"], t[b + "ffn_norm_b"] = vec(dim), vec(dim)
            t[b + "bq"], t[b + "bk"], t[b + "bv"], t[b + "bo"] = vec(dim), vec(kvd), vec(kvd), vec(dim)
            t[b + "b_up"], t[b + "b_down"] = vec(ffn), vec(dim)
        else:
            t[b + "w_gate"] = mat(ffn, dim)
    tokens = [(bytes([i]), NO_MERGE) for i in range(256)] + [(b"<s>", NO_MERGE), (b"</s>", NO_MERGE)]
    assert vocab >= len(tokens), "vocab must hold 256 bytes plus bos and eos"
    m.tokens = tokens + [(b"", NO_MERGE)] * (vocab - len(tokens))
    m.bos, m.eos = 256, 257
    return m
