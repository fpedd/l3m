"""ctypes wrapper over libl3m.so (src/l3m.h).

    import l3m
    m = l3m.Model("stories15M.l3m")
    print("Once upon a time" + m.decode(m.generate(m.encode("Once upon a time"), 200)))
"""
import ctypes as C
import functools
import os
import threading
from pathlib import Path

_CB = C.CFUNCTYPE(C.c_int, C.c_int32, C.c_void_p)


class Opts(C.Structure):
    _fields_ = [("cpus", C.POINTER(C.c_int)), ("n_cpus", C.c_int), ("ctx", C.c_int),
                ("reference", C.c_int), ("force", C.c_int), ("verbose", C.c_int), ("cold", C.c_int)]


class Perf(C.Structure):
    _fields_ = [(f, C.c_double) for f in ("tokens", "ns_token", "ns_compute", "ns_wait", "bytes_token", "l2_bytes",
                                           "l3_miss", "dram_bytes", "energy_j", "llc_occupancy")]


def _find_lib():
    for p in (os.environ.get("L3M_LIB"), Path(__file__).parents[2] / "libl3m.so", Path("libl3m.so")):
        if p and Path(p).exists():
            return str(Path(p).resolve())
    raise OSError("libl3m.so not found: run make, or set L3M_LIB")


@functools.cache
def _lib():
    lib = C.CDLL(_find_lib())
    sig = {
        "l3m_load": (C.c_void_p, [C.c_char_p, C.POINTER(Opts)]),
        "l3m_free": (None, [C.c_void_p]),
        "l3m_generate": (C.c_int, [C.c_void_p, C.POINTER(C.c_int32), C.c_int, C.c_int, C.c_float, C.c_uint64, _CB, C.c_void_p]),
        "l3m_logits": (C.POINTER(C.c_float), [C.c_void_p]),
        "l3m_vocab": (C.c_int, [C.c_void_p]),
        "l3m_ctx": (C.c_int, [C.c_void_p]),
        "l3m_describe": (C.c_char_p, [C.c_void_p]),
        "l3m_encode": (C.c_int, [C.c_void_p, C.c_char_p, C.c_int, C.POINTER(C.c_int32), C.c_int]),
        "l3m_decode": (C.c_void_p, [C.c_void_p, C.c_int32, C.POINTER(C.c_int)]),
        "l3m_bos": (C.c_int, [C.c_void_p]),
        "l3m_eos": (C.c_int, [C.c_void_p]),
        "l3m_perf_read": (C.c_int, [C.c_void_p, C.POINTER(Perf)]),
    }
    for name, (res, args) in sig.items():
        f = getattr(lib, name)
        f.restype, f.argtypes = res, args
    return lib


class Model:
    """A loaded .l3m file, sharded over cpus (default: the cores of the largest L3) with a KV cache of ctx
    (default min(model, 512)). Thread-safe; on_token may read the model but not generate or close it."""
    _m = None

    def __init__(self, path, cpus=None, ctx=0, reference=False, force=False, verbose=False, cold=False):
        self._lock, self._generating = threading.RLock(), False
        if ctx and ctx < 2:
            raise ValueError(f"ctx {ctx}: needs room for a prompt token and one more")
        self._lib = _lib()
        opts = Opts(ctx=ctx, reference=int(reference), force=int(force), verbose=int(verbose), cold=int(cold))
        if cpus:
            self._cpus = (C.c_int * len(cpus))(*cpus)
            opts.cpus, opts.n_cpus = self._cpus, len(cpus)
        self._m = self._lib.l3m_load(os.fsencode(path), C.byref(opts))
        if not self._m:
            raise RuntimeError(f"l3m_load failed for {path} (the reason is on stderr)")
        self.vocab, self.ctx = self._lib.l3m_vocab(self._m), self._lib.l3m_ctx(self._m)
        self.bos, self.eos = self._lib.l3m_bos(self._m), self._lib.l3m_eos(self._m)

    def _handle(self, writing=False):
        if writing and self._generating:
            raise RuntimeError("l3m: on_token cannot call generate or close")
        if not self._m:
            raise ValueError("l3m: model is closed")
        return self._m

    def close(self):
        with self._lock:
            if self._m:
                self._lib.l3m_free(self._handle(writing=True))
                self._m = None

    __del__ = close

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def describe(self):
        with self._lock:
            return self._lib.l3m_describe(self._handle()).decode()

    def encode(self, text, bos=True):
        if "\0" in text:
            raise ValueError("l3m: text contains a NUL")
        data = text.encode()
        buf = (C.c_int32 * (len(data) + 8))()
        with self._lock:
            n = self._lib.l3m_encode(self._handle(), data, int(bos), buf, len(buf))
        return list(buf[:n])

    def decode_bytes(self, ids):
        out, n = bytearray(), C.c_int()
        with self._lock:
            for t in ids:
                if not 0 <= t < self.vocab:
                    raise ValueError(f"token {t} is outside vocab {self.vocab}")
                out += C.string_at(self._lib.l3m_decode(self._handle(), t, C.byref(n)), n.value)
        return bytes(out)

    def decode(self, ids):
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def generate(self, prompt, max_new, temperature=0.8, seed=0, on_token=None):
        """Up to max_new token ids after prompt, stopping at EOS or BOS; on_token(id) returning False stops."""
        if not 0 < len(prompt) < self.ctx or not all(0 <= t < self.vocab for t in prompt):
            raise ValueError(f"prompt of {len(prompt)} ids: needs 1 to {self.ctx - 1} ids, each below vocab {self.vocab}")
        out, err = [], []

        def cb(tok, _):
            out.append(tok)
            try:
                return 0 if on_token is not None and on_token(tok) is False else 1
            except BaseException as e:
                err.append(e)
                return 0

        buf = (C.c_int32 * len(prompt))(*prompt)
        with self._lock:
            m = self._handle(writing=True)
            self._generating = True
            try:
                self._lib.l3m_generate(m, buf, len(prompt), min(max_new, self.ctx), temperature, seed % 2**64, _CB(cb), None)
            finally:
                self._generating = False
        if err:
            raise err[0]
        return out

    @property
    def logits(self):
        """The last position's logits."""
        import numpy as np
        with self._lock:
            return np.ctypeslib.as_array(self._lib.l3m_logits(self._handle()), shape=(self.vocab,)).copy()

    def perf(self):
        """Per-token counters since the previous call; NaN where unavailable."""
        p = Perf()
        with self._lock:
            if self._lib.l3m_perf_read(self._handle(), C.byref(p)):
                return {f: float("nan") for f, _ in Perf._fields_}
        return {f: getattr(p, f) for f, _ in Perf._fields_}
