import ctypes as C
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest

import l3m
from l3m import export
from tests.conftest import CPUS, DTYPES, MODELS, ROOT, SPECS, cores, tiny_spec

PROMPT = "Once upon a time"
ISAS = ["scalar", "avx2", "avx512"]

_spec = importlib.util.spec_from_file_location("vs_hf", ROOT / "bench" / "vs_hf.py")
vs_hf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vs_hf)
log_softmax = vs_hf.log_softmax


def run(cli, *args, code=0, **kw):
    r = subprocess.run([str(cli), *map(str, args)], capture_output=True, encoding="utf-8", errors="replace", **kw)
    assert r.returncode == code, r.stdout + r.stderr
    return r.stdout


def last_logits(m, ids):
    m.generate(ids, 0)
    return m.logits.astype(np.float64)


def assert_fast_matches_reference(fast, path):
    ref = l3m.Model(path, reference=True)
    ids = ref.encode(PROMPT)
    lr, lf = log_softmax(last_logits(ref, ids)), log_softmax(last_logits(fast, ids))
    assert np.exp(lr) @ (lr - lf) < 0.01
    assert lr.argmax() == lf.argmax()


def test_cli_check_info_hw(cli, exports):
    assert run(cli, "check", exports[("tiny_gpt2", "q4_0")], "--force", "-p", PROMPT).strip().endswith("PASS")
    assert "dim 64" in run(cli, "info", exports[("tiny_llama", "f32")])
    assert "cores" in run(cli, "hw")


def test_cli_generate_matches_python(cli, exports):
    path = exports[("tiny_gpt2", "bf16")]
    m = l3m.Model(path, force=True)
    want = m.decode(m.encode(PROMPT, bos=False) + m.generate(m.encode(PROMPT), 12, temperature=0))
    cpus = ",".join(map(str, cores(3)))
    out = run(cli, "generate", path, "--force", "--cores", cpus, "--ctx", 32, "-p", PROMPT, "-n", 12, "-t", 0)
    assert out == want + "\n"
    assert out == run(cli, "generate", path, "--ref", "-p", PROMPT, "-n", 12, "-t", 0)


def test_cli_chat_and_bench(cli, exports):
    path = exports[("tiny_llama", "q8_0")]
    out = run(cli, "chat", path, "--force", "-n", 4, "-s", 1, input="hello\nagain\n")
    assert out.count("> ") == 3
    assert run(cli, "chat", path, "--force", "-n", 4, "-p", "Lily said", input="hi\n").count("> ") == 2
    r = subprocess.run([str(cli), "chat", str(path), "--force", "--ctx", "8", "-n", "4"], input="x" * 20 + "\n",
                       capture_output=True, text=True)
    assert r.returncode == 0 and "fills the context" in r.stderr
    assert "tok/s" in run(cli, "bench", path, "--force", "-n", 8)
    ref = run(cli, "bench", path, "--ref", "-n", 4)
    assert "tok/s" in ref and "GB/s" not in ref and "root" not in ref
    cold = run(cli, "bench", path, "--force", "--cold", "-v", "-n", 4)
    assert "tok/s" in cold
    rows = run(cli, "bench", path, exports[("tiny_gpt2", "q8_0")], "--force", "-n", 8).splitlines()
    assert len(rows) == 4 and rows[2].startswith("| tiny_llama-q8_0.l3m")
    assert len(run(cli, "bench", path, "--force", "--md", "-n", 4).splitlines()) == 3


def test_cli_help_and_usage_errors(cli, exports):
    assert run(cli, "-h").startswith("usage:")
    path = exports[("tiny_llama", "f32")]
    for args, msg in [(["frob", path], "unknown command frob"), (["generate", path, "-n", "abc"], "bad -n abc"),
                      (["generate", path, "-n", "-5"], "bad -n"), (["generate", path, "--ctx", "1"], "bad --ctx"),
                      (["generate", path, "-t", "nan"], "bad -t"), (["generate", path, "-s", "x"], "bad -s"),
                      (["generate", path, "--bogus"], "unknown option --bogus"), (["generate", path, path], "only bench"),
                      (["bench", path, "-n", "0"], "-n 1 or more"), (["generate", path, "--cores", "0,"], "bad --cores"),
                      (["generate", path, "--ref", "--cold"], "--ref runs on one thread")]:
        r = subprocess.run([str(cli), *map(str, args)], capture_output=True, text=True)
        assert r.returncode == 2 and msg in r.stderr, (args, r.stderr)
    assert "tok/s" in run(cli, "bench", "-n", 2, "--force", path)


def test_load_refuses_over_budget(tmp_path):
    spec = tiny_spec()
    spec.dim, spec.ffn, spec.n_layers = 512, 2048, 4
    export.write(export.load_model(spec), "f32", "f32", tmp_path / "big.l3m")
    with pytest.raises(RuntimeError):
        l3m.Model(tmp_path / "big.l3m", cpus=cores(1))
    assert "on 1 core " in l3m.Model(tmp_path / "big.l3m", cpus=cores(1), force=True).describe()


def siblings():
    for c in CPUS:
        sib = Path(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")
        if not sib.exists():
            continue
        text = sib.read_text().strip().replace("-", ",")
        others = [int(s) for s in text.split(",") if int(s) != c and int(s) in CPUS]
        if others:
            return c, others[0]
    return None


def test_smt_siblings(exports):
    """A sibling counts as its core."""
    pair = siblings()
    if not pair:
        pytest.skip("no SMT siblings in our affinity")
    path = exports[("tiny_llama", "f32")]
    with pytest.raises(RuntimeError):
        l3m.Model(path, cpus=list(pair))
    l3m.Model(path, cpus=list(pair), force=True)
    a, b = (l3m.Model(path, cpus=[c], verbose=True) for c in pair)
    assert a.describe() == b.describe()


@pytest.mark.parametrize("isa", ISAS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("spec", SPECS)
def test_fast_matches_reference(exports, monkeypatch, spec, dtype, isa):
    monkeypatch.setenv("L3M_ISA", isa)
    path = exports[(spec, dtype)]
    fast = l3m.Model(path, force=True)
    if f"({isa})" not in fast.describe():
        pytest.skip(f"{isa} not supported by this CPU")
    assert_fast_matches_reference(fast, path)


@pytest.mark.parametrize("body,head", [("q4_0", "q8_0"), ("mxfp4", "q8_0"), ("q8_0", "bf16")])
@pytest.mark.parametrize("spec", SPECS)
def test_mixed_head_dtype(tmp_path, spec, body, head):
    path = export.export(str(MODELS / f"{spec}.py"), dtype=body, head_dtype=head, out=tmp_path / "m.l3m")
    assert_fast_matches_reference(l3m.Model(path, force=True), path)


def test_attention_without_qkv_biases(tmp_path):
    """GPT-Neo: an output bias but no q, k, v biases."""
    m = export.load_model(tiny_spec("tiny_gpt2"))
    for i in range(m.n_layers):
        for b in "qkv":
            del m.tensors[f"blk.{i}.b{b}"]
    export.write(m, "q8_0", "q8_0", tmp_path / "neo.l3m")
    assert_fast_matches_reference(l3m.Model(tmp_path / "neo.l3m", force=True), tmp_path / "neo.l3m")


@pytest.mark.parametrize("spec", SPECS)
def test_matches_numpy_reference(exports, spec):
    """bench/vs_hf.py, then both paths at every position against its fp64 forward."""
    path = exports[(spec, "f32")]
    assert vs_hf.main([str(MODELS / f"{spec}.py"), "--file", str(path), "--prompt", PROMPT, "--new", "24"])
    fast, ref = l3m.Model(path, force=True), l3m.Model(path, reference=True)
    ids = ref.encode(PROMPT)
    seq = ids + fast.generate(ids, 24, temperature=0)
    want = vs_hf.numpy_forward(export.load_model(tiny_spec(spec)), seq, bf16_kv=True)
    for m in (ref, fast):
        assert np.abs(vs_hf.l3m_logits(m, seq) - want).max() < 1e-4 * np.abs(want).max()


@pytest.mark.parametrize("spec", SPECS)
def test_core_counts_give_identical_logits(exports, spec):
    path = exports[(spec, "f32")]
    ref = l3m.Model(path, reference=True)
    ids = ref.encode(PROMPT)
    lr = last_logits(ref, ids)
    counts = sorted({len(cores(n)) for n in (1, 3, 5)})
    if len(counts) < 2:
        pytest.skip("needs 2 or more cpus")
    runs = [last_logits(l3m.Model(path, cpus=cores(n), force=True), ids) for n in counts]
    assert all(np.array_equal(runs[0], r) for r in runs[1:])
    assert np.abs(runs[0] - lr).max() < 1e-5 * np.abs(lr).max()


def test_cold_matches_warm(exports):
    path = exports[("tiny_llama", "q8_0")]
    warm, cold = l3m.Model(path, force=True), l3m.Model(path, force=True, cold=True)
    a, b = (m.generate(m.encode(PROMPT), 24, temperature=0) for m in (warm, cold))
    assert a == b and len(a) == 24
    p = cold.perf()
    assert p["ns_token"] > 0 and 0 <= p["ns_wait"] < p["ns_token"]


@pytest.mark.parametrize("spec", SPECS)
def test_sampling_matches_reference(exports, spec):
    path = exports[(spec, "f32")]
    ref, fast = l3m.Model(path, reference=True), l3m.Model(path, cpus=cores(3), force=True)
    ids = ref.encode(PROMPT)
    for seed in range(8):
        a = ref.generate(ids, 32, temperature=1.0, seed=seed)
        assert a == fast.generate(ids, 32, temperature=1.0, seed=seed)
    assert a != ref.generate(ids, 32, temperature=1.0, seed=seed + 1)


def test_sampling_distribution(exports):
    """Chi-square of the first sampled token over 4000 seeds against softmax(logits / T)."""
    m = l3m.Model(exports[("tiny_gpt2", "f32")], cpus=cores(3), force=True)
    ids, temp, n = [m.bos], 0.5, 4000
    p = np.exp(log_softmax(last_logits(m, ids) / temp))
    counts = np.zeros(m.vocab)
    p[m.eos] += p[m.bos]   # both end generation
    p[m.bos] = 0
    for seed in range(n):
        out = m.generate(ids, 1, temperature=temp, seed=seed)
        counts[out[0] if out else m.eos] += 1
    big = p * n >= 5
    obs, exp = np.append(counts[big], counts[~big].sum()), np.append(p[big], p[~big].sum()) * n
    df = len(obs) - 1
    assert ((obs - exp) ** 2 / exp).sum() < df + 5 * np.sqrt(2 * df)


def test_stops_at_eos(exports, tmp_path):
    m = export.load_model(tiny_spec())
    free = l3m.Model(exports[("tiny_llama", "f32")], force=True)
    ids = free.encode(PROMPT)
    greedy = free.generate(ids, 24, temperature=0)
    m.eos = greedy[8]
    export.write(m, "f32", "f32", tmp_path / "eos.l3m")
    for model in (l3m.Model(tmp_path / "eos.l3m", force=True), l3m.Model(tmp_path / "eos.l3m", reference=True)):
        assert model.generate(ids, 24, temperature=0) == greedy[:greedy.index(m.eos)]


def test_callback_stops_and_perf_counts(exports):
    m = l3m.Model(exports[("tiny_llama", "q4_0")], force=True)
    seen = []
    ids = m.encode(PROMPT)
    out = m.generate(ids, 50, temperature=0, on_token=lambda t: seen.append(t) or len(seen) < 5)
    assert out == seen and len(out) == 5
    p = m.perf()
    assert p["tokens"] == len(ids) + len(out) - 1 and p["ns_token"] > 0 and p["bytes_token"] > 0
    assert all(np.isnan(v) for v in m.perf().values())


def test_c_api_contracts(exports):
    """What the Python wrappers guard, refused by the C API too."""
    m = l3m.Model(exports[("tiny_llama", "f32")], ctx=16, force=True)
    lib, h, cb = m._lib, m._m, l3m._CB(lambda t, u: 1)
    assert lib.l3m_generate(h, (C.c_int32 * 1)(m.vocab), 1, 2, 0.0, 0, cb, None) == -1
    assert lib.l3m_generate(h, (C.c_int32 * 16)(*[1] * 16), 16, 2, 0.0, 0, cb, None) == 0
    buf = (C.c_int32 * 2)()
    assert lib.l3m_encode(h, b"hello", 1, buf, 2) == 2 and list(buf) == [m.bos, ord("h")]
    n = C.c_int(7)
    lib.l3m_decode(h, 99999, C.byref(n))
    assert n.value == 0
    assert lib.l3m_perf_read(h, C.byref(l3m.Perf())) == -1


def test_python_api(exports):
    path = exports[("tiny_llama", "f32")]
    with l3m.Model(path, force=True, verbose=True) as m:
        assert m.generate(m.encode(PROMPT), 3, temperature=0)
    with pytest.raises(ValueError):
        m.describe()
    with pytest.raises(ValueError):
        l3m.Model(path, ctx=1)
    m = l3m.Model(path, force=True)
    for bad in (lambda: m.encode("a\0b"), lambda: m.decode([m.vocab]), lambda: m.decode([2**32 + 97])):
        with pytest.raises(ValueError):
            bad()


def test_ctx_option_limits_generation(exports):
    m = l3m.Model(exports[("tiny_llama", "f32")], ctx=16, force=True)
    assert m.ctx == 16
    assert len(m.generate(m.encode("x"), 100, temperature=0)) == 16 - 2
    with pytest.raises(ValueError):
        m.generate(m.encode("x" * 15), 10, temperature=0)
