import contextlib
import os
import struct
import subprocess
import threading

import pytest

import l3m
from l3m import export
from tests.conftest import ROOT, cores, tiny_spec, toc


def run(cli, *args):
    return subprocess.run([str(cli), *map(str, args)], capture_output=True, text=True)


def patched(src, dst, edits):
    """where: a header offset, (tensor, offset in its toc entry) or ("tok", offset in the tokenizer)."""
    b = bytearray(src.read_bytes())
    base = toc(b) | {"tok": struct.unpack_from("<Q", b, 80)[0]}
    for fmt, where, v in edits:
        struct.pack_into(fmt, b, base[where[0]] + where[1] if isinstance(where, tuple) else where, v)
    dst.write_bytes(b)
    return dst


CASES = {   # header: n_heads at 36, vocab 48, n_tensors 64, toc 72
    "n_tensors": ([("<I", 64, 0xFFFFFFFF)], "truncated file"),
    "toc_offset": ([("<Q", 72, 1 << 63)], "truncated file"),
    "heads": ([("<I", 36, 3)], "bad config"),
    "dtype": ([("<I", ("blk.0.w_up", 64), 9)], "bad dtype"),
    "shape": ([("<Q", ("blk.0.wk", 72), 1)], "nbytes does not match"),
    "offset": ([("<Q", ("blk.0.wq", 104), 1)], "misaligned"),
    "offset_wrap": ([("<Q", ("blk.0.wq", 104), (1 << 64) - 64)], "past the end"),
    "missing": ([("<2s", ("blk.1.w_down", 0), b"zz")], "blk.1.w_down missing"),
    "norm_dtype": ([("<I", ("norm", 64), 1), ("<Q", ("norm", 112), 128)], "tensor norm: expected f32"),
    "tok_count": ([("<I", ("tok", 0), 0x7FFFFFFF)], "bad tokenizer"),
    "tok_len": ([("<H", ("tok", 20), 65535)], "bad tokenizer"),
    "tok_bos": ([("<i", ("tok", 4), 99999)], "bad tokenizer"),
    "tok_vocab": ([("<I", 48, 100)], "tensor embed: expected"),
}


@pytest.mark.parametrize("case", CASES)
def test_malformed_file_is_a_clean_error(cli, exports, tmp_path, case):
    edits, msg = CASES[case]
    path = patched(exports[("tiny_llama", "q4_0")], tmp_path / f"{case}.l3m", edits)
    r = run(cli, "generate", path, "-n", 2, "-p", "hi", "--ref")
    assert r.returncode == 1 and msg in r.stderr, (r.returncode, r.stderr)


def test_qkv_biases_come_together(cli, tmp_path):
    m = export.load_model(tiny_spec("tiny_gpt2"))
    del m.tensors["blk.1.bk"]
    export.write(m, "f32", "f32", tmp_path / "bq.l3m")
    r = run(cli, "generate", tmp_path / "bq.l3m", "-n", 2, "--ref")
    assert r.returncode == 1 and "come together" in r.stderr, r.stderr


def test_truncated_file(cli, exports, tmp_path):
    data = exports[("tiny_gpt2", "q8_0")].read_bytes()
    toc_off, tok_off = struct.unpack_from("<QQ", data, 72)
    for cut in (0, 100, 1024, toc_off + 10, tok_off + 3, len(data) - 1):
        (tmp_path / "cut.l3m").write_bytes(data[:cut])
        r = run(cli, "generate", tmp_path / "cut.l3m", "-n", 2, "--ref")
        assert r.returncode == 1 and r.stderr, (cut, r.returncode)
        assert run(cli, "info", tmp_path / "cut.l3m").returncode in (0, 1), cut


def test_bad_arguments(cli, exports, tmp_path):
    path = exports[("tiny_llama", "f32")]
    (tmp_path / "zeros.l3m").write_bytes(b"\0" * 2048)
    for args, code in [(["generate", path, "--cores", "x"], 2), (["generate"], 2),
                       (["generate", tmp_path / "missing.l3m"], 1), (["generate", tmp_path / "zeros.l3m"], 1),
                       (["info", tmp_path / "zeros.l3m"], 1)]:
        r = run(cli, *args)
        assert r.returncode == code and r.stderr, (args, r.returncode)
    for bad in ("0,0", "0-", "-1", "4095", "4294967296"):
        r = run(cli, "generate", path, "--cores", bad, "-n", 2)
        assert r.returncode in (1, 2) and ("cores" in r.stderr or "cpu" in r.stderr), (bad, r.stderr)


def test_model_without_bos(exports, tmp_path):
    m = l3m.Model(patched(exports[("tiny_llama", "f32")], tmp_path / "nobos.l3m", [("<i", ("tok", 4), -1)]), force=True)
    assert m.bos == -1 and m.encode("ab") == [97, 98]


def test_prompt_must_fit_ctx(cli, exports):
    path = exports[("tiny_llama", "f32")]
    r = run(cli, "check", path, "--ctx", 4)
    assert r.returncode == 1 and "PASS" not in r.stdout and "--ctx" in r.stderr
    r = run(cli, "generate", path, "-p", "x" * 100, "-n", 2)
    assert r.returncode == 1 and "--ctx" in r.stderr


def test_api_misuse_raises(exports):
    m = l3m.Model(exports[("tiny_llama", "q8_0")], cpus=cores(3), force=True)
    for bad in ([], [-1], [m.vocab], [2**40]):
        with pytest.raises(ValueError):
            m.generate(bad, 2)

    def boom(t):
        raise KeyError("stop")
    with pytest.raises(KeyError):
        m.generate([1], 5, on_token=boom)
    for reenter in (lambda t: m.generate([1], 1), lambda t: m.close()):
        with pytest.raises(RuntimeError):
            m.generate([1], 5, on_token=reenter)
    streamed = []
    out = m.generate([1], 5, on_token=lambda t: streamed.append(m.decode_bytes([t])))
    assert b"".join(streamed) == m.decode_bytes(out)
    m.close()
    m.close()
    for f in (m.describe, lambda: m.generate([1], 2), lambda: m.encode("a")):
        with pytest.raises(ValueError):
            f()


def test_bad_cpus_and_affinity(exports):
    path = exports[("tiny_llama", "f32")]
    for cpus in (cores(1) * 2, [100000], [-1]):
        with pytest.raises(RuntimeError):
            l3m.Model(path, cpus=cpus, force=True)
    r = subprocess.run([str(ROOT / "l3m"), "generate", str(path), "--cores", f"{cores(1)[0]},{cores(1)[0]}"], capture_output=True, text=True)
    assert r.returncode == 1 and "appears twice" in r.stderr
    before = os.sched_getaffinity(0)
    m = l3m.Model(path, cpus=cores(2), force=True)
    m.generate([1], 3)
    assert os.sched_getaffinity(0) == before


def test_threads_share_a_model(exports):
    m = l3m.Model(exports[("tiny_gpt2", "bf16")], cpus=cores(3), force=True)
    want, got = m.generate([1, 5], 12, temperature=0), []

    def worker():
        with contextlib.suppress(ValueError):
            for _ in range(5):
                got.append(m.generate([1, 5], 12, temperature=0))

    def together(*fns):
        threads = [threading.Thread(target=f) for f in fns]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    together(worker, worker, worker)
    assert got == [want] * 15
    together(worker, worker, worker, m.close)
    assert got == [want] * len(got)
