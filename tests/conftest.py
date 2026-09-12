import os
import struct
import subprocess
from pathlib import Path

import pytest

import l3m
from l3m import archs, export

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "tests" / "models"

SPECS = ["tiny_llama", "tiny_gpt2"]
DTYPES = ["f32", "bf16", "q8_0", "q4_0", "mxfp4"]
CPUS = sorted(os.sched_getaffinity(0))


def cores(n):
    return CPUS[:n]


def tiny_spec(name="tiny_llama"):
    return export.load_spec(str(MODELS / f"{name}.py"))


def toc(data):
    """{tensor name: offset of its table of contents entry}."""
    n, _, off, _ = struct.unpack_from("<IIQQ", data, 64)
    return {bytes(data[e:e + 64]).split(b"\0")[0].decode(): e for e in range(off, off + 120 * n, 120)}


def with_tokens(path, tokens, flags=0):
    s = tiny_spec()
    s.vocab = len(tokens)
    m = archs.random(s)
    m.tokens, m.flags = tokens, m.flags | flags
    export.write(m, "f32", "f32", path)
    return l3m.Model(path, force=True)


@pytest.fixture(scope="session")
def cli():
    if "L3M_SKIP_MAKE" not in os.environ:
        subprocess.run(["make", "-s", "-C", str(ROOT)], check=True)
    return ROOT / "l3m"


@pytest.fixture(scope="session")
def exports(cli, tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    return {(s, d): export.export(str(MODELS / f"{s}.py"), dtype=d, head_dtype=d, out=out / f"{s}-{d}.l3m")
            for s in SPECS for d in DTYPES}
