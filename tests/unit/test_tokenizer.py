import random
import time

import pytest

import l3m
from l3m import archs
from tests.conftest import tiny_spec, with_tokens

BYTES = archs.random(tiny_spec()).tokens[:258]   # 256 bytes, bos, eos
TEXTS = ["Once upon a time, there was a little girl named Lily.", "tabs\tand\nnewlines", "ünïcödé 🐍 bytes", ""]


@pytest.fixture(scope="module")
def model(exports):
    return l3m.Model(exports[("tiny_llama", "f32")], force=True)


@pytest.mark.parametrize("text", TEXTS)
def test_round_trip(model, text):
    ids = model.encode(text, bos=False)
    assert ids == list(text.encode())
    assert model.decode(ids) == text


def test_bos_eos(model):
    assert model.bos == 256 and model.eos == 257
    assert model.encode("a")[0] == model.bos
    assert model.decode_bytes([model.bos, model.eos]) == b"<s></s>"


@pytest.mark.parametrize("digits", [False, True])
def test_byte_level_pieces(cli, tmp_path, digits):
    """Merges stay inside GPT-2 pre-tokenizer pieces."""
    model = with_tokens(tmp_path / "bpe.l3m", BYTES + [(b"\n\n", -1.0), (b"'s", -2.0)], archs.DIGITS if digits else 0)
    assert model.encode("a\n\nb", bos=False) == [97, 10, 10, 98]
    assert model.encode("a\n\n", bos=False) == [97, 258]
    assert model.encode("it's '", bos=False) == [105, 116, 259, 32, 39]
    assert model.encode("x='s", bos=False) == [120, 61, 39, 115]
    assert model.encode("\n\n1", bos=False) == ([258, 49] if digits else [10, 10, 49])


MERGES = {258: (b"ab", -3.0), 259: (b"bc", -1.0), 260: (b"xy", -1.0), 261: (b"xyz", -2.0), 262: (b"aa", -4.0),
          263: (b" a", -5.0), 264: ("ü".encode(), archs.NO_MERGE), 265: (b",", 0.0)}


@pytest.fixture(scope="module")
def merges(cli, tmp_path_factory):
    out = tmp_path_factory.mktemp("tok")
    return {sp: with_tokens(out / f"sp{sp}.l3m", BYTES + list(MERGES.values()), archs.SP_PREFIX if sp else 0) for sp in (False, True)}


@pytest.mark.parametrize("text, ids", [
    ("abc", [97, 259]),              # bc outranks ab
    ("xyz", [261]),                  # xy, then xy + z
    ("aaa", [262, 97]),              # equal candidates: the leftmost merges
    ("üé", [264, 0xC3, 0xA9]),       # a whole-character piece, else its bytes
    (",", [265]),                    # a duplicate piece resolves to the higher score
])
def test_merges(merges, text, ids):
    assert merges[False].encode(text, bos=False) == ids


def test_sentencepiece_prefix(merges):
    m = merges[True]
    assert m.encode("a", bos=False) == [263]
    assert m.encode("", bos=False) == []
    assert m.encode("b") == [m.bos, 32, 98]


def test_merge_round_trip(merges):
    rng = random.Random(0)
    for _ in range(200):
        text = "".join(rng.choices(["a", "b", "c", "x", "y", "z", ",", " ", "ü", "é"], k=rng.randrange(12)))
        assert merges[False].decode(merges[False].encode(text, bos=False)) == text


def test_merges_match_quadratic_reference(merges):
    vocab = {b: i for i, (b, _) in enumerate(BYTES)} | {b: i for i, (b, _) in MERGES.items()}
    score = {i: s for i, (_, s) in MERGES.items()}
    rng = random.Random(1)
    for _ in range(50):
        text = "".join(rng.choices("abcxyz", k=rng.randrange(1, 300)))
        syms = [bytes([c]) for c in text.encode()]
        while cands := [(score[vocab[a + b]], -i) for i, (a, b) in enumerate(zip(syms, syms[1:])) if a + b in vocab]:
            i = -max(cands)[1]
            syms[i:i + 2] = [syms[i] + syms[i + 1]]
        assert merges[False].encode(text, bos=False) == [vocab[s] for s in syms]


def test_long_piece_is_not_quadratic(merges):
    t = time.perf_counter()
    assert merges[False].encode("a" * (1 << 20), bos=False) == [262] * (1 << 19)
    assert time.perf_counter() - t < 10
