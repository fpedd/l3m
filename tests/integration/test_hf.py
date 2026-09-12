import subprocess
import sys

import pytest
from huggingface_hub import hf_hub_download

from l3m import archs
from tests.conftest import ROOT, with_tokens

pytestmark = pytest.mark.network


@pytest.mark.parametrize("model", [
    "stories15M",               # llama2.c .bin, tied, against numpy
    "SmolLM2-135M",             # HF Llama, tied, grouped-query attention
    "distilgpt2",               # HF GPT-2: Conv1D transposes, fused qkv
    "TinyStories-33M",          # GPT-Neo: pytorch_model.bin, the attention scale folded into wq
])
def test_matches_independent_reference(cli, tmp_path, model):
    if model != "stories15M":
        pytest.importorskip("transformers")
    cmd = [sys.executable, str(ROOT / "bench" / "vs_hf.py"), model, "--dtype", "f32", "--file", str(tmp_path / "m.l3m")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


TEXTS = ["Once upon a time, there was a little girl named Lily.", "don't stop, they'll see", "numbers 12345 and 3.14",
         "ünïcödé 🐍 中文", "tabs\tand\nnewlines", "  two  spaces", "   leading", "trailing   ", "\n\n\nx",
         "a" * 40]


@pytest.mark.parametrize("source", ["llama2c", "tokenizer.json", "tokenizer.model"])
def test_sentencepiece_tokenizers(tmp_path, source):
    """Llama's vocabulary from each source the exporter reads, against sentencepiece."""
    spm = pytest.importorskip("sentencepiece")
    model = hf_hub_download("hf-internal-testing/llama-tokenizer", "tokenizer.model")
    tokens = {"llama2c": lambda: archs.llama2c_tokenizer(32000),
              "tokenizer.json": lambda: archs.hf_tokens("amd/AMD-Llama-135m", 32000)[0],
              "tokenizer.model": lambda: archs.sentencepiece(model, 32000)}[source]()
    ours, theirs = with_tokens(tmp_path / "t.l3m", tokens, archs.SP_PREFIX), spm.SentencePieceProcessor(model_file=model)
    for text in TEXTS:
        assert ours.encode(text, bos=False) == theirs.encode(text), text


@pytest.mark.parametrize("repo", ["openai-community/gpt2", "HuggingFaceTB/SmolLM2-135M"])
def test_byte_level_bpe_tokenizers(tmp_path, repo):
    tokenizers = pytest.importorskip("tokenizers")
    theirs = tokenizers.Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
    ours = with_tokens(tmp_path / "t.l3m", *archs.hf_tokens(repo, theirs.get_vocab_size()))
    for text in TEXTS:
        assert ours.encode(text, bos=False) == theirs.encode(text, add_special_tokens=False).ids, text
