<h1 align="center">l3m</h1>

<p align="center">Tiny language models that never touch DRAM.</p>

<p align="center">
  <a href="https://github.com/fpedd/l3m/actions/workflows/checks.yml"><img src="https://img.shields.io/github/actions/workflow/status/fpedd/l3m/checks.yml?branch=main&label=checks" alt="Checks"></a>
  <a href="https://github.com/fpedd/l3m/blob/main/LICENSE"><img src="https://img.shields.io/github/license/fpedd/l3m" alt="License"></a>
  <img src="https://img.shields.io/badge/C11-GCC%20%7C%20clang-blue" alt="C11">
</p>

l3m is a minimal inference engine that keeps small language models entirely in CPU cache.
Each core of a modern desktop CPU reads its L2 and L3 at around 100 GB/s, so 16 cores
together stream about 1.6 TB/s. Dual-channel DDR5 delivers 60 to 90 GB/s, shared by all of them.
l3m shards every weight matrix across the cores until each shard fits in its core's cache,
then streams the weights from there on every token. For models that fit, token generation
runs 3 to 5 times faster than llama.cpp on the same cores. l3m runs on Linux, on x86-64 CPUs
with AVX2.

## Quick start

From the command line:
```bash
make all
uv run l3m-export stories15M # HF weights -> stories15M.l3m (bf16, 29 MiB)
./l3m generate stories15M.l3m -p "Once upon a time" -n 200
```

Or from Python:
```python
import l3m
m = l3m.Model("stories15M.l3m")
prompt = "Once upon a time"
response = m.decode(m.generate(m.encode(prompt), 200))
print(prompt + response)
```

`models/` holds specs for various models, including stories15M and SmolLM2-135M. To make sure a model
really runs from cache, the loader refuses a model that does not fit the cache budget. If you hit
that limit, `--ctx 128` shrinks the KV cache, and `--force` loads the model anyway. `./l3m hw` shows
each core's budget, and `./l3m check stories15M.l3m` compares the output against the fp32 reference.

## Results

For the models that fit, llama.cpp streams the weights at 80 to 140 GB/s, where l3m reaches
330 to 500 GB/s: each core keeps the same rows from token to token, and the cores
synchronize 4 times per layer. Evicting l3m's weights before every token (`--cold`) makes it
6 to 8 times slower. Once a model outgrows the cache, the lead shrinks toward what the memory
bus allows.

| model                              | cores | l3m tok/s | llama.cpp tok/s | speedup | l3m from DRAM |
| ---------------------------------- | ----: | --------: | --------------: | ------: | ------------: |
| stories15M bf16                    |     8 |    15,300 |           3,180 |    4.8x |          1.1% |
| stories15M q8_0                    |     8 |    20,300 |           5,200 |    3.9x |          0.5% |
| stories42M q8_0                    |    16 |    10,700 |           3,120 |    3.4x |          1.6% |
| stories110M q4_0, ctx 128          |    16 |     5,800 |           1,340 |    4.3x |          3.3% |
| SmolLM2-135M q8_0, ctx 256, forced |    16 |       530 |             430 |    1.2x |           72% |

Measured on a Ryzen 9 7950X (16 cores, 2 x 32 MiB L3, DDR5-5600) with
`bench/vs_llamacpp.py <file.l3m> <file.gguf> --cores 0-7` (or `0-15`, plus `--ctx` or `--force`):
256 greedy tokens, median of 7 runs, llama.cpp at its best thread count. Every model is exported
with `--head-dtype` equal to its `--dtype`. llama.cpp (master of 2026-09-25) runs GGUFs of the same
dtypes, made with `llama-quantize --pure`. "From DRAM" is machine-wide memory-controller reads
as a share of the weight bytes l3m streams per token (`sudo ./l3m bench`). Against fp32, bf16
and q8_0 stay within 0.2% perplexity, and q4_0 costs stories110M 1.8% (`bench/ppl.py`).

## Prior work

The closest prior work is
[Cache-Resident LLM Inference in GB-Scale Last-Level Caches](https://arxiv.org/abs/2606.25353)
(Zhang et al., 2026). It holds INT8 weights in the 1152 MB L3 of EPYC 9684X sockets, shards
each GEMV by output channel across a socket's cores and pipelines layers across servers. On
Llama-3.2-3B and Llama-2-7B it reports 2 to 11.5x lower time per token than llama.cpp, but I
could not find any code for it. l3m is partly inspired by it: the same output-sharded,
cache-resident GEMV, on one consumer CPU, in the open. The idea is older:
[DeepCPU](https://www.usenix.org/conference/atc18/presentation/zhang-minjia) (2018) kept each
core's slice of RNN weights in its L2. SRAM-based accelerators such as the Cerebras wafer-scale
engine, Groq's LPU and d-Matrix's Corsair apply it in hardware.

## Tests

```bash
make test                                                  # kernels, then their throughput
uv sync --extra dev && uv run pytest                       # engine, exporter, tokenizer
uv sync --extra dev --extra hf && uv run pytest -m network # real weights against transformers
```

## License

Copyright 2026 Fabian Peddinghaus. Licensed under the MIT License. See [LICENSE](LICENSE).