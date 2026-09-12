<h1 align="center">l3m</h1>

<p align="center">Tiny language models that never touch DRAM.</p>

<p align="center">
  <a href="https://github.com/fpedd/l3m/actions/workflows/checks.yml"><img src="https://img.shields.io/github/actions/workflow/status/fpedd/l3m/checks.yml?branch=main&label=checks" alt="Checks"></a>
  <a href="https://github.com/fpedd/l3m/blob/main/LICENSE"><img src="https://img.shields.io/github/license/fpedd/l3m" alt="License"></a>
  <img src="https://img.shields.io/badge/C11-GCC%20%7C%20clang-blue" alt="C11">
</p>

l3m is a minimal inference engine that keeps small language models entirely in CPU cache.
Each core of a modern desktop CPU reads its L2 and L3 at around 100 GB/s, so 16 cores
together stream about 1.5 TB/s. Dual-channel DDR5 delivers 60 to 90 GB/s, shared by all of them.
l3m shards every weight matrix across the cores until each shard fits in its core's cache,
then streams the weights from there on every token. For models that fit, token generation
runs 3 to 5 times faster than llama.cpp on the same cores.

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
that limit, `--ctx 128` shrinks the KV cache, and `--force` loads the model anyway.

## Results

The speedup holds as long as the model fits in cache. Once it outgrows the cache, more and more
of its weights come from DRAM, and the lead shrinks toward what the memory bus allows.

| model | cores | l3m tok/s | llama.cpp tok/s | speedup | from DRAM |
|---|---:|---:|---:|---:|---:|
| stories15M bf16 | 8 | 15,300 | 3,180 | 4.8x | 1.1% |
| stories15M q8_0 | 8 | 20,300 | 5,200 | 3.9x | 0.5% |
| stories42M q8_0 | 16 | 10,700 | 3,120 | 3.4x | 1.6% |
| stories110M q4_0, ctx 128 | 16 | 5,800 | 1,340 | 4.3x | 3.3% |
| SmolLM2-135M q4_0, ctx 256, forced | 16 | 1,590 | 700 | 2.3x | 38% |
| SmolLM2-135M q8_0, ctx 256, forced | 16 | 530 | 430 | 1.2x | 72% |

Measured on a Ryzen 9 7950X (16 cores, 2 x 32 MiB L3, DDR5-5600): 256 greedy tokens,
median of 7 runs, `sudo ./l3m bench <file> --cores 0-7 -n 256` (or `0-15`). Every model is
exported with `--head-dtype` equal to its `--dtype`. llama.cpp (master of 2026-09-25) runs GGUFs
of the same dtypes, made with `llama-quantize --pure`, under `llama-bench -p 0 -n 256`, pinned
to the same cores. "From DRAM" is memory-controller reads as a share of the weight bytes
l3m streams per token.

## Prior work

The closest prior work is
[Cache-Resident LLM Inference in GB-Scale Last-Level Caches](https://arxiv.org/abs/2606.25353)
(Zhang et al., 2026). It holds INT8 weights in the 1152 MB L3 of EPYC 9684X sockets, shards
each GEMV by output channel across a socket's cores and pipelines layers across sockets. On
Llama-3.2-3B and Llama-2-7B it reports 2 to 11.5x lower time per token than llama.cpp, but the
paper publishes no code. l3m is partly inspired by it and aims to fill that gap: the same
output-sharded, cache-resident GEMV, on one consumer CPU, in the open. It differs in the details:
attention stays on the cores that own the heads, weights use ggml blocks rather than INT8, and
each all-gather waits on a full barrier rather than per-head ready signals.

The same idea drives some inference accelerators, which skip off-chip memory and keep the
weights in on-chip SRAM. The best-known example is Groq, whose technology NVIDIA licensed in
late 2025. For a deep dive into its architecture, see
[Inside Groq LPU Architecture](https://github.com/zartbot/blog/issues/4).

## License

Copyright 2026 Fabian Peddinghaus. Licensed under the MIT License. See [LICENSE](LICENSE).