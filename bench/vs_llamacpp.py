#!/usr/bin/env python3
"""l3m against llama.cpp on the same model and cores: token generation, median tok/s over --runs.

    uv run bench/vs_llamacpp.py stories15M.l3m ~/llama.cpp/gguf/stories15M-bf16.gguf --cores 0-7

llama.cpp runs on the first t of --cores for each thread count t, so its best count is reported, not only
the widest. Prompt processing is not measured: l3m feeds a prompt one token at a time.
"""
import argparse
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def expand(cores):
    out = []
    for part in cores.split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out


def l3m_tps(path, cores, n, runs):
    def once():
        out = subprocess.run([ROOT / "l3m", "bench", path, "--cores", cores, "-n", str(n), "--md"],
                             capture_output=True, text=True, check=True).stdout
        return float(out.splitlines()[2].split("|")[2])
    return statistics.median(once() for _ in range(runs))


def llamacpp_tps(bench, gguf, cpus, n, runs):
    out = subprocess.run(["taskset", "-c", ",".join(map(str, cpus)), bench, "-m", gguf, "-p", "0", "-n", str(n),
                          "-r", str(runs), "-t", str(len(cpus)), "-o", "jsonl"], capture_output=True, text=True, check=True).stdout
    return statistics.median(json.loads(out.splitlines()[-1])["samples_ts"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("l3m_file")
    p.add_argument("gguf")
    p.add_argument("--cores", default="0-7")
    p.add_argument("--llama-cpp", default=str(Path.home() / "llama.cpp"), help="checkout with build/bin/llama-bench")
    p.add_argument("-n", type=int, default=256)
    p.add_argument("--runs", type=int, default=7)
    a = p.parse_args(argv)

    lcpp, cpus = Path(a.llama_cpp), expand(a.cores)
    commit = subprocess.run(["git", "-C", lcpp, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"llama.cpp {commit or '?'}, cores {a.cores}, {a.n} tokens, median of {a.runs}")
    counts = sorted({t for t in (1, 2, 4, 8, 16, 32, len(cpus)) if t <= len(cpus)})
    best = 0
    for t in counts:
        tps = llamacpp_tps(lcpp / "build/bin/llama-bench", a.gguf, cpus[:t], a.n, a.runs)
        best = max(best, tps)
        print(f"  llama.cpp {t:3d} threads {tps:10.0f} tok/s", flush=True)
    tps = l3m_tps(a.l3m_file, a.cores, a.n, a.runs)
    print(f"  l3m       {len(cpus):3d} cores   {tps:10.0f} tok/s   {tps / best:.1f}x llama.cpp's best")


if __name__ == "__main__":
    main()
