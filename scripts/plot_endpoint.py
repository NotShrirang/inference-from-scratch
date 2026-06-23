#!/usr/bin/env python3
"""Plot a benchmark_endpoint.py result file into a single 4-panel figure.

Reads the JSON rows emitted by scripts/benchmark_endpoint.py (a list of
per-concurrency-level dicts) and draws the four curves that matter for the
inference-optimization ladder:

  1. throughput (tok/s)  vs concurrency  — does adding load buy throughput?
  2. TTFT p50/p95        vs concurrency  — time to first token (prefill cost)
  3. TPOT p50/p95        vs concurrency  — per-token decode cost (the cache metric)
  4. e2e p99             vs concurrency  — tail latency a client actually feels

Usage:
    python scripts/plot_endpoint.py \
        --results benchmarks/rung2_kv_cache.json \
        --out benchmarks/rung2_kv_cache.png \
        --title "Rung 2 — KV cache (Qwen2.5-0.5B)"
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")  # headless; no display needed
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="bench_endpoint_results.json")
    ap.add_argument("--out", default="benchmarks/latest.png")
    ap.add_argument("--title", default="Inference endpoint benchmark")
    ap.add_argument("--dpi", type=int, default=120)
    a = ap.parse_args()

    with open(a.results) as f:
        rows = json.load(f)
    rows.sort(key=lambda r: r["concurrency"])

    conc = [r["concurrency"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(a.title, fontsize=14, fontweight="bold")

    def setup(ax, ylabel, title):
        ax.set_xscale("log", base=2)
        ax.set_xticks(conc)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("concurrency")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)

    # 1. throughput
    ax = axes[0][0]
    ax.plot(conc, [r["throughput_tok_s"] for r in rows], "o-", color="#2563eb")
    setup(ax, "tokens / s", "Throughput vs concurrency")

    # 2. TTFT
    ax = axes[0][1]
    ax.plot(conc, [r["ttft_p50"] for r in rows], "o-", label="p50", color="#dc2626")
    ax.plot(conc, [r["ttft_p95"] for r in rows], "s--", label="p95", color="#f87171")
    setup(ax, "seconds", "Time to first token (TTFT)")
    ax.legend()

    # 3. TPOT
    ax = axes[1][0]
    ax.plot(conc, [r["tpot_p50"] for r in rows], "o-", label="p50", color="#059669")
    ax.plot(conc, [r["tpot_p95"] for r in rows], "s--", label="p95", color="#34d399")
    setup(ax, "seconds / token", "Time per output token (TPOT)")
    ax.legend()

    # 4. e2e p99
    ax = axes[1][1]
    ax.plot(conc, [r["e2e_p99"] for r in rows], "o-", color="#9333ea")
    setup(ax, "seconds", "End-to-end latency (p99)")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(a.out, dpi=a.dpi)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
