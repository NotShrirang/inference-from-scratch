#!/usr/bin/env python3
"""Plot a benchmark_batch.py result into a 2-panel figure: throughput and
per-request latency vs batch size. The throughput curve climbing then flattening
is the step-3 payoff (memory-bandwidth amortization).

Usage:
    python scripts/plot_batch.py \
        --results benchmarks/step3_batching.json \
        --out benchmarks/step3_batching.png \
        --title "Step 3 — static batching (Qwen2.5-0.5B)"
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="bench_batch_results.json")
    ap.add_argument("--out", default="benchmarks/latest_batch.png")
    ap.add_argument("--title", default="Static batching benchmark")
    ap.add_argument("--dpi", type=int, default=120)
    a = ap.parse_args()

    with open(a.results) as f:
        rows = [r for r in json.load(f) if "throughput_tok_s" in r]
    rows.sort(key=lambda r: r["batch_size"])
    bs = [r["batch_size"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(a.title, fontsize=13, fontweight="bold")

    def setup(ax, ylabel, title):
        ax.set_xscale("log", base=2)
        ax.set_xticks(bs)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("batch size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)

    ax = axes[0]
    ax.plot(bs, [r["throughput_tok_s"] for r in rows], "o-", color="#2563eb")
    setup(ax, "tokens / s (total)", "Throughput vs batch size")

    ax = axes[1]
    ax.plot(bs, [r["median_s"] for r in rows], "o-", color="#9333ea")
    setup(ax, "seconds / request", "Latency vs batch size")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(a.out, dpi=a.dpi)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
