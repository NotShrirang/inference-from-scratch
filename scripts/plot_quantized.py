#!/usr/bin/env python3
"""Plot benchmark_quantized.py results: memory and decode speed per variant.

Two panels make the rung-4 point: memory bars fall sharply fp16 -> int8 -> int4,
while the tok/s bars do NOT rise (the dequant/unpack tax). Memory win, no speed win.

Usage:
    python scripts/plot_quantized.py --results benchmarks/rung4_quantization.json \
        --out benchmarks/rung4_quantization.png
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="benchmarks/rung4_quantization.json")
    ap.add_argument("--out", default="benchmarks/rung4_quantization.png")
    ap.add_argument("--title", default="Rung 4 — quantization: memory vs speed")
    ap.add_argument("--dpi", type=int, default=120)
    a = ap.parse_args()

    with open(a.results) as f:
        rows = json.load(f)
    variants = [r["variant"] for r in rows]
    x = range(len(variants))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(a.title, fontsize=13, fontweight="bold")

    # memory: grouped bars for stored weights vs peak VRAM
    w = 0.38
    axes[0].bar([i - w / 2 for i in x], [r["weight_mb"] for r in rows], w,
                label="stored weights", color="#2563eb")
    axes[0].bar([i + w / 2 for i in x], [r["peak_vram_mb"] for r in rows], w,
                label="peak VRAM", color="#93c5fd")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(variants)
    axes[0].set_ylabel("MB")
    axes[0].set_title("Memory (lower = better)")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    # speed: tok/s bars
    axes[1].bar(list(x), [r["tok_s"] for r in rows], 0.5, color="#dc2626")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(variants)
    axes[1].set_ylabel("decode tok/s")
    axes[1].set_title("Speed (higher = better) — note it doesn't improve")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(a.out, dpi=a.dpi)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
