#!/usr/bin/env python3
"""Plot the closed-loop concurrency sweep into the three README figures (§9).

Reads bench_c{N}.json produced by `vllm bench serve --save-result` and emits:
  1. throughput vs concurrency        (saturation)
  2. p99 latency vs concurrency       (the knee)
  3. throughput (x) vs p99 latency (y) (the Pareto frontier — the key figure)

Usage:
    python plot_sweep.py                      # reads ./bench_c*.json, writes ./plots/
    python plot_sweep.py --glob 'bench_c*.json' --outdir plots
"""
import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")  # headless; no display needed
import matplotlib.pyplot as plt


def load_runs(pattern):
    """Load every bench JSON matching pattern, sorted by max_concurrency."""
    runs = []
    for path in glob.glob(pattern):
        with open(path) as f:
            d = json.load(f)
        # max_concurrency is the x-axis; fall back to filename if absent.
        c = d.get("max_concurrency")
        if c is None:
            m = re.search(r"c(\d+)", os.path.basename(path))
            c = int(m.group(1)) if m else None
        d["_concurrency"] = c
        d["_path"] = path
        runs.append(d)
    runs.sort(key=lambda r: r["_concurrency"])
    if not runs:
        raise SystemExit(f"No files matched {pattern!r}")
    return runs


def annotate_points(ax, xs, ys, labels):
    """Tag each marker with its concurrency level."""
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(f"c={lab}", (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=8, color="#555")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="bench_c*.json")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    runs = load_runs(args.glob)
    os.makedirs(args.outdir, exist_ok=True)

    conc = [r["_concurrency"] for r in runs]
    out_tps = [r["output_throughput"] for r in runs]
    tot_tps = [r["total_token_throughput"] for r in runs]
    p99_ttft = [r["p99_ttft_ms"] for r in runs]
    p99_e2el = [r["p99_e2el_ms"] for r in runs]

    model = runs[0].get("model_id", "")
    subtitle = f"{model}\nrandom 1024 in / 128 out, temp=0, ignore-eos, closed-loop"

    # --- Figure 1: throughput vs concurrency (saturation) ---------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(conc, out_tps, "o-", label="output tok/s", color="#2563eb")
    ax.plot(conc, tot_tps, "s--", label="total tok/s (in+out)", color="#9333ea")
    annotate_points(ax, conc, out_tps, conc)
    ax.set_xscale("log", base=2)
    ax.set_xticks(conc)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("max concurrency")
    ax.set_ylabel("throughput (tokens/s)")
    ax.set_title("Throughput vs concurrency (saturation)")
    ax.figure.suptitle("")
    ax.text(0.5, 1.10, subtitle, transform=ax.transAxes, ha="center",
            fontsize=8, color="#666")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    f1 = os.path.join(args.outdir, "throughput_vs_concurrency.png")
    fig.savefig(f1, dpi=args.dpi)
    plt.close(fig)

    # --- Figure 2: p99 latency vs concurrency (the knee) ----------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(conc, p99_ttft, "o-", label="p99 TTFT", color="#dc2626")
    ax.plot(conc, p99_e2el, "s--", label="p99 end-to-end", color="#ea580c")
    annotate_points(ax, conc, p99_ttft, conc)
    ax.set_xscale("log", base=2)
    ax.set_xticks(conc)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("max concurrency")
    ax.set_ylabel("latency (ms)")
    ax.set_title("p99 latency vs concurrency (the knee)")
    ax.text(0.5, 1.10, subtitle, transform=ax.transAxes, ha="center",
            fontsize=8, color="#666")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    f2 = os.path.join(args.outdir, "latency_vs_concurrency.png")
    fig.savefig(f2, dpi=args.dpi)
    plt.close(fig)

    # --- Figure 3: throughput vs p99 latency (Pareto frontier) ----------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(out_tps, p99_ttft, "o-", color="#059669")
    annotate_points(ax, out_tps, p99_ttft, conc)
    ax.set_xlabel("output throughput (tokens/s)")
    ax.set_ylabel("p99 TTFT (ms)")
    ax.set_title("Pareto frontier: throughput vs p99 latency")
    ax.text(0.5, 1.10, subtitle, transform=ax.transAxes, ha="center",
            fontsize=8, color="#666")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    f3 = os.path.join(args.outdir, "pareto_throughput_vs_latency.png")
    fig.savefig(f3, dpi=args.dpi)
    plt.close(fig)

    print("Wrote:")
    for f in (f1, f2, f3):
        print(" ", f)


if __name__ == "__main__":
    main()
