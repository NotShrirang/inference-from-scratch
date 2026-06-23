#!/usr/bin/env python3
"""Benchmark /generate-batch: throughput vs batch size (static batching, rung 3).

Unlike the streaming benchmark (which sweeps concurrency), this sends a single
request of N prompts and sweeps N. The endpoint is non-streaming, so the metric
is total throughput — output tokens (= batch * max_length) / wall — which should
climb with batch size as the cost of loading weights is amortized across more
tokens, then plateau when the GPU saturates.

Request shape: model_name and max_length go as query params; the prompt list is
sent as repeated `input_texts` query params (matches `input_texts: list` in the
endpoint). If your endpoint reads the list from the JSON body instead, pass
--body and it will send the list as the JSON payload.

Usage:
    python scripts/benchmark_batch.py --model Qwen/Qwen2.5-0.5B --load \
        --batch-sizes 1,2,4,8,16,32 --max-length 64 --repeats 3
"""
import argparse
import json
import statistics
import time

import httpx

# Varied-length prompts so batches exercise padding, not identical rows.
PROMPTS = [
    "Explain how a KV cache speeds up autoregressive decoding.",
    "Write a haiku about GPUs.",
    "What is batching in LLM inference?",
    "Describe memory bandwidth in one sentence.",
    "Summarize why quantization saves memory.",
    "List three benefits of speculative decoding.",
    "Define throughput versus latency.",
    "Why does attention scale quadratically with sequence length?",
]


def make_batch(n):
    return [PROMPTS[i % len(PROMPTS)] for i in range(n)]


def load_model(base, model):
    with httpx.Client(timeout=None) as client:
        r = client.post(f"{base}/load", params={"model_name": model})
        r.raise_for_status()
        print(f"[load] {r.json().get('message')}")


def run_batch(client, base, model, batch_size, max_length, use_body):
    prompts = make_batch(batch_size)
    t0 = time.perf_counter()
    if use_body:
        params = {"model_name": model, "max_length": max_length}
        r = client.post(f"{base}/generate-batch", params=params, json=prompts)
    else:
        params = {"model_name": model, "max_length": max_length}
        for p in prompts:
            params.setdefault("input_texts", []).append(p)
        r = client.post(f"{base}/generate-batch", params=params)
    wall = time.perf_counter() - t0
    return wall, r.status_code, (r.text[:200] if r.status_code != 200 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--body", action="store_true",
                    help="send prompts as JSON body instead of repeated query params")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    ap.add_argument("--repeats", type=int, default=3, help="runs per batch size; median is reported")
    ap.add_argument("--query", action="store_true",
                    help="send prompts as repeated query params instead of the default JSON body")
    ap.add_argument("--out", default="bench_batch_results.json")
    a = ap.parse_args()

    if a.load:
        load_model(a.base, a.model)

    rows = []
    hdr = f"{'batch':>6} {'median_s':>9} {'tok/s':>9} {'tok/s/seq':>10} {'errors':>7}"
    print(f"\nBenchmarking {a.base}/generate-batch  (model={a.model}, max_length={a.max_length})")
    print(hdr)
    print("-" * len(hdr))
    with httpx.Client(timeout=None) as client:
        for b in (int(x) for x in a.batch_sizes.split(",")):
            walls, errors, sample = [], 0, None
            for _ in range(a.repeats):
                wall, status, err = run_batch(client, a.base, a.model, b, a.max_length, use_body=a.body)
                if status == 200:
                    walls.append(wall)
                else:
                    errors += 1
                    sample = err
            if not walls:
                print(f"{b:>6} {'-':>9} {'-':>9} {'-':>10} {errors:>7}  (e.g. {sample})")
                rows.append({"batch_size": b, "errors": errors, "sample_error": sample})
                continue
            med = statistics.median(walls)
            tok_s = (b * a.max_length) / med
            row = {
                "batch_size": b,
                "median_s": round(med, 4),
                "throughput_tok_s": round(tok_s, 2),
                "tok_s_per_seq": round(tok_s / b, 2),
                "errors": errors,
            }
            rows.append(row)
            print(f"{b:>6} {med:>9.4f} {tok_s:>9.1f} {tok_s / b:>10.2f} {errors:>7}")

    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
