#!/usr/bin/env python3
"""Profile the running continuous-batching service.

Resets the engine's timing counters, fires N requests through
/generate-continuous-batch, then reads back the averaged per-op profile the
engine accumulated (tokenize / prefill_fwd / decode_fwd / sample).

The engine times each op with a CUDA sync so async kernels are attributed
correctly — so the server is a bit slower while profiling, which is fine.

Usage:
    # server already running + model loaded
    python scripts/profile_service.py --model Qwen/Qwen2.5-0.5B --n 100 --concurrency 1
"""
import argparse
import asyncio

import httpx
from tqdm.asyncio import tqdm

PROMPTS = [
    "What is a KV cache?",
    "Explain how continuous batching improves LLM serving throughput.",
    "List three techniques to reduce LLM inference latency.",
    "Why is autoregressive decoding memory-bandwidth bound?",
    "Summarize the difference between prefill and decode.",
]


async def one(client, base, model, prompt, max_length):
    params = {"model_name": model, "input_text": prompt, "max_length": max_length}
    async with client.stream("POST", f"{base}/generate-continuous-batch", params=params) as r:
        async for _ in r.aiter_lines():   # drain the stream to completion
            pass


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--n", type=int, default=100, help="number of requests")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="requests in flight at once (1 = clean per-op numbers; higher = batched)")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--load", action="store_true", help="POST /load first (model must be loaded to profile)")
    a = ap.parse_args()

    async with httpx.AsyncClient(timeout=None) as client:
        if a.load:
            r = await client.post(f"{a.base}/load", params={"model_name": a.model})
            r.raise_for_status()
            print(f"[load] {r.json().get('message')}")

        # clear counters from any prior traffic
        rc = await client.get(f"{a.base}/engine-profile", params={"model_name": a.model, "reset": "true"})
        if rc.status_code != 200:
            print(f"[error] /engine-profile returned {rc.status_code}: {rc.text}\n"
                  f"Is the model loaded? Re-run with --load.")
            return

        sem = asyncio.Semaphore(a.concurrency)

        async def worker(i):
            async with sem:
                await one(client, a.base, a.model, PROMPTS[i % len(PROMPTS)], a.max_length)

        await tqdm.gather(*[worker(i) for i in range(a.n)], desc="requests", total=a.n)

        r = await client.get(f"{a.base}/engine-profile", params={"model_name": a.model})
        prof = r.json().get("profile", {})
    if not prof:
        print("[warn] empty profile — no ops were timed (did the requests actually run?).")
        return

    print(f"\nProfile over {a.n} requests (concurrency={a.concurrency}, max_length={a.max_length})\n")
    print(f"{'op':<14}{'avg ms':>12}{'count':>10}{'total ms':>14}{'% total':>10}")
    print("-" * 60)
    grand_total = sum(v["total_ms"] for v in prof.values())
    for name, v in sorted(prof.items(), key=lambda kv: kv[1]["total_ms"], reverse=True):
        share = v["total_ms"] / grand_total * 100 if grand_total else 0.0
        print(f"{name:<14}{v['avg_ms']:>12.4f}{v['count']:>10}{v['total_ms']:>14.1f}{share:>9.1f}%")
    print("-" * 60)
    print(f"{'sum':<14}{'':>12}{'':>10}{grand_total:>14.1f}{100.0:>9.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
