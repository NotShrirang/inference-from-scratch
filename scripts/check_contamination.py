#!/usr/bin/env python3
"""Fire N *different* prompts concurrently at the continuous-batching endpoint and
print each completion separately.

This is the text-level correctness check the throughput benchmark can't give you:
the requests are batched together in the engine, so if there's a padding/position/
cache-aliasing bug, one request's output will bleed in content from another's. Each
printed OUTPUT should answer its OWN prompt and nothing else.

Usage:
    python scripts/check_contamination.py --model Qwen/Qwen2.5-0.5B --load -n 3
"""
import argparse
import asyncio

import httpx

# Deliberately distinct topics so contamination is obvious at a glance.
PROMPTS = [
    "In one sentence, what is the capital of France?",
    "Write the first line of a poem about the ocean.",
    "List the first five prime numbers.",
    "What color do you get by mixing blue and yellow?",
    "Name a famous physicist and one thing they discovered.",
    "Translate 'good morning' into Spanish.",
    "What is 7 multiplied by 8?",
    "Describe a cat in three words.",
    "What planet is known as the Red Planet?",
    "Give one tip for staying hydrated.",
]


async def load_model(base, model):
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(f"{base}/load", params={"model_name": model})
        r.raise_for_status()
        print(f"[load] {r.json().get('message')}\n")


async def one(client, base, endpoint, model, prompt, max_length, temperature, top_p):
    params = {
        "model_name": model, "input_text": prompt, "max_length": max_length,
        "temperature": temperature, "top_p": top_p,
    }
    text = ""
    async with client.stream("POST", f"{base}/{endpoint}", params=params) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if line.startswith("data:"):
                text += line.removeprefix("data: ")   # keep the token's own spacing
    return prompt, text


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--endpoint", default="generate-continuous-batch")
    ap.add_argument("-n", "--num", type=int, default=3, help="how many concurrent prompts")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--load", action="store_true")
    a = ap.parse_args()

    if a.load:
        await load_model(a.base, a.model)

    # distinct prompts; cycle only if you ask for more than we have
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(a.num)]

    # fire all of them at once so the engine batches them in the same forward passes
    async with httpx.AsyncClient(timeout=None) as client:
        results = await asyncio.gather(*[
            one(client, a.base, a.endpoint, a.model, p, a.max_length, a.temperature, a.top_p)
            for p in prompts
        ])

    print(f"Fired {a.num} concurrent prompts at /{a.endpoint} — each OUTPUT should match its own PROMPT:\n")
    for i, (prompt, text) in enumerate(results, 1):
        print(f"─── Request {i} " + "─" * 50)
        print(f"PROMPT: {prompt}")
        print(f"OUTPUT:{text}\n")


if __name__ == "__main__":
    asyncio.run(main())
