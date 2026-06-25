"""Benchmark fp16 vs int8 vs int4 (rung 4): weight size, peak VRAM, decode tok/s.

The point of this benchmark is the *negative* result: int8/int4 shrink stored
weights and VRAM, but decode does NOT get faster — every forward dequantizes (and
int4 also unpacks) on the fly, so latency is flat or worse. That gap is what a
fused kernel (rung 5) exists to close.

Each variant is built, measured, then freed before the next so all three fit in
~8 GB. fp16 is the baseline (no quantization).

Usage:
    python benchmark_quantized.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --variants fp16,int8,int4 --max-new-tokens 128 --reps 3
"""
import argparse
import gc
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quntization_script import QuantizedModel, model_bytes

PROMPT = "Explain how a KV cache speeds up autoregressive decoding."


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def build(variant, model_id, device):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    if variant != "fp16":
        QuantizedModel(model, bits=int(variant[3:]))   # "int8" -> 8, "int4" -> 4, mutates in place
    return model.to(device).eval()


def measure(model, tok, device, max_new_tokens, reps):
    inputs = tok(PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():                                # warmup (kernels, dequant path)
        model.generate(**inputs, max_new_tokens=8, do_sample=False)
    sync(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(reps):
        sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        sync(device)
        times.append(time.perf_counter() - t0)

    gen = out.shape[1] - inputs["input_ids"].shape[1]
    med = sorted(times)[len(times) // 2]
    peak = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0
    return {
        "weight_mb": round(model_bytes(model) / 1e6, 1),
        "peak_vram_mb": round(peak, 1),
        "latency_s": round(med, 3),
        "tok_s": round(gen / med, 2),
        "gen_tokens": gen,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--variants", default="fp16,int8,int4")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="benchmarks/rung4_quantization.json")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)

    rows = []
    hdr = f"{'variant':>8} {'weight_MB':>10} {'peak_VRAM_MB':>13} {'latency_s':>10} {'tok/s':>8}"
    print(hdr)
    print("-" * len(hdr))
    for variant in a.variants.split(","):
        model = build(variant, a.model, device)
        r = measure(model, tok, device, a.max_new_tokens, a.reps)
        r["variant"] = variant
        rows.append(r)
        print(f"{variant:>8} {r['weight_mb']:>10.1f} {r['peak_vram_mb']:>13.1f} "
              f"{r['latency_s']:>10.3f} {r['tok_s']:>8.2f}")
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
