1. Naive generation loop — load a small model, generate token-by-token in plain PyTorch with no cache. Measure tok/s. Feel it crawl.
2. Add a KV cache yourself. Measure the jump. Now you own the concept, not just the word.
3. Add batching yourself — run several prompts through your loop at once, watch throughput climb. You'll feel the memory-bandwidth amortization that all of this rests on.
4. Quantize one layer by hand — int4 weights + dequant, measure it. This is the rung that would've explained your AWQ result from the inside: you'd see the dequant overhead eat the memory savings.
5. Write one Triton kernel — a fused softmax or toy attention, compared to PyTorch. Your first real taste of the kernel level, on your terms.
6. Implement speculative decoding on a toy model — draft and verify, yourself.
7. Then read FlashAttention / vLLM's paged attention — and now the numbers mean something because you've built the primitives they optimize.