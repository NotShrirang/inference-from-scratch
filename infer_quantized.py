"""Load a model quantized by quntization_script.py and run inference.

The saved dir has a standard Qwen2 config.json but a state_dict of custom
quantized buffers (w_quantized / scale), so AutoModelForCausalLM.from_pretrained
can't load it. Instead we rebuild the architecture from config, swap in the same
QuantizationLayer modules (so buffer names match), then load_state_dict.

Tokenizer comes from the original repo (the quantized dir has no tokenizer files).
"""
import argparse

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from quntization_script import QuantizedModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B_quantized_4bit")
    ap.add_argument("--tokenizer", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--bits", type=int, default=4, choices=(4, 8))
    ap.add_argument("--prompt", default="Explain what a KV cache is in one sentence.")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Build the architecture from config WITHOUT downloading fp weights (random init).
    config = AutoConfig.from_pretrained(a.path)
    base = AutoModelForCausalLM.from_config(config)

    # 2. Recreate the QuantizationLayer modules so the buffer names line up.
    QuantizedModel(base, bits=a.bits)   # mutates `base` in place

    # 3. Load the saved quantized buffers into the skeleton.
    sd = load_file(f"{a.path}/model.safetensors")
    missing, unexpected = base.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")

    base.to(device).eval()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    # 4. Generate.
    inputs = tok(a.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = base.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
