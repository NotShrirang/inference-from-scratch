import torch
import torch.nn as nn


class QuantizationLayer(nn.Module):
    def __init__(self, layer, bits=8):
        super(QuantizationLayer, self).__init__()
        assert bits in (4, 8)
        self.bits = bits
        self.qmax = 2 ** (bits - 1) - 1
        self.out_features, self.in_features = layer.weight.shape

        bias = layer.bias.data.clone() if layer.bias is not None else None
        self.register_buffer("bias", bias)
        self._quantize(layer)

    def _quantize(self, layer):
        w = layer.weight.data.float()
        scale = (w.abs().amax(dim=1, keepdim=True) / self.qmax).clamp(min=1e-8)
        q = torch.round(w / scale).clamp(-self.qmax - 1, self.qmax).to(torch.int8)
        self.register_buffer("scale", scale)
        self.register_buffer("w_quantized", q if self.bits == 8 else self._pack_int4(q))

    @staticmethod
    def _pack_int4(q):
        q = q.reshape(-1)
        if q.numel() % 2:
            q = torch.cat([q, q.new_zeros(1)])
        lo = (q[0::2] & 0x0F).to(torch.uint8)
        hi = (q[1::2] & 0x0F).to(torch.uint8)
        return lo | (hi << 4)

    def _unpack_int4(self, n):
        lo = (self.w_quantized & 0x0F).to(torch.int8)
        hi = (self.w_quantized >> 4).to(torch.int8)
        lo = torch.where(lo >= 8, lo - 16, lo)
        hi = torch.where(hi >= 8, hi - 16, hi)
        return torch.stack([lo, hi], dim=1).reshape(-1)[:n]

    def _dequantize(self):
        if self.bits == 8:
            q = self.w_quantized
        else:
            q = self._unpack_int4(self.out_features * self.in_features)
        return q.float().reshape(self.out_features, self.in_features) * self.scale

    def forward(self, x):
        w = self._dequantize().to(x.dtype)
        return torch.nn.functional.linear(x, w, self.bias)


class QuantizedModel(nn.Module):
    def __init__(self, model, bits=8, skip=("lm_head",)):
        super(QuantizedModel, self).__init__()
        self.model = model
        self._quantize_module(self.model, bits, skip)
        self.model.eval()

    def _quantize_module(self, module, bits, skip):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and name not in skip:
                setattr(module, name, QuantizationLayer(child, bits=bits))
            else:
                self._quantize_module(child, bits, skip)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def model_bytes(module):
    return (sum(p.numel() * p.element_size() for p in module.parameters())
            + sum(b.numel() * b.element_size() for b in module.buffers()))


if __name__ == "__main__":
    import argparse
    from transformers import AutoModelForCausalLM

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Path to the model directory")
    parser.add_argument("--bits", type=int, choices=[4, 8], default=8, help="Number of bits for quantization")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model)
    before = model_bytes(model)
    quantized_model = QuantizedModel(model, bits=args.bits)
    after = model_bytes(model)
    print(f"{args.model}: {before/1e6:.1f} MB -> {after/1e6:.1f} MB " f"(int{args.bits}, per-channel) = {before/after:.2f}x smaller")
    quantized_model_path = f"{args.model}_quantized_{args.bits}bit"
    quantized_model.model.save_pretrained(quantized_model_path)
    print(f"Quantized model saved to {quantized_model_path}")