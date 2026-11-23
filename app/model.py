from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F

from app import logger
from app.api.utils import sample_next_token


class ChatModel(nn.Module):
    def __init__(self, model_name, device='cuda', dtype=torch.bfloat16):
        super(ChatModel, self).__init__()
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
        self.pad_token = int(self.tokenizer(self.tokenizer.pad_token, return_tensors="pt").input_ids[0][0].item()) if self.tokenizer.pad_token else None
        self.eos_token = int(self.tokenizer(self.tokenizer.eos_token, return_tensors="pt").input_ids[0][0].item()) if self.tokenizer.eos_token else None
        logger.info(f"ChatModel initialized with {model_name}")

    def __str__(self):
        return f"ChatModel({self.model_name})"

    @torch.no_grad()
    def forward(self, inputs, past_key_values=None):
        # no_grad is essential for serving: without it every forward builds an autograd
        # graph, and since the KV cache is a graph output fed back in each step, the whole
        # graph + activations stays pinned and grows every decode token -> OOM.
        outputs = self.model(**inputs, past_key_values=past_key_values, use_cache=True)
        return outputs.logits, outputs.past_key_values

    def generate(self, x, max_length=50, temperature=0.6, top_p=0.9):
        past_key_values = None
        for _ in range(max_length):
            logits, past_key_values = self.forward(x, past_key_values=past_key_values)
            next_token_id = sample_next_token(logits[:, -1, :], temperature=temperature, top_p=top_p).item()
            x += self.tokenizer.decode(next_token_id)
            logger.info(f"Generated so far: {x}")
        return x
    
    def generate_batch(self, x_batch, max_length=50, temperature=0.6, top_p=0.9):
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        cur = self.tokenizer(x_batch, return_tensors="pt", padding=True).to(self.device)
        out_ids = [[] for _ in x_batch]
        past_key_values = None
        for _ in range(max_length):
            logits, past_key_values = self.forward(cur, past_key_values=past_key_values)
            next_ids = sample_next_token(logits[:, -1, :], temperature=temperature, top_p=top_p)
            for i, tid in enumerate(next_ids.tolist()):
                out_ids[i].append(tid)
            attn = torch.cat(
                [cur["attention_mask"],
                torch.ones((len(x_batch), 1), dtype=cur["attention_mask"].dtype, device=self.device)],
                dim=-1,
            )
            cur = {"input_ids": next_ids.unsqueeze(-1), "attention_mask": attn}

        completions = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        return [p + c for p, c in zip(x_batch, completions)]
