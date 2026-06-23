from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F

from app import logger


class ChatModel(nn.Module):
    def __init__(self, model_name, device='cuda'):
        super(ChatModel, self).__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        logger.info(f"ChatModel initialized with {model_name}")

    def forward(self, inputs, past_key_values=None):
        # inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs, past_key_values=past_key_values, use_cache=True)
        return outputs.logits, outputs.past_key_values

    def generate(self, x, max_length=50):
        past_key_values = None
        for _ in range(max_length):
            logits, past_key_values = self.forward(x, past_key_values=past_key_values)
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
            x += self.tokenizer.decode(next_token_id)
            logger.info(f"Generated so far: {x}")
        return x
    
    def generate_batch(self, x_batch, max_length=50):
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        cur = self.tokenizer(x_batch, return_tensors="pt", padding=True).to(self.device)
        out_ids = [[] for _ in x_batch]
        past_key_values = None
        for _ in range(max_length):
            logits, past_key_values = self.forward(cur, past_key_values=past_key_values)
            next_ids = torch.argmax(logits[:, -1, :], dim=-1)
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
