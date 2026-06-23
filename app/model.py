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
