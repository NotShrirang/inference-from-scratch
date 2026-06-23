from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F

from app import logger


class ChatModel(nn.Module):
    def __init__(self, model_name):
        super(ChatModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        logger.info(f"ChatModel initialized with {model_name}")

    def forward(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.logits

    def generate(self, x, max_length=50):
        for _ in range(max_length):
            outputs = self.forward(x)
            next_token_id = torch.argmax(outputs[:, -1, :], dim=-1).item()
            x += self.tokenizer.decode(next_token_id)
            logger.info(f"Generated so far: {x}")
        return x
