import torch
import torch.nn.functional as F


def clear_cuda_cache():
    """Clears the CUDA cache to free up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_device():
    """Returns the appropriate device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> torch.Tensor:
    """
    Sample token(s) from logits using temperature and nucleus (top-p) sampling.

    Args:
        logits: (..., vocab_size)
        temperature: > 0
        top_p: (0, 1]

    Returns:
        Tensor of sampled token ids with shape (...)
    """

    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    logits = logits / temperature

    if top_p >= 1.0:
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

    filtered_probs = F.softmax(sorted_logits, dim=-1)
    sampled = torch.multinomial(filtered_probs, num_samples=1)

    return sorted_indices.gather(-1, sampled).squeeze(-1)


def sample_next_token_batched(logits, temperature, top_p):
      logits = logits / temperature
      probs = torch.softmax(logits, dim=-1)

      sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
      cumsum = sorted_probs.cumsum(dim=-1)
      remove = cumsum > top_p
      remove[..., 1:] = remove[..., :-1].clone()
      remove[..., 0] = False
      sorted_probs[remove] = 0.0
      sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

      choice = torch.multinomial(sorted_probs, 1)
      return sorted_idx.gather(-1, choice).squeeze(-1)
