import torch

def clear_cuda_cache():
    """Clears the CUDA cache to free up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_device():
    """Returns the appropriate device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")