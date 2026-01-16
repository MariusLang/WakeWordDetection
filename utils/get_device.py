import torch


def get_device(user_device: str | None = None) -> torch.device:
    """
    Get the best available device for PyTorch.

    Args:
        user_device: Optional user-specified device string (e.g., 'cpu', 'cuda', 'mps')

    Returns:
        torch.device: The selected device
    """
    if user_device:
        return torch.device(user_device)
    #if torch.backends.mps.is_available():
    #    return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
