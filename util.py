import torch
def get_device(device_type: str = "auto") -> torch.device:
    """
    Return a torch.device based on the input string.

    Parameters:
        device_type (str): One of "cpu", "cuda", "mps", or "auto".

    Returns:
        torch.device: The selected device.
    """
    device_type = device_type.lower()
    
    if device_type == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("⚠️ CUDA is not available, falling back to CPU.")
            return torch.device("cpu")
    
    elif device_type == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        else:
            print("⚠️ MPS is not available, falling back to CPU.")
            return torch.device("cpu")
    
    elif device_type == "cpu":
        return torch.device("cpu")
    
    elif device_type == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    else:
        raise ValueError(f"Unsupported device type: {device_type}")