import torch

def optimize_gpu_memory(device=None):
    """
    Optimizes GPU memory usage for a specific device, if provided.

    Args:
        device (torch.device or str, optional): The device to optimize GPU memory on.
            If not provided, defaults to the current active CUDA device.
    """
    if torch.cuda.is_available():
        # Set the device if specified
        if device is not None:
            torch.cuda.set_device(device)

        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        # Clear GPU cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    else:
        print("GPU is not available. Skipping GPU memory optimization.")

def print_gpu_memory(device=None):
    """
    Prints the current GPU memory usage for a specific device, if provided.

    Args:
        device (torch.device or str, optional): The device to query memory for.
            If not provided, defaults to the current active CUDA device.
    """
    if torch.cuda.is_available():
        # Set the device if specified
        if device is not None:
            torch.cuda.set_device(device)

        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory Allocated: {allocated:.2f} GB")
        print(f"GPU Memory Reserved: {reserved:.2f} GB")
    else:
        print("GPU is not available. Cannot print GPU memory usage.")