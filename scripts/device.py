"""Pick the compute device for YOLO inference: NVIDIA GPU, Apple Silicon GPU, or CPU."""
import torch


def pick_device(requested="auto"):
    """Return an Ultralytics device string.

    "auto" picks CUDA if an NVIDIA GPU is usable, else Apple's MPS, else CPU. Any other
    value ("cuda", "cuda:1", "mps", "cpu") is checked and returned as given.
    """
    req = requested.lower()
    if req == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if req.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {requested}: no usable NVIDIA GPU (is a CUDA build of PyTorch installed?)")
    if req == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("--device mps: Apple Silicon GPU not available on this machine")
    return "cuda:0" if req == "cuda" else req


def describe(device):
    """Human-readable name of the device, for logs."""
    if device.startswith("cuda"):
        return f"{device} ({torch.cuda.get_device_name(torch.device(device))})"
    return {"mps": "mps (Apple Silicon GPU)", "cpu": "cpu"}.get(device, device)
