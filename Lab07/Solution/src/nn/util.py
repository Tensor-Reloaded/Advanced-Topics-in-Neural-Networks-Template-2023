import torch

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    torch.xpu = None


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mos")
    if hasattr(torch, "xpu") and torch.xpu is not None and torch.xpu.device_count() > 0:
        return torch.device("xpu")
    return torch.device("cpu")

print(get_default_device())
