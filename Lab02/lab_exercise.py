from typing import Union
from torch import Tensor
import torch


def get_normal_tensors(x: Tensor) -> Union[Tensor, None]:
    norms = torch.norm(x, dim=(1,2))
    print(norms)

    mean_norm = torch.mean(norms)
    std_dev = torch.std(norms)
    print(f"Mean: {mean_norm}")
    print(f"Standard deviation: {std_dev}")

    min_threshold = mean_norm - 1.5 * std_dev
    max_threshold = mean_norm + 1.5 * std_dev

    valid_indices = torch.where((norms >= min_threshold) & (norms <= max_threshold))
    print(f"Valid indices: {valid_indices[0].tolist()}")

    if valid_indices[0].shape[0] > 0:
        valid_tensors = x[valid_indices]
        return valid_tensors
    else:
        return None


get_normal_tensors(torch.rand((100, 10, 256)))