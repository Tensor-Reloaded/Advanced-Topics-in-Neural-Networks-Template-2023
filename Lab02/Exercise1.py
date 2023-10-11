from typing import Union
import torch

def get_normal_tensors(x: torch.Tensor) -> Union[torch.Tensor, None]:
    norms = torch.norm(x, dim=(1, 2))
    
    mean_norm = norms.mean()
    std_norm = norms.std()
    
    filtered_gradients = x[(norms >= (mean_norm - 1.5 * std_norm)) & (norms <= (mean_norm + 1.5 * std_norm))]    
    return None if len(filtered_gradients) == 0 else filtered_gradients


if __name__ == "__main__":
    gradients = torch.rand((100, 10, 256))
    result = get_normal_tensors(gradients)

    print("Filtered gradients shape:", result.shape if result is not None else None)
