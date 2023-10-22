import torch

def mean_squared_error(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y) ** 2 / 2

def mean_squared_error_derivative(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -2 * (x - y)

