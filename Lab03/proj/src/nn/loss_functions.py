import torch

def mean_squared_error(y: torch.Tensor, y_hat: torch.Tensor, X: torch.tensor) -> torch.Tensor:
    return (y - y_hat) ** 2 / 2

def mean_squared_error_derivative(y: torch.Tensor, y_hat: torch.Tensor, X: torch.tensor) -> torch.Tensor:
    return (-2 * (y - y_hat)) / X.shape[0]

