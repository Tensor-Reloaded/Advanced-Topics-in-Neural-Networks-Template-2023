import torch


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))


def sigmoid_derivative(x: torch.Tensor) -> torch.Tensor:
    return sigmoid(x) * (1 - sigmoid(x))


def reLU(x: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.zeros(x.shape), x)


def reLU_derivative(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 0, torch.tensor(1.0), torch.tensor(0.0))
