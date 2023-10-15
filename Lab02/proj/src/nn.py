import torch


def forward_propagation(
    X: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
) -> float:
    z = W.t() @ X + b
    y = activation_function(z)
    return y


def activation_function(z: float) -> float:
    return 1 / (1 + torch.exp(-z))


def backward_propagation(
    X: torch.Tensor, W: torch.Tensor, b: torch.Tensor, y_true: torch.Tensor, mu: float
):
    y = forward_propagation(X, W, b)
    error = y_true - y

    W_prime = W + mu * error.t() * X
    b_prime = b + mu * error

    return W_prime, b_prime
