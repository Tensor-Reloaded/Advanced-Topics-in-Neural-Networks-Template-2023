import torch
from torch import Tensor
from typing import Tuple

FEATURES = 784
INPUTS = 10
PERCEPTRONS = 10

def generate_inputs() -> Tensor:
    """Generate the input instance tensor with n features and m entries."""
    return (10 - 1) * torch.rand((INPUTS, FEATURES)) + 1

def generate_weights() -> Tensor:
    """Generate and return weights for all perceptrons."""
    return (0.01 - (-0.01)) * torch.rand((FEATURES, PERCEPTRONS)) -0.01

def generate_biases() -> Tensor:
    """Generate and return a biases tensor, one for each perceptron."""
    return (0.01 - (-0.01)) * torch.rand((PERCEPTRONS)) - 0.01

def generate_true_labels() -> Tensor:
    """Generate the correct labels for each entry."""
    return (10 - 1) * torch.rand((INPUTS, PERCEPTRONS)) + 1

def forward(X: Tensor, W:Tensor, B: Tensor):
    return X @ W + B

def activate(X: Tensor) -> Tensor:
    return X.sigmoid()

def gradients(X: Tensor, Y: Tensor, Y_hat: Tensor, Z: Tensor) -> Tuple[Tensor, Tensor]:
    # Compute the error.
    error = (Y_hat - Y)
    # Compute the gradients dE/dw = (y_hat - y) * sig(z) * (1 - sig(z)) * x.
    delta_w = ((error @ Y_hat) @ (1 - Y_hat)) @ X
    # dE/db = (y_hat - y) * sig(z) * (1 - sig(z))
    delta_b = ((error @ Y_hat)) @ (1 - Y_hat)
    return delta_w, delta_b.mean(dim=0)


def train_perceptron(X: Tensor, W: Tensor, B: Tensor, y_true: Tensor, mu: float) -> Tuple[Tensor, Tensor]:
    # 1. Forward propagation.
    Z = forward(X, W, B)

    # 2. Sigmoid activation function
    y_hat = activate(Z)

    # 3. Compute the gradients.
    delta_w, delta_b = gradients(X, y_true, y_hat, Z)

    # 4. Update the weights and biases.
    W -= mu * delta_w.T
    B -= mu * delta_b
    return W, B

def main():
    """Main entrypoint of the application."""

    # 1. Generate X -> input instance tensor with 100 entries.
    X = generate_inputs()

    # 2. Generate W -> the initial weights. 
    W = generate_weights()

    # 3. Generate B -> the initial biases for each perceptron.
    B = generate_biases()
    print(W)

    # 4. Generate y_true -> the correct labels for entries.
    y_true = generate_true_labels()

    # 5. Let the learning rate be an arbitrary value.
    mu = 0.2

    # 6. Train all 10 perceptrons with forward and backwards propagation.
    W, B = train_perceptron(X, W, B, y_true, mu)
    print(W)
    print(W.shape)
    print(B.shape)

if __name__ == "__main__":
    main()