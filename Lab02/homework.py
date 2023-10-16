import torch
from torch import Tensor


def sigmoid(z):
    return 1 / (1 + torch.exp(-z))


def train_perceptron(X: Tensor, W: Tensor, b: Tensor, y_true: Tensor, mu: float):
    m = X.shape[0]

    # Forward Propagation
    z = torch.mm(X, W) + b
    y_pred = sigmoid(z)
    print(y_true)
    print(y_pred)

    error = y_true - y_pred
    print(error)

    # Backward Propagation
    dW = torch.mm(X.t(), error) / m
    db = torch.sum(error, dim=0) / m

    W += mu * dW
    b += mu * db

    return W, b


if __name__ == "__main__":
    m = 5  # Number of examples
    n_inputs = 784 # Number of inputs
    n_outputs = 10 # Number of outputs
    X = torch.randn(m, n_inputs) # Input tensor
    W = torch.randn(n_inputs, n_outputs) # Weights tensor
    b = torch.zeros(n_outputs) # Initial biases initialized with 0
    y_true = torch.randint(0, 2, (m, n_outputs)).float()  # Random binary true labels
    mu = 0.1  # Learning rate

    updated_W, updated_b = train_perceptron(X, W, b, y_true, mu)

    print("Updated weights:")
    print(updated_W)
    print("Updated biases:")
    print(updated_b)