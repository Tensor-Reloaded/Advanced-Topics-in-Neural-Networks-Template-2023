import torch
from torch import Tensor


def calculate_sigmoid(z):
    return 1 / (1 + torch.exp(-z))

def calculate_heavyside(z):
    return 0 if z <= 0 else 1

def do_forward_propagation(X: Tensor, W: Tensor, b: Tensor, func):
    z = torch.mm(X, W) + b

    return func(z)

def do_back_propagation(X: Tensor, W: Tensor, b: Tensor, error: Tensor):
    
    W += mu * torch.mm(X.t(), error)
    b += mu * torch.sum(error, dim=0)

    return W, b

def train_perceptron(X: Tensor, W: Tensor, b: Tensor, y_true: Tensor, mu: float):
    y_pred = do_forward_propagation(X, W, b, calculate_sigmoid)

    error = y_true - y_pred

    W, b = do_back_propagation(X, W, b, error)

    return W, b

m=10

X = torch.rand((m, 784))
W = torch.rand((784, 10))
b = torch.rand((10,))
y_true = torch.randint(0, 2, (m, 10), dtype=torch.float32)
mu = 0.02

updated_W, updated_b = train_perceptron(X, W, b, y_true, mu)

print("New weights:", updated_W)
print("New biases:", updated_b)