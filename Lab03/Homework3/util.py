import torch
from torch import Tensor


def sigmoid(z: Tensor) -> Tensor:
    return 1 / (1 + torch.exp(-z))


def sigmoid_derivative(z: Tensor) -> Tensor:
    return sigmoid(z) * (1 - sigmoid(z))


def error(y: Tensor, y_true: Tensor) -> Tensor:
    return y - y_true


def softmax(z: Tensor) -> Tensor:
    return torch.exp(z) / torch.sum(torch.exp(z))


def softmax_derivative(z: Tensor) -> Tensor:
    return softmax(z) * (1 - softmax(z))


def cross_entropy_error(y: Tensor, y_true: Tensor) -> Tensor:
    return torch.nn.functional.cross_entropy(y, y_true)


def convert_to_one_hot(y):
    one_hot = torch.zeros(y.size(0), 10)
    for i in range(y.size(0)):
        one_hot[i][y[i]] = 1
    return one_hot
