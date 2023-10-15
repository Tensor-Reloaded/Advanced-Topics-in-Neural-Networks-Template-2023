import torch
import torchvision
import numpy as np
from torch import Tensor


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_one_perceptron(x: Tensor, w: Tensor, b: int, expected_value: Tensor, mu: float):
    z = torch.full((x.shape[0], 1), 0)
    for i in range(x.shape[0]):
        z[i] = (x[i].float() * w).sum() + b
    y = torch.full((z.shape[0], 1), 0)
    for i in range(z.shape[0]):
        y[i] = sigmoid(z[i])
    error = expected_value - y
    w_final = w + ((error * x).sum() * mu).T
    for i in range(x.shape[0]):
        b_final = b + mu * error[i]
    return w_final, b_final


def train_perceptron(x: Tensor, w: Tensor, b: Tensor, y_true: Tensor, mu: float):
    for i in range(10):
        expected_value = torch.full((x.shape[0], 1), 0)
        for j in range(x.shape[0]):
            if y_true[j][i] == 1:
                expected_value[j] = 1
        w_final, b_final = train_one_perceptron(x, w[i], b[i][0], expected_value, mu)
        w[i] = w_final
        b[i][0] = b_final
    return


def test_perceptron(x: Tensor, w: Tensor, b: Tensor, y_true: Tensor, mu: float):
    correct_guesses = 0
    for i in range(10):
        expected_value = torch.full((x.shape[0], 1), 0)
        for j in range(x.shape[0]):
            if y_true[j][i] == 1:
                expected_value[j] = 1
        correct_guesses += test_one_perceptron(x, w[i], b[i][0], expected_value, mu)
    return correct_guesses / (x.shape[0] * 10)


def test_one_perceptron(x: Tensor, w: Tensor, b: int, expected_value: Tensor, mu: float):
    z = torch.full((x.shape[0], 1), 0)
    for i in range(x.shape[0]):
        z[i] = (x[i].float() * w).sum() + b
    y = torch.full((z.shape[0], 1), 0)
    contor = 0
    for i in range(z.shape[0]):
        y[i] = sigmoid(z[i])
        if expected_value[i] == y[i]:
            contor += 1
    return contor


if __name__ == '__main__':
    mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=None)
    w = torch.rand((10, 784))
    b = torch.rand((10, 1))
    mu = 0.1
    # calculam acuratetea inainte de antrenare
    x = torch.flatten(mnist_testset.data, start_dim=1)
    y_true = torch.full((x.shape[0], 10), 0)
    for i in range(x.shape[0]):
        y_true[i][mnist_testset.targets[i]] = 1
    print('Acuratetea inaintea procesului de antrenament este: ' + str(test_perceptron(x, w, b, y_true, mu)))
    # ----------------------------------------
    x = torch.flatten(mnist_trainset.data, start_dim=1)
    y_true = torch.full((x.shape[0], 10), 0)
    for i in range(x.shape[0]):
        y_true[i][mnist_trainset.targets[i]] = 1
    train_perceptron(x, w, b, y_true, mu)
    # calculam acuratetea dupa antrenare
    x = torch.flatten(mnist_testset.data, start_dim=1)
    y_true = torch.full((x.shape[0], 10), 0)
    for i in range(x.shape[0]):
        y_true[i][mnist_testset.targets[i]] = 1
    print('Acuratetea dupa procesul de antrenament este: ' + str(test_perceptron(x, w, b, y_true, mu)))
    # ----------------------------------------