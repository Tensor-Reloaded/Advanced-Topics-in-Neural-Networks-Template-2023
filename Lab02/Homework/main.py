import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorboard
from torch import Tensor
import torchvision
import torchvision.datasets as datasets

torch.set_default_device('cpu')

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)


def sigmoid(z):
    return 1 / (1 + torch.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def error(y_true: Tensor, y: Tensor) -> Tensor:
    return y_true - y


def forward_pass(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    x = x.reshape(x.shape[0], 1)
    w = torch.transpose(w, 0, 1)
    z = w @ x + b
    return sigmoid(z)


def gradient_descent(x: Tensor, w: Tensor, b: Tensor, y: Tensor, y_true: Tensor, mu: float = 0.01) -> (Tensor, Tensor):
    x = x.reshape(x.shape[0], 1)
    y_true = y_true.reshape(y_true.shape[0], 1)
    err = error(y_true, y)
    x = torch.transpose(x, 0, 1)
    err_x = err @ x
    err_x = torch.transpose(err_x, 0, 1)
    w = w + mu * err_x
    b = b + mu * err
    return w, b


def train_perceptron(x: Tensor, w: Tensor, b: Tensor, y_true: Tensor, mu: float = 0.01) -> (Tensor, Tensor):
    for i in range(x.shape[0]):
        y = forward_pass(x[i], w, b)
        w, b = gradient_descent(x[i], w, b, y, y_true[i], mu)
    return w, b


# def train_batch_perceptron(x: Tensor, w: Tensor, b: Tensor, y_true: Tensor, mu: float = 0.01) -> (Tensor, Tensor):
#     batch_size = 100
#     for i in range(0, x.shape[0], batch_size):
#         y = forward_pass_batch(x[i:i + batch_size], w, b)
#         w, b = gradient_descent_batch(x[i:i + batch_size], w, b, y, y_true[i:i + batch_size], mu)
#     return w, b
#
#
# def forward_pass_batch(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
#     w = torch.transpose(w, 0, 1)
#     x = torch.transpose(x, 0, 1)
#     w_x = w @ x
#     z = w_x + b
#     z = torch.transpose(z, 0, 1)
#     return sigmoid(z)
#
#
# def gradient_descent_batch(x: Tensor, w: Tensor, b: Tensor, y: Tensor, y_true: Tensor, mu: float = 0.01) -> (Tensor, Tensor):
#     err = error(y_true, y)
#     err = torch.transpose(err, 0, 1)
#     err_x = err @ x
#     err_x = torch.transpose(err_x, 0, 1)
#     w = w + mu * err_x
#     b = b + mu * err
#     return w, b


def train(x: Tensor, w: Tensor, b: Tensor, y_true: Tensor, mu: float = 0.01, epochs: int = 1000) -> (Tensor, Tensor):
    for i in range(epochs):
        w, b = train_perceptron(x, w, b, y_true, mu)
        # w, b = train_batch_perceptron(x, w, b, y_true, mu)
        print(w.shape, b.shape)
        if i % 10 == 0:
            print("Epoch: ", i)
            print("Accuracy: ", accuracy(x, w, b, y_true))
    return w, b


def convert_to_one_hot(y: Tensor) -> Tensor:
    y_one_hot = torch.zeros(y.shape[0], 10)
    for i in range(y.shape[0]):
        y_one_hot[i][y[i]] = 1
    return y_one_hot


def test(x: Tensor, w: Tensor, b: Tensor, y_true: Tensor) -> (Tensor, Tensor):
    y = forward_pass(x, w, b)
    y_true = y_true.reshape(y_true.shape[0], 1)
    err = error(y_true, y)
    return err


def accuracy(x_test: Tensor, w: Tensor, b: Tensor, y_true: Tensor) -> float:
    correct = 0
    for x, y in zip(x_test, y_true):
        x = x.reshape(x.shape[0], 1)
        y = y.reshape(y.shape[0], 1)
        y_pred = forward_pass(x, w, b)
        if torch.argmax(y_pred) == torch.argmax(y):
            correct += 1
    return correct / x_test.shape[0]


if __name__ == '__main__':
    # m = 100
    # x = torch.rand(m, 784)
    # w = torch.rand(784, 10)
    # b = torch.rand(10, 1)
    # y_true = torch.rand(m, 10)
    # w, b = train_perceptron(x, w, b, y_true)

    x_train = mnist_trainset.data
    x_train = torch.flatten(x_train, start_dim=1)
    y_train_init = mnist_trainset.targets
    y_train = convert_to_one_hot(y_train_init)

    x_train = x_train.float()
    y_train = y_train.float()

    x_test = mnist_testset.data
    x_test = torch.flatten(x_test, start_dim=1)
    y_test_init = mnist_testset.targets
    y_test = convert_to_one_hot(y_test_init)

    x_test = x_test.float()
    y_test = y_test.float()

    w = torch.rand(784, 10)
    b = torch.rand(10, 1)

    initial_accuracy = accuracy(x_test, w, b, y_test)
    print("Initial accuracy: ", initial_accuracy)

    w, b = train(x_train, w, b, y_train, mu=0.01, epochs=100)
    final_accuracy = accuracy(x_test, w, b, y_test)
    print("Final accuracy: ", final_accuracy)
