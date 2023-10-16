import torch
from torch import Tensor
import numpy as np

import gzip
import pickle

import random


def construct_pytorch_set(numpy_tuple_set: tuple[np.ndarray, np.ndarray]) -> tuple[Tensor, Tensor]:
    instances = Tensor(numpy_tuple_set[0])

    # Transform the targets given as the correct digit of an instance
    # in a one hot tensor for each instance
    no_instances = len(numpy_tuple_set[1])
    targets = torch.eye(10)[numpy_tuple_set[1][0:no_instances], :]

    return instances, targets


def extract_sets():
    with gzip.open('mnist.pkl.gz', 'rb') as fd:
        training_set, validation_set, test_set = pickle.load(fd, encoding='latin')

    # Each set is a tuple
    # [0] has the number of instances of images 28x28,flattened into 784 values
    # [1] has the labels associated

    training_set = construct_pytorch_set(training_set)
    validation_set = construct_pytorch_set(validation_set)
    test_set = construct_pytorch_set(test_set)
    return training_set, validation_set, test_set


def sigmoid(z: Tensor) -> Tensor:
    return 1.0 / (1.0 + torch.exp(-z))


def train_perceptron(X: Tensor, W: Tensor, b: Tensor, y_true: Tensor, mu: float):
    # Feed forward
    y_pred = sigmoid(X @ W + b)

    # Error
    error = y_true - y_pred

    # Update
    W += mu * (X.T @ error)
    b += mu * torch.sum(error, dim=0)

    return error


def main():
    # Showing how the error changes after 1 batch update
    training_set, validation_set, test_set = extract_sets()

    batch_size = 128
    mu = 0.01

    indexes = [*range(len(training_set[1]))]
    random.shuffle(indexes)

    X = training_set[0][indexes[0:batch_size]]

    W = torch.rand((784, 10))

    b = torch.rand((10,))

    y_true = training_set[1][indexes[0:batch_size]]

    print("X shape: ", X.shape)
    print("y_true shape: ", y_true.shape)

    previous_error = train_perceptron(X, W, b, y_true, mu)
    new_error = y_true - sigmoid(X @ W + b)

    print("Previous error: ", abs(torch.sum(previous_error)))
    print("New error: ", abs(torch.sum(new_error)))


# After 1 run we got:
# X shape:  torch.Size([128, 784])
# y_true shape:  torch.Size([128, 10])
# Previous error:  tensor(1152.)
# New error:  tensor(1061.3304)
main()
