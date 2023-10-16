import torch
from torch import Tensor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorboard
from typing import Callable
import random

import gzip
import pickle


def construct_pytorch_set(numpy_tuple_set: tuple[np.ndarray, np.ndarray]) -> tuple[Tensor, Tensor]:
    instances = Tensor(numpy_tuple_set[0])

    # Transform the targets given as the correct digit of an instance
    # in a one hot tensor for each instance
    no_instances = len(numpy_tuple_set[1])
    targets = torch.eye(10)[numpy_tuple_set[1][0:no_instances], :]

    return instances, targets


def sigmoid(z: Tensor) -> Tensor:
    return 1.0 / (1.0 + torch.exp(-z))


def binary_threshold(z: Tensor) -> Tensor:
    output = torch.tensor(z)
    output[output < 0] = 0
    output[output > 0] = 1
    return output


def feed_forward(x: Tensor, weights: Tensor, bias: Tensor, activation_function: Callable[[Tensor], Tensor]) -> Tensor:
    # We expect a tensor of shape (n,10)
    return activation_function(x @ weights + bias)


def minibatch_train_perceptron(no_epochs: int, batch_size: int, learning_rate: float,
                               training_set: tuple[torch.tensor, torch.tensor],
                               activation_function: Callable[[Tensor], Tensor]):
    weights = torch.rand((784, 10))
    bias = torch.rand((10,))

    no_instances = len(training_set[1])

    # Used to shuffle the instances in each epoch in order to converge faster
    indexes = [*range(no_instances)]

    while no_epochs > 0:
        random.shuffle(indexes)

        start = 0
        end = min(batch_size, no_instances)
        while start < no_instances:
            x_input = training_set[0][indexes[start:end]]

            target = training_set[1][indexes[start:end], :]

            prediction = feed_forward(x_input, weights, bias, activation_function)

            error = target - prediction

            # Update
            weights += learning_rate * (x_input.T @ error)
            bias += learning_rate * torch.sum(error, dim=0)

            start = end
            end = min(end + batch_size, no_instances)
        no_epochs -= 1

    return weights, bias


def compute_accuracy(data_set: tuple[Tensor, Tensor], weights: Tensor, bias: Tensor,
                     activation_function: Callable[[Tensor], Tensor]) -> float:
    prediction = feed_forward(data_set[0], weights, bias, activation_function)

    no_instances = len(data_set[1])

    no_correctly_classified = 0
    for index in range(no_instances):
        predicted_digit = -1
        best_value = 0
        for digit in range(10):
            if best_value < prediction[index, digit]:
                best_value = prediction[index, digit]
                predicted_digit = digit

        one_hot_prediction = torch.eye(10)[predicted_digit]

        no_correctly_classified += torch.equal(one_hot_prediction, data_set[1][index])

    return 100.0 * no_correctly_classified / no_instances


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


def main():
    training_set, validation_set, test_set = extract_sets()

    weights, bias = minibatch_train_perceptron(10, 128, 0.1, training_set, sigmoid)

    print("Accuracy on validation set: ", compute_accuracy(validation_set, weights, bias, sigmoid))


main()
