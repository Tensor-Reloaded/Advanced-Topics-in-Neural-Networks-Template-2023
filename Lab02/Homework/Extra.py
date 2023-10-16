import torch
from torch import Tensor
import numpy as np

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


def binary_threshold(z: Tensor) -> Tensor:
    output = torch.clone(z).detach()
    output[output < 0] = 0
    output[output > 0] = 1
    return output


def feed_forward(x: Tensor, weights: Tensor, bias: Tensor, activation_function: Callable[[Tensor], Tensor]) -> Tensor:
    # We expect a tensor of shape (n,10)
    z = x @ weights + bias
    return z if activation_function is None else activation_function(z)


def minibatch_train_perceptron(weights: Tensor, bias: Tensor, no_epochs: int, batch_size: int, learning_rate: float,
                               training_set: tuple[torch.tensor, torch.tensor],
                               activation_function: Callable[[Tensor], Tensor]):
    if weights is None:
        weights = torch.rand((784, 10))

    if bias is None:
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
        target = data_set[1][index]

        no_correctly_classified += torch.equal(one_hot_prediction, target)

    return 100.0 * no_correctly_classified / no_instances


def main(no_epochs: int, batch_size: int, learning_rate: float):
    training_set, validation_set, test_set = extract_sets()

    print("No epochs: ", no_epochs)
    print("Batch size: ", batch_size)
    print("Learning rate: ", learning_rate)

    activation_functions = [sigmoid, binary_threshold]
    names = ["Sigmoid", "Binary Threshold"]
    for index, activation_function in enumerate(activation_functions):
        name = names[index]
        weights = torch.rand((784, 10))
        bias = torch.rand((10,))

        print("\nWe are using the activation function: ", name)

        print("Accuracy on training set before training: ",
              compute_accuracy(training_set, weights, bias, activation_function), "%")

        print("Accuracy on validation set before training: ",
              compute_accuracy(validation_set, weights, bias, activation_function), "%")

        minibatch_train_perceptron(weights, bias, no_epochs, batch_size, learning_rate, training_set,
                                   activation_function)

        print("---\nAccuracy on training set after training: ",
              compute_accuracy(training_set, weights, bias, activation_function), "%")

        print("Accuracy on validation set after training: ",
              compute_accuracy(validation_set, weights, bias, activation_function), "%")

        print("Accuracy on test set: ", compute_accuracy(test_set, weights, bias, activation_function), "%")


main(no_epochs=30, batch_size=128, learning_rate=0.01)

# Perceptron for the MNIST dataset using 2 different activation functions

# One run got this results:
# No epochs:  30
# Batch size:  128
# Learning rate:  0.01
#
# We are using the activation function:  Sigmoid
# Accuracy on training set before training:  9.97 %
# Accuracy on validation set before training:  10.03 %
# ---
# Accuracy on training set after training:  91.924 %
# Accuracy on validation set after training:  91.73 %
# Accuracy on test set:  91.61 %
#
# We are using the activation function:  Binary Threshold
# Accuracy on training set before training:  9.864 %
# Accuracy on validation set before training:  9.91 %
# ---
# Accuracy on training set after training:  85.804 %
# Accuracy on validation set after training:  86.01 %
# Accuracy on test set:  85.29 %
