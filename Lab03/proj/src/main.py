#!/usr/bin/env python3
from dataset.mnist import MNIST
from nn.util import benchmark, train_batched_epochs
from nn.loss_functions import (
    mean_squared_error,
    mean_squared_error_derivative,
)
from nn.activation_functions import sigmoid, sigmoid_derivative, reLU, reLU_derivative
from nn.nn import MultilayeredNeuralNetwork
import torch


def main():
    cpu_tests()
    gpu_tests()


def cpu_tests():
    abstract_tests(device=torch.device("cpu"))


def gpu_tests():
    abstract_tests(device=torch.device("cuda"))


def abstract_tests(device: torch.device) -> None:
    dataset = MNIST(train_data_percentage=0.75, device=device)
    nn = MultilayeredNeuralNetwork(
        layers=[784, 100, 10],
        learning_rate=0.002,
        activation_function=sigmoid,
        activation_function_derivative=sigmoid_derivative,
        cost_function=mean_squared_error,
        cost_function_derivative=mean_squared_error_derivative,
        device=device,
    )
    testing_data = dataset.testing_data
    batch_size = 500
    max_epochs = 100

    benchmark(nn, testing_data)
    train_batched_epochs(
        nn, dataset.randomise_training_data().training_data, batch_size, max_epochs
    )
    benchmark(nn, testing_data)


if __name__ == "__main__":
    main()
