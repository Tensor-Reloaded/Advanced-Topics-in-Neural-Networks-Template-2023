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
    if not torch.cuda.is_available():
        return

    abstract_tests(device=torch.device("cuda"))


def abstract_tests(device: torch.device) -> None:
    dataset = MNIST(train=False, device=device)
    test_dataset = MNIST(train=True, device=device)
    nn = MultilayeredNeuralNetwork(
        layers=[784, 100, 10],
        learning_rate=0.04,
        activation_function=sigmoid,
        activation_function_derivative=sigmoid_derivative,
        cost_function=mean_squared_error,
        cost_function_derivative=mean_squared_error_derivative,
        device=device,
    )
    batch_size = 1000
    max_epochs = 4000

    benchmark(nn, test_dataset)
    train_batched_epochs(nn, dataset, batch_size, max_epochs)
    benchmark(nn, test_dataset)


if __name__ == "__main__":
    main()
