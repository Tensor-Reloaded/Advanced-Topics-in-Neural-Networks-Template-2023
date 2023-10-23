#!/usr/bin/env python3
from nn.loss_functions import (
    mean_squared_error,
    mean_squared_error_derivative,
)
from nn.activation_functions import sigmoid, sigmoid_derivative, reLU, reLU_derivative
from nn.nn import MultilayeredNeuralNetwork
from torchvision import datasets
import torch
import typing as t


def main():
    dataset = datasets.MNIST(root="./data", train=False, download=True, transform=None)
    nn = MultilayeredNeuralNetwork(
        layers=[784, 100, 10],
        learning_rate=0.002,
        activation_function=sigmoid,
        activation_function_derivative=sigmoid_derivative,
        cost_function=mean_squared_error,
        cost_function_derivative=mean_squared_error_derivative,
    )

    train_data_percentage = dataset.data.shape[0] // 4 * 3
    vectorised_labels = vectorise_labels(dataset.targets)
    vectorised_dataset = vectorise_dataset(dataset.data)
    normalised_dataset = normalise(vectorised_dataset)
    training_data, testing_data = partition(
        normalised_dataset, vectorised_labels, train_data_percentage
    )

    true_positives = 0
    for X, y in zip(testing_data[0], testing_data[1]):
        true_positives += torch.argmax(nn.predict(X)) == torch.argmax(y)

    print(f"True positives: {true_positives}")
    print(
        f"Accuracy: {true_positives} / {testing_data[2]} = {true_positives / testing_data[2] * 100:.5f}%"
    )

    max_epochs = 20
    total_loss = 0
    for i in range(0, max_epochs):
        print(f"Epoch: {i + 1} / {max_epochs}", end="\r")
        randomised_training_data = randomise(training_data)
        for X, y in zip(randomised_training_data[0], randomised_training_data[1]):
            total_loss += nn.train(X, y)
    print()

    true_positives = 0
    for X, y in zip(testing_data[0], testing_data[1]):
        true_positives += torch.argmax(nn.predict(X)) == torch.argmax(y)

    print(f"True positives: {true_positives}")
    print(
        f"Accuracy: {true_positives} / {testing_data[2]} = {true_positives / testing_data[2] * 100:.5f}%"
    )


def vectorise_labels(labels: torch.Tensor) -> torch.Tensor:
    unique_labels = torch.unique(labels).tolist()
    data = torch.Tensor(len(labels), len(unique_labels))

    for i in range(0, len(labels)):
        data[i][labels[i]] = 1

    return data.view(data.shape[0], -1, 1)


def vectorise_dataset(dataset: torch.Tensor) -> torch.Tensor:
    return (
        dataset.view(dataset.shape[0], -1)
        .type(torch.float32)
        .view(dataset.shape[0], -1, 1)
    )


def normalise(dataset: torch.Tensor) -> torch.Tensor:
    return dataset / torch.max(dataset)


def partition(
    dataset: torch.Tensor, labels: torch.Tensor, train_data_count: int
) -> t.Tuple[
    t.Tuple[torch.Tensor, torch.Tensor, int], t.Tuple[torch.Tensor, torch.Tensor, int]
]:
    training_data = [
        dataset[:train_data_count],
        labels[:train_data_count],
        train_data_count,
    ]
    testing_data = [
        dataset[train_data_count:],
        labels[train_data_count:],
        dataset.shape[0] - train_data_count,
    ]

    return training_data, testing_data


def randomise(
    dataset: t.Tuple[torch.Tensor, torch.Tensor, int]
) -> t.Tuple[torch.Tensor, torch.Tensor, int]:
    indexes = torch.randperm(dataset[0].shape[0])
    random_data = dataset[0][indexes]
    random_labels = dataset[1][indexes]

    return random_data, random_labels, dataset[2]


if __name__ == "__main__":
    main()
