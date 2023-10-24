import typing as t
import torch
from .nn import MultilayeredNeuralNetwork


def benchmark(
    nn: MultilayeredNeuralNetwork,
    test_dataset,
):
    testing_data = test_dataset.randomise().data
    loss_mean = 0
    accuracy = 0
    index = 0

    for X, y in zip(testing_data[0], testing_data[1]):
        y_hat = nn.predict(X.view(-1, 1))

        is_true_positive = torch.argmax(y_hat) == torch.argmax(y.view(-1, 1))
        loss = nn.cost_function(y.view(-1, 1), y_hat, X)
        loss_mean = (index * loss_mean + loss.mean()) / (index + 1)
        accuracy = (index * accuracy + is_true_positive) / (index + 1)

        print(f"Testing: accuracy = {accuracy * 100:.5f}%, loss mean = {loss_mean:.5f}", end="\r")

        index += 1

    print()


def train_batched_epochs(
    nn: MultilayeredNeuralNetwork,
    dataset,
    batch_size: int,
    max_epochs: int,
):
    loss_mean = 0

    for epoch in range(0, max_epochs):
        training_data = dataset.randomise().data
        X = training_data[0]
        y = training_data[1]

        std_dev, mean = torch.std_mean(X)
        X = X - mean / std_dev

        loss = nn.train_batched(X, y, batch_size)

        loss_mean = (epoch * loss_mean + loss.mean()) / (epoch + 1)

        print(
            f"Training: epoch = {epoch + 1} / {max_epochs}, loss mean = {loss_mean:.5f}",
            end="\r",
        )

    print()
