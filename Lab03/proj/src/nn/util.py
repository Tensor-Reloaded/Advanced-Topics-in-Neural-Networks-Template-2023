import typing as t
import torch
from .nn import MultilayeredNeuralNetwork


def benchmark(
    nn: MultilayeredNeuralNetwork, testing_data: t.Tuple[torch.Tensor, torch.Tensor]
):
    true_positives = 0

    for X, y in zip(testing_data[0], testing_data[1]):
        true_positives += torch.argmax(nn.predict(X.view(-1, 1))) == torch.argmax(
            y.view(-1, 1)
        )

    print(f"True positives: {true_positives}")
    print(
        f"Accuracy: {true_positives} / {testing_data[2]} = {true_positives / testing_data[2] * 100:.5f}%"
    )


def train_individual_epochs(
    nn: MultilayeredNeuralNetwork,
    training_data: t.Tuple[torch.Tensor, torch.Tensor],
    max_epochs: int,
):
    total_loss = 0

    for i in range(0, max_epochs):
        print(f"Training epoch {i + 1} / {max_epochs}. Loss: {total_loss}", end="\r")

        for X, y in zip(training_data[0], training_data[1]):
            total_loss += nn.train(X, y)

    print()


def train_batched_epochs(
    nn: MultilayeredNeuralNetwork,
    training_data: t.Tuple[torch.Tensor, torch.Tensor],
    batch_size: int,
    max_epochs: int,
):
    total_loss = 0

    for i in range(0, max_epochs):
        print(f"Training epoch {i + 1} / {max_epochs}. Loss: {total_loss}", end="\r")
        X = training_data[0]
        y = training_data[1]

        total_loss += nn.train_batched(X, y, batch_size)

    print()
