from math import sqrt
import torch
from .nn import MultilayeredNeuralNetwork


def benchmark(nn: MultilayeredNeuralNetwork, dataset):
    testing_data = dataset.randomise_testing_data().testing_data
    true_positives = 0

    for X, y in zip(testing_data[0], testing_data[1]):
        true_positives += torch.argmax(nn.predict(X.view(-1, 1))) == torch.argmax(
            y.view(-1, 1)
        )

    print(
        f"Accuracy: {true_positives} / {testing_data[2]} = {true_positives / testing_data[2] * 100:.5f}%"
    )


def train_batched_epochs(
    nn: MultilayeredNeuralNetwork,
    dataset,
    batch_size: int,
    max_epochs: int,
):
    training_data = dataset.randomise_training_data().training_data
    loss_mean = 0
    total_loss = 0

    for epoch in range(0, max_epochs):
        X = training_data[0]
        y = training_data[1]

        std_dev, mean = torch.std_mean(X)
        X = X - mean / std_dev

        current_loss = nn.train_batched(X, y, batch_size)
        loss_mean = (epoch * loss_mean + current_loss.sum()) / (epoch + 1)
        total_loss += current_loss.sum()

        print(
            f"Training epoch {epoch + 1} / {max_epochs}. Total loss: {total_loss:.2f}\tAverage loss: {loss_mean:.2f}",
            end="\r",
        )

    print()
