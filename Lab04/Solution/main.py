#!/usr/bin/env python3

import typing as t
from functools import reduce
from nn import NeuralNetwork
from dataset import Dataset
from transforms import Flatten, ToFloat
import torch
import torchvision
import torch.utils.data as torch_data


def main():
    transformations = [
        torchvision.transforms.RandomRotation(degrees=90),
        torchvision.transforms.RandomInvert(p=0.5),
        torchvision.transforms.GaussianBlur(kernel_size=15),
        torchvision.transforms.Grayscale(),
        Flatten(),
        ToFloat(),
    ]
    dataset = Dataset(
        root="../Homework Dataset", transformations=transformations, device="cpu"
    )
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [0.7, 0.15, 0.15]
    )
    train_dataloader = torch_data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_dataloader = torch_data.DataLoader(
        validation_dataset, batch_size=64, shuffle=True
    )
    test_dataloader = torch_data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    model = NeuralNetwork(image_size=dataset.get_image_size())
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_function = torch.nn.MSELoss()

    train_fn = train(
        model=model,
        optimiser=optimiser,
        loss_function=loss_function,
        train_dataloader=train_dataloader,
    )
    val_fn = val(
        model=model,
        loss_function=loss_function,
        validation_dataloader=validation_dataloader,
    )

    test(model=model, test_dataloader=test_dataloader)
    training_loss_means, validation_loss_means = run(
        train=train_fn, val=val_fn, epochs=30
    )
    test(model=model, test_dataloader=test_dataloader)


def run(
    train: t.Callable[[], t.List[torch.Tensor]],
    val: t.Callable[[], t.List[torch.Tensor]],
    epochs: int,
) -> t.Tuple[t.List[t.Tuple[int, float]], t.List[t.Tuple[int, float]]]:
    training_loss_means: t.List[t.Tuple[int, float]] = []
    validation_loss_means: t.List[t.Tuple[int, float]] = []

    try:
        for epoch in range(0, epochs):
            training_losses = train()
            validation_losses = val()

            training_loss_mean = (
                reduce(lambda x, y: x + y, training_losses) / len(training_losses)
            ).item()
            validation_loss_mean = (
                reduce(lambda x, y: x + y, validation_losses) / len(validation_losses)
            ).item()
            training_loss_means.append(training_loss_mean)
            validation_loss_means.append(validation_loss_mean)

            print(
                f"Training epoch {epoch + 1}: training loss mean = {training_loss_mean}, validation loss mean = {validation_loss_mean}",
                end="\r",
            )

    except StopIteration:
        pass

    print()

    return training_loss_means, validation_loss_means


def train(
    model,
    optimiser,
    loss_function,
    train_dataloader,
):
    def fn() -> t.List[torch.Tensor]:
        model.train()
        training_losses: t.List[torch.Tensor] = []

        for training_image_set in train_dataloader:
            optimiser.zero_grad()
            y_hat = model(training_image_set[0])
            loss = loss_function(y_hat, training_image_set[1])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimiser.step()

            training_losses.append(loss)

        return training_losses

    return fn


def val(
    model,
    loss_function,
    validation_dataloader,
):
    def fn() -> t.List[torch.Tensor]:
        model.eval()
        validation_losses: t.List[torch.Tensor] = []

        for validation_image_set in validation_dataloader:
            y_hat = model(validation_image_set[0])
            loss = loss_function(y_hat, validation_image_set[1])

            validation_losses.append(loss)

        return validation_losses

    return fn


def test(model, test_dataloader):
    model.eval()
    total = 0
    correct = 0
    accuracy_threshold = 0.2

    with torch.no_grad():
        for test_image_set in test_dataloader:
            y_hat = model(test_image_set[0])
            total += test_image_set[1].size(0)
            correct += (
                ((y_hat - test_image_set[1]).abs() < accuracy_threshold).sum().item()
            )

            print(f"Testing: Accuracy = {correct / total:.2f}%", end="\r")

    print()


if __name__ == "__main__":
    main()
