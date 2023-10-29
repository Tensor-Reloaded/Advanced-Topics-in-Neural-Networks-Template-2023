#!/usr/bin/env python3

import typing as t
import torch
from nn import NeuralNetwork
from dataset import Dataset
from transforms import Flatten, RandomRotation, ToFloat
import torchvision
import torch.utils.data as torch_data


def main():
    transformations = [
        # RandomRotation(min_angle=-15, max_angle=15),
        # torchvision.transforms.RandomResizedCrop(size=(64, 64), antialias=True),
        torchvision.transforms.Grayscale(),
        Flatten(),
        ToFloat(),
    ]
    dataset = Dataset(
        root="../Homework Dataset", transformations=transformations, device="cpu"
    )
    model = NeuralNetwork(image_size=16384)
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_function = torch.nn.MSELoss()

    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [0.7, 0.15, 0.15]
    )
    train_dataloader = torch_data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_dataloader = torch_data.DataLoader(
        validation_dataset, batch_size=64, shuffle=True
    )
    test_dataloader = torch_data.DataLoader(test_dataset, batch_size=64, shuffle=True)

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
    run(train=train_fn, val=val_fn, epochs=30)
    test(model=model, test_dataloader=test_dataloader)


def run(
    train,
    val,
    epochs: int,
):
    current_loss_mean: t.Union[None, torch.Tensor] = None
    previous_loss_mean: t.Union[None, torch.Tensor] = None

    try:
        for epoch in range(0, epochs):
            train()
            current_loss_mean = val(current_loss_mean, epoch)

            print(
                f"Training epoch {epoch + 1}: previous mean loss = {previous_loss_mean}, current mean loss = {current_loss_mean}",
                end="\r",
            )

            # if (
            #     previous_loss_mean is not None
            #     and current_loss_mean > previous_loss_mean
            # ):
            #     raise StopIteration

            previous_loss_mean = current_loss_mean

    except StopIteration:
        pass

    print()


def train(
    model,
    optimiser,
    loss_function,
    train_dataloader,
):
    def fn():
        model.train()

        for training_image_set in train_dataloader:
            optimiser.zero_grad()
            y_hat = model(training_image_set[0])
            loss = loss_function(y_hat, training_image_set[1])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimiser.step()

    return fn


def val(
    model,
    loss_function,
    validation_dataloader,
):
    def fn(
        current_loss_mean,
        epoch,
    ):
        model.eval()

        for validation_image_set in validation_dataloader:
            y_hat = model(validation_image_set[0])
            loss = loss_function(y_hat, validation_image_set[1])

            if current_loss_mean == None:
                current_loss_mean = loss.mean().item()
            else:
                current_loss_mean = (epoch * current_loss_mean + loss.mean().item()) / (
                    epoch + 1
                )

        return current_loss_mean

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
