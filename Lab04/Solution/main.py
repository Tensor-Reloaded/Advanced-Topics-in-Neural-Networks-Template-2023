#!/usr/bin/env python3

import typing as t
import torch
from nn import NeuralNetwork
from dataset import Dataset
from transforms import RandomRotation
import torch.utils.data as torch_data


def main():
    transformations = [
        RandomRotation(-15, 15)
        # transforms.
    ]
    dataset = Dataset(
        root="../Homework Dataset", transformations=transformations, device="cpu"
    )
    model = NeuralNetwork(16384)
    optimiser = torch.optim.SGD(model.parameters(), lr=0.02)
    loss_function = torch.nn.CrossEntropyLoss()

    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [0.7, 0.15, 0.15]
    )
    train_dataloader = torch_data.DataLoader(train_dataset, batch_size=12, shuffle=True)
    test_dataloader = torch_data.DataLoader(test_dataset, batch_size=12, shuffle=True)
    current_loss_mean: t.Union[None, torch.Tensor] = None
    previous_loss_mean: t.Union[None, torch.Tensor] = None

    try:
        for epoch in range(0, 1000):
            for training_image_set in train_dataloader:
                y_hat = model(training_image_set[0])
                loss = loss_function(y_hat, training_image_set[1])
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                for validation_image_set in validation_dataset:
                    y_hat = model(validation_image_set[0])
                    loss = loss_function(y_hat, validation_image_set[1])

                    if current_loss_mean == None:
                        current_loss_mean = loss.mean()
                    else:
                        current_loss_mean = (
                            epoch * current_loss_mean + loss.mean()
                        ) / (epoch + 1)

                if (
                    previous_loss_mean is not None
                    and current_loss_mean > previous_loss_mean
                ):
                    raise StopIteration

                previous_loss_mean = current_loss_mean.mean()

            print(
                f"Training epoch {epoch}: previous mean loss = {previous_loss_mean}, current mean loss = {current_loss_mean}",
                end="\r",
            )
    except StopIteration:
        pass

    print()

    accuracy = None
    for entry, test_image_set in enumerate(test_dataset):
        y_hat = model(test_image_set[0])
        is_accurate = y_hat == test_image_set[1]

        if accuracy is None:
            accuracy = float(is_accurate)
        else:
            accuracy = (entry * accuracy + float(is_accurate)) / (entry + 1)

        print(f"Testing: Accuracy = {accuracy * 100:.2f}%", end="\r")

    print()


if __name__ == "__main__":
    main()
