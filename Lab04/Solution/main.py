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
        RandomRotation(min_angle=-15, max_angle=15),
        torchvision.transforms.RandomResizedCrop(size=(64, 64), antialias=True),
        torchvision.transforms.Grayscale(),
        Flatten(),
        ToFloat(),
    ]
    dataset = Dataset(
        root="../Homework Dataset", transformations=transformations, device="cpu"
    )
    model = NeuralNetwork(image_size=4096)
    optimiser = torch.optim.SGD(model.parameters(), lr=0.02)
    loss_function = torch.nn.MSELoss()

    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [0.7, 0.15, 0.15]
    )
    train_dataloader = torch_data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = torch_data.DataLoader(test_dataset, batch_size=64, shuffle=True)
    current_loss_mean: t.Union[None, torch.Tensor] = None
    previous_loss_mean: t.Union[None, torch.Tensor] = None

    try:
        model.train()
        for epoch in range(0, 50):
            for training_image_set in train_dataloader:
                optimiser.zero_grad()
                y_hat = model(training_image_set[0])
                loss = loss_function(y_hat, training_image_set[1])
                loss.backward()
                optimiser.step()

            for validation_image_set in validation_dataset:
                y_hat = model(validation_image_set[0])
                loss = loss_function(y_hat, validation_image_set[1])

                if current_loss_mean == None:
                    current_loss_mean = loss.mean().item()
                else:
                    current_loss_mean = (
                        epoch * current_loss_mean + loss.mean().item()
                    ) / (epoch + 1)

            print(
                f"Training epoch {epoch}: previous mean loss = {previous_loss_mean}, current mean loss = {current_loss_mean}",
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

    model.eval()
    accuracy = None
    # accuracy_threshold = 0.95
    for i, test_image_set in enumerate(test_dataloader):
        y_hat = model(test_image_set[0])
        # similarity = torch.nn.functional.cosine_similarity(y_hat, test_image_set[1])
        # similarity_mean = similarity.mean().item()
        # is_accurate = similarity_mean > accuracy_threshold
        mask = y_hat == test_image_set[1]
        current_accuracy = mask.float().mean().item()


        if accuracy is None:
            accuracy = current_accuracy
        else:
            accuracy = (i * accuracy + current_accuracy) / (i + 1)

        print(f"Testing: Accuracy = {accuracy * 100:.2f}%", end="\r")

    print()


if __name__ == "__main__":
    main()
