#!/usr/bin/env python3

import torch
import torchvision
import torch.utils.data as torch_data
from util import run, test, train, val
from nn import NeuralNetwork
from dataset import Dataset
from transforms import Flatten, ToFloat


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



if __name__ == "__main__":
    main()
