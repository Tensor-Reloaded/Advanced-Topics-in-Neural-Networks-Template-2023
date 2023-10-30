#!/usr/bin/env python3
import time
import torch
import torchvision
import torch.utils.data as torch_data
from transforms import Flatten, ToFloat
from dataset import Dataset
from nn import NeuralNetwork
from nn_util import run, test, train, val
from util import Timer
from graph import graph


def main():
    timer = Timer()
    transformations = [
        torchvision.transforms.RandomInvert(p=0.2),
        torchvision.transforms.GaussianBlur(kernel_size=7),
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

    print(f"Loading: {timer()}s for {len(dataset)} instances")

    model = NeuralNetwork(image_size=dataset.get_image_size())
    optimiser = torch.optim.SGD(model.parameters(), lr=0.03)
    loss_function = torch.nn.MSELoss()

    epochs = 20
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

    timer = Timer()
    test(model=model, test_dataloader=test_dataloader)
    training_loss_means, validation_loss_means = run(
        train=train_fn, val=val_fn, epochs=epochs
    )
    test(model=model, test_dataloader=test_dataloader)

    print(f"Running: {timer()}s for {epochs} epochs")

    graph("Training loss means", training_loss_means)
    graph("Validation loss means", validation_loss_means)

if __name__ == "__main__":
    main()
