#!/usr/bin/env python3
import argparse
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from nn.util.device import get_default_device
from nn.model.model import Model
from nn.dataset.custom_dataset import CustomDataset
from util.args import parse_arguments

current_path = os.path.dirname(__file__)


def main():
    args = parse_arguments(require_weights=True)
    device = get_default_device()
    weights_path = get_weights_path(args)

    dataset = CustomDataset(
        train=False, cache=False, data_path=f"{current_path}/../data/datasets"
    )
    dataset = torch.stack(dataset.images)
    dataset = TensorDataset(dataset)
    dataloader = DataLoader(
        dataset, batch_size=100, num_workers=2, pin_memory=device == "cuda"
    )

    model = Model(device=device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.zero_grad()
    model.eval()

    for images in dataloader:
        y = images[0]
        y_hat = model(x=images[0])

        for image in range(y.shape[0]):
            plt.figure(figsize=(8, 4))

            plt.subplot(1, 2, 1)
            plt.imshow(y[image].permute(1, 2, 0))
            plt.title("Image 1")

            plt.subplot(1, 2, 2)
            plt.imshow(y_hat[image].permute(1, 2, 0).detach().numpy(), cmap="gray")
            plt.title("Image 2")

            plt.savefig(f"{current_path}/../data/outputs/output_{image}.png")
            plt.show()


def get_weights_path(args: argparse.Namespace):
    path = f"{current_path}/../{args.weights}"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Weights path not found at {path}")

    return path


if __name__ == "__main__":
    main()
